"""A tiny OpenAI-compatible chat client, configured through environment variables.
Defaults to the Kimi Coding Plan endpoint, since that is what this example ships with;
any other OpenAI-compatible `/chat/completions` endpoint works too by overriding the
base URL and model:

    OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (see README for the
                      api.kimi.ai alternative for accounts outside mainland China)
    OPENAI_MODEL      default "k3"
    OPENAI_API_KEY    bearer token (KIMI_API_KEY is also accepted)

See https://www.kimi.com/code/docs/en/ for the Kimi Coding Plan API. Kimi keeps
Python's default User-Agent; WaytoCC uses an explicit client identifier and Accept
header because its gateway rejects the default urllib request.

Every call has a timeout (default 12 s) and the whole run has a total time budget
(default 300 s) and a call cap (default 100), so the two LLM-advised planning steps
respect the remaining real-time limit; network waits do not consume the
standardized CPU budget on fair-clock platforms. A call that fails or times out is retried up to
`max_retries` times; if all of those fail, that one planning step falls back to its
rule-based answer for this night -- the next night's calls still go through normally.

Standard library only (urllib) so the example has zero third-party dependencies.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def validate_reply(reply: dict, stage: str) -> dict:
    """Reject malformed model advice before it can influence planning."""
    if not isinstance(reply, dict):
        raise ValueError("model reply must be an object")
    if stage == "fault_review":
        if set(reply) != {"report"} or not isinstance(reply["report"], bool):
            raise ValueError("fault review requires a boolean report")
    else:
        if stage not in {"forecast_plan", "feedback_review"}:
            raise ValueError("unknown LLM stage")
        if set(reply) != {"avoid_directions", "duration_scale"}:
            raise ValueError("night advice has unexpected fields")
        directions, scale = reply["avoid_directions"], reply["duration_scale"]
        if not isinstance(directions, list) or any(not isinstance(d, str) or d not in DIRECTIONS for d in directions):
            raise ValueError("invalid compass directions")
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.7 <= scale <= 1.4:
            raise ValueError("invalid duration scale")
    return reply


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key() -> None:
    """Raise MissingAPIKeyError if neither OPENAI_API_KEY nor KIMI_API_KEY is set.
    Called once at process startup, before reading anything from stdin."""
    if not (os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()):
        raise MissingAPIKeyError("missing API key: set OPENAI_API_KEY")


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout_seconds: float = 12.0,
                 total_budget_seconds: float = 300.0, max_calls: int = 100, max_retries: int = 3,
                 offline: bool = False):
        self.log = log
        self.offline = offline
        self.base_url = "" if offline else os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        self.api_key = "" if offline else os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()
        self.model = "offline" if offline else os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self.call_timeout_seconds = (call_timeout_seconds if offline else
                                     min(45.0, max(1.5, float(os.environ.get("OPENAI_TIMEOUT_SECONDS") or call_timeout_seconds))))
        self.total_budget_seconds = total_budget_seconds
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.spent_seconds = 0.0
        self.calls_made = 0

    def _budget_left(self, wallclock_remaining_seconds: float) -> float:
        # Never let a model call eat into the last minute of wall clock, and never
        # exceed this run's own LLM time allowance.
        return min(self.call_timeout_seconds, self.total_budget_seconds - self.spent_seconds,
                   max(0.0, wallclock_remaining_seconds - 60.0))

    def _attempt(self, system_prompt: str, user_payload: dict, timeout: float) -> dict:
        """One HTTP attempt. Raises on any problem; the caller retries or gives up."""
        if self.offline:
            raise RuntimeError("network disabled in offline mode")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 0,
            "max_tokens": 250,
        }
        if self.model.startswith("gpt-6"):
            payload.pop("temperature")
            payload.pop("max_tokens")
            payload.update(max_completion_tokens=2048, reasoning_effort="low")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key}
        if self.base_url == "https://api.waytocc.com/v1":
            headers.update({"User-Agent": "GOSIM-Observer-Agent/0.45", "Accept": "application/json"})
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError("no JSON object in model reply")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("model reply was not a JSON object")
        return parsed

    def ask_json(self, system_prompt: str, user_payload: dict, wallclock_remaining_seconds: float,
                 *, stage: str) -> Optional[dict]:
        """One planning question, answered as exactly one JSON object. Retries up to
        `max_retries` times on failure; returns None once the budget/call cap/retries
        are exhausted, so the caller's rule-based answer can take over for this step."""
        def audit(outcome: str, **details) -> None:
            # No URL, credentials, prompts, payloads or raw model text in audit records.
            self.log("llm_audit " + json.dumps({"stage": stage, "outcome": outcome,
                     "network_attempts_total": self.calls_made, **details}, separators=(",", ":")))

        if self.offline:
            audit("offline_skipped")
            return None
        last_error: Optional[Exception] = None
        question_started = time.monotonic()
        for _attempt_number in range(self.max_retries):
            if self.calls_made >= self.max_calls:
                audit("call_cap_fallback")
                self.log("llm: call cap reached for this run; using the rule-based path")
                return None
            timeout = self._budget_left(wallclock_remaining_seconds - (time.monotonic() - question_started))
            if timeout < 1.5:
                audit("time_budget_fallback")
                self.log("llm: LLM time budget exhausted; using the rule-based path")
                return None
            started = time.monotonic()
            self.calls_made += 1
            audit("attempt_started", attempt=_attempt_number + 1)
            try:
                reply = validate_reply(self._attempt(system_prompt, user_payload, timeout), stage)
                audit("success", attempt=_attempt_number + 1, elapsed_ms=round((time.monotonic() - started) * 1000))
                return reply
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                audit("attempt_failed", attempt=_attempt_number + 1, error_type=type(exc).__name__)
            finally:
                self.spent_seconds += time.monotonic() - started
        audit("retries_exhausted_fallback", error_type=type(last_error).__name__)
        self.log(f"llm: call failed after {self.max_retries} attempts ({type(last_error).__name__}); "
                 "this step falls back to its rule-based answer")
        return None
