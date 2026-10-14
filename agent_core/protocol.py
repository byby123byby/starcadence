"""Transport layer for participant-agent-protocol-v4: one JSON object per line.

Reads `initialize` / `decision_request` / `finish` messages from stdin and writes
`decision_response` messages to stdout. Logging must never touch stdout -- the
platform only accepts protocol JSON there -- so every log line goes to stderr.

Standard library only.
"""
from __future__ import annotations

import json
import sys
from typing import Iterable, Iterator

PROTOCOL_VERSION = "participant-agent-protocol-v4"

# Fields the platform accepts on a decision_response, per action. Sending anything
# else (or missing the envelope fields) ends the run as agent_error.
_RESPONSE_FIELDS = {
    "observe": {"pointing", "assignments", "duration_seconds", "program"},
    "wait": {"duration_seconds", "until_utc"},
    "report": set(),
    "finish": set(),
}
_OPTIONAL_FIELDS = {"reason", "decision_source"}


def log(text: str) -> None:
    """Write one diagnostic line to stderr. Never raises."""
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


def read_messages(stream: Iterable[str]) -> Iterator[dict]:
    """Yield parsed JSON objects from an iterable of lines, skipping blanks.

    A line that is not valid JSON is logged and skipped rather than raising --
    the platform should never send one, but a malformed line must not crash
    the agent before it has a chance to answer later requests.
    """
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            log(f"protocol: could not parse input line ({exc}); skipping")
            continue
        if not isinstance(message, dict):
            log("protocol: input line did not decode to a JSON object; skipping")
            continue
        yield message


def send_response(decision_sequence: int, action: dict, max_bytes: int = 524288) -> None:
    """Write one decision_response line for `decision_sequence`, keeping only
    the fields the protocol allows for this action's `action` kind."""
    kind = action.get("action")
    allowed = _RESPONSE_FIELDS.get(kind, set()) | _OPTIONAL_FIELDS | {"action"}
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "decision_response",
        "decision_sequence": decision_sequence,
    }
    envelope = dict(base)
    for key, value in action.items():
        if key in allowed and value is not None:
            envelope[key] = value
    try:
        encoded = json.dumps(envelope, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        log(f"protocol: could not encode response ({type(exc).__name__}); sending finish")
        envelope = {**base, "action": "finish"}
        encoded = json.dumps(envelope, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes:
        for field in _OPTIONAL_FIELDS:
            envelope.pop(field, None)
        encoded = json.dumps(envelope, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > max_bytes:
            log("protocol: response exceeds byte limit after removing optional fields; sending finish")
            encoded = json.dumps({**base, "action": "finish"}, separators=(",", ":"))
    print(encoded, flush=True)
