#!/usr/bin/env python3
"""Entry point for the python-agent example (participant-agent-protocol-v4).

Reads one JSON object per line on stdin, writes one JSON object per line on
stdout, logs only to stderr. The loop itself is deliberately thin: all the
decision-making lives in agent_core/ (protocol I/O, state tracking, geometry,
scoring, planning, the LLM client, memory/log, action validation) so this file
stays a readable map of "what happens for each message type".

The default platform mode checks for an API key. --codex uses a local Codex CLI
login instead; --offline explicitly skips all model calls. See the README.

  initialize        -> build SurveyState + Planner from the public payload
  decision_request   -> Planner.decide(), validated, sent back as decision_response
  finish              -> Planner.on_finish() logs a summary and the process exits

Any planner exception is caught here and replaced with a safe fallback action --
a bug in the strategy must never end the run as agent_error or hang the process.
"""
from __future__ import annotations

import argparse
import sys

if sys.version_info < (3, 9):
    sys.stderr.write("agent: Python 3.9 or newer is required\n")
    raise SystemExit(3)

from agent_core.llm_client import MissingAPIKeyError, require_api_key
from agent_core.codex_client import codex_binary
from agent_core.planner import Planner
from agent_core.protocol import log, read_messages, send_response
from agent_core.state import SurveyState
from agent_core.validation import ActionRejected, fallback_action, validate_action


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="debug without API keys or network; not an LLM entry")
    mode.add_argument("--codex", action="store_true", help="local model calls using the existing Codex CLI login")
    args = parser.parse_args(argv)
    if args.offline:
        log("agent: OFFLINE DEBUG MODE; no LLM requests; not evidence of LLM award eligibility")
    elif args.codex:
        try:
            codex_binary()
        except OSError as exc:
            log(f"agent: {exc}")
            return 1
        log("agent: LOCAL CODEX MODE; using the official CLI login")
    else:
        try:
            require_api_key()
        except MissingAPIKeyError as exc:
            log(f"agent: {exc}")
            return 1

    state = None
    planner = None
    for message in read_messages(sys.stdin):
        kind = message.get("message_type")

        if kind == "initialize":
            try:
                state = SurveyState(message["payload"])
                planner = Planner(state, log=log, offline=args.offline, codex=args.codex)
            except Exception as exc:  # noqa: BLE001 - never crash on a malformed initialize
                log(f"agent: failed to initialize ({type(exc).__name__}: {exc}); will fall back on every decision")
                state = None
                planner = None

        elif kind == "decision_request":
            sequence = message["decision_sequence"]
            consecutive_reports = planner.consecutive_reports if planner is not None else 0
            try:
                action = planner.decide(message["payload"]) if planner is not None else fallback_action("not initialized", state)
                action = validate_action(action, state, consecutive_reports, now_utc=message["payload"].get("now_utc"))
            except ActionRejected as exc:
                log(f"agent: planner produced an invalid action ({exc}); falling back")
                action = fallback_action("validation-rejected", state)
            except Exception as exc:  # noqa: BLE001 - a strategy bug must not end the run
                log(f"agent: planner error ({type(exc).__name__}: {exc}); falling back")
                action = fallback_action("planner-exception", state)
            if planner is not None:
                planner.note_action(action)
            send_response(sequence, action, max_bytes=state.response_max_bytes if state is not None else 524288)

        elif kind == "finish":
            if planner is not None:
                try:
                    planner.on_finish(message.get("payload", {}))
                except Exception as exc:  # noqa: BLE001 - finish must not raise after the score is fixed
                    log(f"agent: error during finish logging ({type(exc).__name__}: {exc})")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
