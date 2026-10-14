#!/usr/bin/env python3
"""Run a small synthetic JSONL session in offline mode: python3 scripts/smoke.py."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from agent_core.state import SurveyState
from agent_core.validation import validate_action


def invoke(messages):
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "agent.py"), "--offline"],
        input="".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True, text=True, cwd=ROOT, timeout=30,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    for error in ("failed to initialize", "planner error", "falling back", "finish logging"):
        assert error not in result.stderr, result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def main():
    sources = [ROOT / "agent.py", *sorted((ROOT / "agent_core").glob("*.py"))]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    messages = [json.loads(line) for line in
                (ROOT / "examples/session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert invoke([messages[0], messages[-1]]) == [], "initialize/finish emitted a reply"
    requests = [m for m in messages if m["message_type"] == "decision_request"]
    replies = invoke(messages)
    assert len(replies) == len(requests), "expected one response per decision request"

    state = SurveyState(messages[0]["payload"])
    actions = []
    for request, response in zip(requests, replies):
        assert response["protocol_version"] == "participant-agent-protocol-v4"
        assert response["message_type"] == "decision_response"
        assert response["decision_sequence"] == request["decision_sequence"]
        validate_action(response, state, now_utc=request["payload"]["now_utc"])
        actions.append(response["action"])
    assert any(action in {"observe", "wait"} for action in actions)
    print(f"PASS: {len(sources)} Python sources parsed; initialize/finish stayed silent;")
    print(f"      {len(replies)} matching, valid decision responses: {', '.join(actions)}.")
    print("      Synthetic public inputs, offline mode, no model calls.")


if __name__ == "__main__":
    main()
