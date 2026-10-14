"""Validate a decision_response before it goes to stdout, and provide a
deterministic fallback for when the planner errors out or produces something
invalid. The protocol ends a run as `agent_error` for things like a fibre used
twice, an unknown target, a value out of range, or an unknown field -- this
module's whole job is to make sure none of those ever reach stdout.
"""
from __future__ import annotations

from datetime import datetime

PROGRAMS = {"DARK", "BRIGHT", "BACKUP"}


class ActionRejected(ValueError):
    pass


def fallback_action(reason: str = "fallback", state=None) -> dict:
    """Wait within the current card's public limits, or the default limits before initialize."""
    lo, hi = (state.min_exposure, state.max_exposure) if state is not None else (60, 3600)
    return {"action": "wait", "duration_seconds": min(hi, max(lo, 900)),
            "reason": reason, "decision_source": "deterministic"}


def validate_action(action: dict, state, consecutive_reports: int = 0, now_utc: str = None) -> dict:
    """Return a sanitized copy of `action` containing only protocol-legal fields,
    or raise ActionRejected. `state` may be None (before initialize)."""
    if not isinstance(action, dict):
        raise ActionRejected("action is not a dict")
    for field in ("reason", "decision_source"):
        if action.get(field) is not None and not isinstance(action[field], str):
            raise ActionRejected(f"{field} must be a string")
    kind = action.get("action")
    if kind == "observe":
        return _validate_observe(action, state)
    if kind == "wait":
        return _validate_wait(action, state, now_utc)
    if kind == "report":
        limit = state.max_consecutive_reports if state is not None else 32
        if consecutive_reports >= limit:
            raise ActionRejected(f"consecutive report limit reached ({limit})")
        return {"action": "report", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    if kind == "finish":
        return {"action": "finish", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    raise ActionRejected(f"unknown action kind {kind!r}")


def _validate_wait(action: dict, state, now_utc: str = None) -> dict:
    duration = action.get("duration_seconds")
    until = action.get("until_utc")
    out = {"action": "wait", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    if until is not None:
        if duration is not None:
            raise ActionRejected("wait cannot have both duration_seconds and until_utc")
        if not isinstance(until, str) or not until.endswith("Z"):
            raise ActionRejected(f"until_utc must be a UTC string ending in Z, got {until!r}")
        try:
            until_time = datetime.fromisoformat(until[:-1] + "+00:00")
        except ValueError as exc:
            raise ActionRejected("until_utc is not a valid timestamp") from exc
        if until_time.tzinfo is None:
            raise ActionRejected("until_utc requires a UTC timestamp")
        if now_utc is not None and until_time <= datetime.fromisoformat(now_utc.replace("Z", "+00:00")):
            raise ActionRejected("until_utc must be later than now_utc")
        out["until_utc"] = until
        return out
    if duration is None:
        raise ActionRejected("wait needs duration_seconds or until_utc")
    if type(duration) is not int:
        raise ActionRejected("wait duration_seconds must be an integer")
    lo = state.min_exposure if state else 60
    hi = state.max_exposure if state else 3600
    if not lo <= duration <= hi:
        raise ActionRejected(f"wait duration_seconds out of range: {duration}")
    out["duration_seconds"] = duration
    return out


def _validate_observe(action: dict, state) -> dict:
    pointing = action.get("pointing") or {}
    if not isinstance(pointing, dict):
        raise ActionRejected("pointing must be an object")
    alt = pointing.get("alt_deg")
    az = pointing.get("az_deg")
    if type(alt) not in (int, float) or not 0.0 <= alt <= 90.0:
        raise ActionRejected(f"pointing.alt_deg out of range: {alt!r}")
    if type(az) not in (int, float) or not 0.0 <= az < 360.0:
        raise ActionRejected(f"pointing.az_deg out of range: {az!r}")

    assignments = action.get("assignments") or {}
    if not isinstance(assignments, dict) or not assignments:
        raise ActionRejected("observe needs at least one fibre assignment")
    seen_fibers: set[int] = set()
    seen_targets: set[str] = set()
    clean_assignments: dict[str, str] = {}
    for fiber_key, target_id in assignments.items():
        if not isinstance(fiber_key, str) or not isinstance(target_id, str):
            raise ActionRejected("assignments must map string fibre ids to string target ids")
        try:
            fiber_id = int(fiber_key)
        except ValueError as exc:
            raise ActionRejected("fibre id must be an integer string") from exc
        if not 0 <= fiber_id < (state.fiber_grid.n if state else 16):
            raise ActionRejected(f"fibre id out of range: {fiber_key!r}")
        if fiber_id in seen_fibers:
            raise ActionRejected(f"fibre {fiber_id} used twice")
        if target_id in seen_targets:
            raise ActionRejected(f"target {target_id} assigned twice")
        if state is not None and target_id not in state.index_of:
            raise ActionRejected(f"unknown target id {target_id!r}")
        seen_fibers.add(fiber_id)
        seen_targets.add(target_id)
        clean_assignments[str(fiber_id)] = target_id

    duration = action.get("duration_seconds")
    if duration is None:
        raise ActionRejected("observe needs duration_seconds")
    if type(duration) is not int:
        raise ActionRejected("duration_seconds must be an integer")
    lo = state.min_exposure if state else 60
    hi = state.max_exposure if state else 3600
    if not lo <= duration <= hi:
        raise ActionRejected(f"duration_seconds out of range: {duration}")

    program = action.get("program", "BACKUP")
    if program not in PROGRAMS:
        raise ActionRejected(f"unknown program {program!r}")

    return {
        "action": "observe",
        "pointing": {"alt_deg": float(alt), "az_deg": float(az)},
        "assignments": clean_assignments,
        "duration_seconds": duration,
        "program": program,
        "reason": action.get("reason"),
        "decision_source": action.get("decision_source"),
    }
