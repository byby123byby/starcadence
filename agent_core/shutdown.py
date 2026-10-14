"""Multi-night whole-telescope shutdowns announced in the duty log.

Only a formal notice with two calendar dates is read: a start evening and a morning of restoration
("... 3/14 evening starts full shutdown ... 3/26 morning restored; no observing"). Rumours, questions,
camera work, weather statements and clock-range stops are not shutdown windows. Years and time zone
come from the run input, never from constants.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .dutylog import normalise

_DATE = re.compile(r"(\d{1,2})\s*(?:/|月)\s*(\d{1,2})\s*日?")
_STOP = re.compile(r"停机|整台停|全停|全面停止|停止|無法観測|无法观测|観測は一切")
_START_AFTER = re.compile(r"^[^。；;，,]{0,12}?(?:晚|夜|晩)")
_END_AFTER = re.compile(r"^[^。；;，,]{0,10}?(?:早上|早晨|早|朝|才恢复|恢复|復旧)")
_SKIP = re.compile(r"听说|有人说|传闻|没确认|没看到|没见|别当真|先别|可能不对|我只是记得|没有正式|没通知|"
                   r"导星|相机|カメラ|ガイド|风|雨|[?？]")
_CANCEL = re.compile(r"取消|作废|不再停机|不停机|照常观测|不用停机|无需停机")


def extract_shutdowns(text: str) -> list[tuple[tuple[int, int], tuple[int, int], bool]]:
    """Return [((m1, d1), (m2, d2), cancelled)] for each formal multi-night shutdown line.

    The window covers observing nights labelled d1 .. d2-1: evening of d1 until the morning of d2.
    """
    found = []
    for raw in normalise(text).splitlines():
        line = re.sub(r"^【[^】]*】", "", raw)
        if not _STOP.search(line):
            continue
        if _SKIP.search(line):
            continue
        cancelled = bool(_CANCEL.search(line))
        dates = list(_DATE.finditer(line))
        if len(dates) < 2:
            continue
        first, second = dates[0], dates[1]
        if not _START_AFTER.match(re.sub(r"^[(（][^)）]*[)）]", "", line[first.end():])) or \
                not _END_AFTER.match(re.sub(r"^[(（][^)）]*[)）]", "", line[second.end():])):
            continue
        found.append(((int(first.group(1)), int(first.group(2))), (int(second.group(1)), int(second.group(2))), cancelled))
    return found


def resolve_dates(start_md, end_md, issued_local: date):
    """Pick calendar years from the issue date: the start is the occurrence of month/day within
    [issued - 30 d, issued + 370 d]; the end is the first later occurrence of its month/day."""
    best = None
    for year in (issued_local.year - 1, issued_local.year, issued_local.year + 1):
        try:
            candidate = date(year, *start_md)
        except ValueError:
            continue
        delta = (candidate - issued_local).days
        if -30 <= delta <= 370 and (best is None or abs(delta) < abs((best - issued_local).days)):
            best = candidate
    if best is None:
        return None
    for year in (best.year, best.year + 1):
        try:
            end = date(year, *end_md)
        except ValueError:
            continue
        if end > best:
            return best, end
    return None


def update_windows(windows: list, start: date, end: date, cancelled: bool, issued: datetime) -> list:
    """A formal notice replaces a window with the same restoration date or the same start date (a correction of
    that outage); independent outages are never merged by distance. A cancellation removes the same outage."""
    kept = [w for w in windows if w[0] != start and w[1] != end]
    if not cancelled:
        kept.append((start, end, issued))
    return kept


def night_label(night_start_utc: datetime, utc_offset_hours: float, public_label: str = "") -> date:
    """The night's public label: the initialize calendar's night_date when given, else the local date at night start."""
    if public_label:
        try:
            return date.fromisoformat(public_label[:10])
        except ValueError:
            pass
    return (night_start_utc + timedelta(hours=utc_offset_hours)).date()
