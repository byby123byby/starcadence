"""Compile the station duty log (cards A1-D1 attach it to observation requests) into repair times.

Only public text is read. Rules are deterministic; report feedback bounds repeated mistakes,
but parsing does not prove the truth of an announcement. Handled forms: local, UTC and
UTC+9 clock times, Chinese and solfege time words, postponements, cancellations, corrections,
rumours (ignored), "N after the first flat-lamp test" and rain-conditional slips.
"""
import re
import unicodedata
from datetime import datetime, timedelta, timezone

_TRAD = str.maketrans({
    "導": "导", "機": "机", "動": "动", "報": "报", "時": "时", "聽": "听", "說": "说", "沒": "没",
    "確": "确", "認": "认", "預": "预", "後": "后", "點": "点", "當": "当", "這": "这", "對": "对",
    "傳": "传", "錯": "错", "週": "周", "點": "点", "個": "个", "鐘": "钟", "兩": "两", "隊": "队",
    "說": "说", "將": "将", "誤": "误", "場": "场", "測": "测", "為": "为", "準": "准", "從": "从",
    "裡": "里", "來": "来", "們": "们", "遲": "迟", "訊": "讯", "號": "号", "還": "还", "應": "应",
    "別": "别", "於": "于", "請": "请", "與": "与", "線": "线", "電": "电", "話": "话", "更": "更",
    "週": "周", "號": "号", "鏡": "镜", "蓋": "盖", "據": "据", "單": "单", "續": "续", "間": "间",
    "晚": "晚", "經": "经", "過": "过", "並": "并", "靜": "静", "隨": "随", "擊": "击", "檢": "检",
})
_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SOLFEGE = {"高音do": 8, "高音re": 9, "sol": 5, "do": 1, "re": 2, "mi": 3, "fa": 4, "la": 6, "si": 7, "休止": 0}
_SOLFEGE_TOKEN = r"(?:高音do|高音re|sol|do|re|mi|fa|la|si|休止)"
_SOLFEGE_TIME = re.compile(rf"((?:{_SOLFEGE_TOKEN}\s*){{1,2}}):\s*((?:{_SOLFEGE_TOKEN}\s*){{2}})")
_DATE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})|(\d{1,2})月(\d{1,2})日")
_CLOCK = re.compile(r"(\d{1,2})[:：](\d{2})")
_RUMOUR = ("听说", "传言", "有人说", "没确认", "先别当真", "先别信", "别当真", "不要当真")
_CLOCK_TOKEN = re.compile(r"(\d{1,2})[:：](\d{2})|([零〇一二两三四五六七八九十\d]{1,3})点(整|半|[一二三]刻|[零〇一二三四五六七八九十\d]{1,3}分)?")
_DATE_TOKEN = r"(?:\d{1,2}\s*/\s*\d{1,2}|\d{1,2}月\d{1,2}日)"
# "B 比 A 那次早/晚 …" and "B 与 A 那次相同": the reference word (前面/上面/上述/之前) is optional, the reference DATE is not.
_REL_EVENT = re.compile(rf"比\s*(?:上面|前面|上述|之前|上一个?)?\s*{_DATE_TOKEN}|[与同跟]\s*(?:上面|前面|上述|之前)?\s*{_DATE_TOKEN}[^。]*?(?:相同|一样|同一时间)")
_TEST_DONE = re.compile(r"(?:结束|完了|完成|完毕|完事|收尾|收工|收束|跑完|做完|测完|告一段落|完结)了?(?:以|之)?后")
_CORRECTION = re.compile(r"导星相机(?:其实|实际)?(?:是|应为|实为|(?:的)?正确时间(?:是|为))")


def _cn_number(text: str):
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        head, _, tail = text.partition("十")
        tens = _CN.get(head, 1) if head else 1
        return tens * 10 + (_CN.get(tail, 0) if tail else 0)
    if len(text) == 1:
        return _CN.get(text)
    return None


_EMOTICON = re.compile(r"[(][^()\d\u4e00-\u9fff]{1,16}[)]|[\u2500-\u257f\u30fd\uff89\uff9f\u2570-\u25ff\u2600-\u27bf]+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRAD)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"[(]\s*UTC\s*[)]", " UTC ", text)
    text = _EMOTICON.sub("", text)
    text = re.sub(r"\bm\s*\(_ _\)\s*m", "", text)

    def solfege(match):
        def digits(part):
            return "".join(str(_SOLFEGE[token]) for token in re.findall(_SOLFEGE_TOKEN, part))
        hour, minute = digits(match.group(1)), digits(match.group(2))
        return f"{int(hour)}:{minute}"
    return _SOLFEGE_TIME.sub(solfege, text)


def _clock_words(text: str, start: int = 0):
    """First clock reading at/after `start`: returns (hour, minute, end) or None."""
    candidates = []
    match = _CLOCK.search(text, start)
    if match:
        candidates.append((match.start(), int(match.group(1)), int(match.group(2)), match.end()))
    for match in re.finditer(r"(\d{1,2}|[零〇一二两三四五六七八九十]{1,3})点(整|半|[一二三]刻|"
                             r"\d{1,2}分|[零〇一二两三四五六七八九十]{1,3}分)?", text[start:]):
        hour = _cn_number(match.group(1))
        tail = match.group(2) or ""
        if tail == "半":
            minute = 30
        elif tail.endswith("刻"):
            minute = 15 * _CN.get(tail[0], 0)
        elif tail.endswith("分"):
            minute = _cn_number(tail[:-1])
        else:
            minute = 0
        if hour is not None and minute is not None and hour < 24 and minute < 60:
            candidates.append((start + match.start(), hour, minute, start + match.end()))
    if not candidates:
        return None
    _, hour, minute, end = min(candidates)
    return (hour, minute, end) if hour < 24 and minute < 60 else None


def _duration(text: str):
    total, found = 0, False
    match = re.search(r"(\d+|[零〇一二两三四五六七八九十]{1,3})\s*个?\s*(半)?小时(半)?", text)
    if match and _cn_number(match.group(1)) is not None:
        total += 60 * _cn_number(match.group(1))
        found = True
        if match.group(2) or match.group(3):
            total += 30
    elif "半小时" in text or "半个小时" in text:
        total += 30
        found = True
    match = re.search(r"(\d+|[零〇一二两三四五六七八九十]{1,3})\s*分(?:钟)?", text)
    if match and _cn_number(match.group(1)) is not None:
        total += _cn_number(match.group(1))
        found = True
    return timedelta(minutes=total) if found else None


def _clock_tokens(text):
    """Every clock reading in order: HH:MM or Chinese words (二十一点半, 二点一刻, 零点半, ...)."""
    out = []
    for m in _CLOCK_TOKEN.finditer(text):
        if m.group(1) is not None:
            hour, minute = int(m.group(1)), int(m.group(2))
        else:
            hour = _cn_number(m.group(3)); tail = m.group(4) or ""
            minute = 30 if tail == "半" else 15 * _CN.get(tail[0], 0) if tail.endswith("刻") else \
                (_cn_number(tail[:-1]) if tail.endswith("分") else 0)
        if hour is None or minute is None or hour >= 24 or minute >= 60:
            continue
        out.append((hour, minute))
    return out


class DutyLogCompiler:
    """Accumulates repair events across observation requests."""

    def __init__(self, utc_offset_hours: float):
        self.offset = float(utc_offset_hours)
        self.events = {}        # utc datetime -> {"kind", "rain_slip"}
        self.cancelled = set()  # utc datetimes
        self.superseded = set() # originals replaced by a postponement
        self._by_date = {}      # local (month, day) -> latest event, for relative notices
        self.test_windows = set()

    def _issue_year_date(self, month, day, issued):
        best = None
        for year in (issued.year - 1, issued.year, issued.year + 1):
            try:
                local = datetime(year, month, day)
            except ValueError:
                continue
            gap = (local - issued.replace(tzinfo=None)).total_seconds()
            if -4 * 86400 <= gap <= 90 * 86400 and (best is None or abs(gap) < best[0]):
                best = (abs(gap), local)
        return best[1] if best else None

    def _parse_datetime(self, text, issued, line_offset):
        """First date+clock in text -> (utc datetime, end index) or None."""
        for date in _DATE.finditer(text):
            month = int(date.group(1) or date.group(3))
            day = int(date.group(2) or date.group(4))
            clock = _clock_words(text, date.end())
            if clock is None or clock[2] - date.end() > 24:
                continue
            base = self._issue_year_date(month, day, issued)
            if base is None:
                continue
            local = base + timedelta(hours=clock[0], minutes=clock[1])
            return (local - timedelta(hours=line_offset)).replace(tzinfo=timezone.utc), clock[2]
        return None

    def _line_offset(self, text):
        match = re.search(r"UTC\s*([+\-\u2212])\s*(\d{1,2}(?:\.\d+)?)(?::(\d{2}))?", text)
        if match:
            sign = -1.0 if match.group(1) in "-\u2212" else 1.0
            return sign * (float(match.group(2)) + float(match.group(3) or 0) / 60)
        if "东京" in text:
            return 9.0
        if "UTC" in text:
            return 0.0
        return self.offset

    def _tests(self, lines, issued):
        """Flat-lamp windows by night date: {date: [(start_local, end_local), ...]} for relative notices."""
        found = {}
        for line in lines:
            if "平场灯" not in line or "段" not in line:
                continue
            if any(word in line for word in _RUMOUR) or re.search(r"[?？]|取消|未确认|可能", line):
                continue
            date = _DATE.search(line)
            if not date:
                continue
            base = self._issue_year_date(int(date.group(1) or date.group(3)),
                                         int(date.group(2) or date.group(4)), issued)
            if base is None:
                continue
            spans = re.findall(r"(\d{1,2})[:：](\d{2})到(\d{1,2})[:：](\d{2})", line)
            windows = []
            for h1, m1, h2, m2 in spans:
                if any(int(h) >= 24 for h in (h1, h2)) or any(int(m) >= 60 for m in (m1, m2)):
                    continue
                def at(hour, minute):
                    stamp = base + timedelta(hours=int(hour), minutes=int(minute))
                    if re.search(r"夜|晚", line) and "UTC" not in line and int(hour) < 12:
                        stamp += timedelta(days=1)
                    return stamp + timedelta(hours=self.offset - self._line_offset(line))
                start, end = at(h1, m1), at(h2, m2)
                if end < start:
                    end += timedelta(days=1)
                if start < end:
                    windows.append((start, end))
            if windows:
                found[(base.month, base.day)] = sorted(windows)
        return found

    def _tests_wide(self, lines, issued):
        """Like _tests but reads every segment notation (到, 至, dashes, Chinese clock words, solfege starts).
        Used only to resolve 'N after the k-th test' repair times; the scheduling set test_windows is NOT built from it."""
        found = {}
        for line in lines:
            if "平场灯" not in line or "段" not in line:
                continue
            if any(word in line for word in _RUMOUR) or re.search(r"[?？]|取消|未确认|可能|好像", line):
                continue
            if re.search(r"第[一二三四五]段平场灯测试", line):
                continue
            date = _DATE.search(line)
            if not date:
                continue
            base = self._issue_year_date(int(date.group(1) or date.group(3)),
                                         int(date.group(2) or date.group(4)), issued)
            if base is None:
                continue
            count = re.search(r"([一二两三四五\d])\s*段", line)
            declared = int(count.group(1)) if count and count.group(1).isdigit() else \
                {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}.get(count.group(1), 0) if count else 0
            tokens = _clock_tokens(line[count.end():] if count else line[date.end():])
            if declared and len(tokens) >= 2 * declared:
                tokens = tokens[:2 * declared]
            elif declared or len(tokens) % 2 or len(tokens) < 2:
                continue
            windows = []
            for (h1, m1), (h2, m2) in zip(tokens[0::2], tokens[1::2]):
                def at(hour, minute):
                    stamp = base + timedelta(hours=hour, minutes=minute)
                    if re.search(r"夜|晚", line) and "UTC" not in line and hour < 12:
                        stamp += timedelta(days=1)
                    return stamp + timedelta(hours=self.offset - self._line_offset(line))
                start, end = at(h1, m1), at(h2, m2)
                if end < start:
                    end += timedelta(days=1)
                if start < end:
                    windows.append((start, end))
            if windows:
                found[(base.month, base.day)] = sorted(windows)
        return found

    def feed(self, reason: str, issued_utc: str) -> int:
        """Parse one request's text; returns the number of events known afterwards."""
        issued = datetime.fromisoformat(issued_utc.replace("Z", "+00:00"))
        lines = [normalise(raw) for raw in str(reason).split("\n")]
        tests = self._tests(lines, issued)          # scheduling set (unchanged)
        tests_wide = self._tests_wide(lines, issued)  # relative repair times only
        deferred = []
        for windows in tests.values():
            for start, end in windows:
                if start < end:
                    self.test_windows.add(((start - timedelta(hours=self.offset)).replace(tzinfo=timezone.utc),
                                           (end - timedelta(hours=self.offset)).replace(tzinfo=timezone.utc)))
        for line in lines:
            body = re.sub(r"^【[^】]+】", "", line)
            if any(word in body for word in _RUMOUR):
                continue
            # Reliability belongs to the assertion, not the speaker's name.
            if re.search(r"[?？]|对吗|难道|说不准", body):
                continue
            offset = self._line_offset(body)
            if "下雨了" in body and re.search(r"顺延一天|往后顺延一天", body):
                date = _DATE.search(body)
                if date:
                    day = (int(date.group(1) or date.group(3)), int(date.group(2) or date.group(4)))
                    for moment, info in list(self.due()):
                        local = moment + timedelta(hours=self.offset)
                        if info["rain_slip"] and (local.month, local.day) == day:
                            self.superseded.add(moment)
                            self._add(moment + timedelta(days=1), "camera", False)
                continue
            if "取消" in body and "导星" in body:
                parsed = self._parse_datetime(body, issued, offset)
                if parsed:
                    self.cancelled.add(parsed[0])
                continue
            if self._is_step(body) and not ("导星" in body and _DATE.search(body)):
                self._step(body, issued)
                continue
            if "导星" not in body:
                continue
            if "平场灯" in body and _TEST_DONE.search(body):
                self._relative_to_test(body, issued, tests_wide)
                continue
            if _REL_EVENT.search(body):
                deferred.append(body)   # needs its reference job, which may be listed later in the same request
                continue
            search_from = 0
            correction = None
            for correction in _CORRECTION.finditer(body):
                pass
            if correction and re.search(r"不对|记错|搞错|岔|有误|错误|错的|订正|更正", body):
                search_from = correction.end()
                wrong = self._parse_datetime(body[:correction.start()], issued, offset)
                if wrong:
                    self.superseded.add(wrong[0])
            parsed = self._parse_datetime(body[search_from:], issued, offset)
            if not parsed:
                continue
            moment = parsed[0]
            if re.search(r"推迟|延后|延迟", body) and re.search(r"原定|原本|本来|原计划", body):
                delay = _duration(body[body.find("推迟") if "推迟" in body else 0:])
                if delay:
                    self.superseded.add(moment)
                    self._add(moment + delay, "camera", False)
                continue
            self._add(moment, "camera", "顺延" in body)
        for _ in range(3):
            pending = [body for body in deferred if not self._relative_to_event(body, issued)]
            if len(pending) == len(deferred):
                break
            deferred = pending
        return len(self.events)

    def _add(self, moment, kind, rain_slip):
        if moment not in self.events:
            self.events[moment] = {"kind": kind, "rain_slip": rain_slip}
            local = moment + timedelta(hours=self.offset)
            self._by_date[(local.month, local.day)] = moment

    @staticmethod
    def _is_step(body):
        if re.search(r"将|计划|会降|会掉|一.*就", body):
            return False
        act = r"提单|叫工程组|工程组上来|把工程组叫|让工程组|报修"
        if re.search(r"单独|自己", body) and re.search(r"往下|下跳|跳下|下降|掉", body):
            return bool(re.search(r"工程组|提单", body))
        if "提单" in body and re.search(r"allsky|机器的事|不怪天|跟天没关系|不是云", body):
            return True
        return bool(re.search(r"吞吐量|效率", body) and re.search(r"掉|下降|下去|只剩|降到", body)
                    and re.search(act, body)) and not re.search(r"一掉就|万一|如果|可能", body)

    def _step(self, body, issued):
        """A drop that began before the message: due on receipt, dated by its clock time if given."""
        dated = self._parse_datetime(body, issued, self._line_offset(body))
        if dated:
            self._add(dated[0], "step", False)
            return
        clock = _clock_words(body)
        moment = issued
        if clock:
            local = issued.replace(tzinfo=None) + timedelta(hours=self.offset)
            stamp = local.replace(hour=clock[0], minute=clock[1], second=0, microsecond=0)
            if stamp > local:
                stamp -= timedelta(days=1)
            moment = (stamp - timedelta(hours=self.offset)).replace(tzinfo=timezone.utc)
        self._add(moment, "step", False)

    def _relative_to_event(self, body, issued):
        """'B 比 A 那次 早/晚 delta (by the clock)' or 'B 与 A 那次相同'. Returns True when a time was added."""
        dates = list(_DATE.finditer(body))
        found = _REL_EVENT.search(body)
        if not found:
            return False
        cut = found.start()
        new_day = next((d for d in dates if d.start() < cut), None) or (dates[0] if len(dates) > 1 else None)
        ref_day = next((d for d in dates if d.start() > cut), None)
        if new_day is None or ref_day is None:
            return False
        same = bool(re.search(r"相同|一样|同一时间", body[cut:]))
        match = re.search(r"([早晚])", body[cut:])
        delay = _duration(body[cut + match.end():]) if match else None
        if not same and (match is None or delay is None):
            return False
        ref_key = (int(ref_day.group(1) or ref_day.group(3)), int(ref_day.group(2) or ref_day.group(4)))
        ref = self._by_date.get(ref_key)
        base = self._issue_year_date(int(new_day.group(1) or new_day.group(3)),
                                     int(new_day.group(2) or new_day.group(4)), issued)
        if ref is None or base is None:
            return False
        local_ref = ref + timedelta(hours=self.offset)
        clock = timedelta(hours=local_ref.hour, minutes=local_ref.minute)
        if not same:
            clock = clock - delay if match.group(1) == "早" else clock + delay
        moment = (base + clock - timedelta(hours=self.offset)).replace(tzinfo=timezone.utc)
        self._add(moment, "camera", False)
        return True

    def _relative_to_test(self, body, issued, tests):
        date = _DATE.search(body)
        done = _TEST_DONE.search(body)
        if not date or not done:
            return
        base = self._issue_year_date(int(date.group(1) or date.group(3)),
                                     int(date.group(2) or date.group(4)), issued)
        windows = tests.get((base.month, base.day)) if base else None
        ordinal = re.search(r"第([一二三四五])段平场灯测试", body)
        index = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4}[ordinal.group(1)] if ordinal else 0
        delay = _duration(body[done.end():])
        if windows and delay is not None and index < len(windows):
            moment = (windows[index][1] + delay - timedelta(hours=self.offset)).replace(tzinfo=timezone.utc)
            self._add(moment, "camera", False)

    def due(self):
        """Live events sorted by time: [(utc datetime, info), ...]."""
        dead = self.cancelled | self.superseded
        return sorted((moment, info) for moment, info in self.events.items() if moment not in dead)
