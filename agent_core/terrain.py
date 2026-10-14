"""Extract explicit ridge heights; retain the existing mask for unknown directions."""
from __future__ import annotations

import re
import unicodedata

from .dutylog import normalise

_DIRECTIONS = {"东北": "NE", "東北": "NE", "东南": "SE", "東南": "SE",
               "西北": "NW", "西南": "SW", "北西": "NW", "南西": "SW",
               "北東": "NE", "南東": "SE", "北": "N", "东": "E", "東": "E",
               "南": "S", "西": "W"}
_DIRECTION = re.compile("|".join(_DIRECTIONS) + r"|\b(?:NE|SE|SW|NW|N|E|S|W)\b")
_HEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:度|°)")
_RIDGE = re.compile(r"山|稜線|岭|脊|挡|擋|遮|实测|實測|実測|对星|對星|星.*对|数字.*准|數字.*準")
_UNCERTAIN = re.compile(r"说不准|說不準|不太确定|不太確定|たぶん|听说|聽說|传闻|傳聞|好像|也说不准|[?？]")
_UNCONFIRMED = re.compile(r"隔壁|隣.*観測所|別の山|听说|聽說|传闻|傳聞|うわさ|未确认|没确认|沒確認")


def extract_terrain_heights(text: str) -> dict[str, float]:
    """Later explicit measurements supersede earlier estimates, in message order.

    A zenith-distance 'safe to observe' bound is deliberately not a ridge measurement.
    The caller applies these heights only to directions in the public terrain bulletin.
    """
    heights = {}
    for line in unicodedata.normalize("NFKC", text).splitlines():
        if "天顶距" in line or "天頂距" in line or _UNCONFIRMED.search(line) or not _RIDGE.search(line):
            continue
        directions = list(_DIRECTION.finditer(line))
        for index, match in enumerate(directions):
            end = directions[index + 1].start() if index + 1 < len(directions) else len(line)
            segment = line[match.end():end]
            number = _HEIGHT.search(segment)
            if not number or _UNCERTAIN.search(segment):
                continue
            height = float(number.group(1))
            if 0 <= height <= 90:
                heights[_DIRECTIONS.get(match.group(), match.group())] = height
    return heights


def extract_wind_updates(text: str) -> list[tuple[float, float, bool]]:
    """Explicit tonight/all-night wind closures, without padding coarse bearings.

    Later-day forecasts and tentative/other-station statements are not closures
    for tonight. The caller binds these sectors to the public observing night.
    """
    bearings = {"北": 0, "东北": 45, "东": 90, "东南": 135,
                "南": 180, "西南": 225, "西": 270, "西北": 315}
    direction = "(?:东北|东南|西南|西北|北|东|南|西)"
    sectors = []
    for line in normalise(text).translate(str.maketrans("東邊風關開", "东边风关开")).splitlines():
        tonight = False
        for clause in re.split(r"[;。]", line):
            # A forecast can mention tonight followed by tomorrow in one line.
            parts = re.split(r"明天|明晚|后天|周[一二三四五六日天]|下周", clause, maxsplit=1)
            clause = parts[0]
            tonight = tonight or bool(re.search(r"今晚|今夜|今天晚上|今儿夜里|今日夜里", clause))
            if (tonight and "风" in clause
                    and not re.search(r"听说|传闻|未确认|没确认|别当真|不要当真|隔壁|如果|可能|假如|假设", clause)):
                match = re.search(rf"({direction})(?:边|方|面)?\s*(?:到|至)\s*({direction})", clause)
                if match:
                    first, last = (bearings[name] for name in match.groups())
                    detail = clause[match.end():]
                    cancelled = bool(re.search(r"没关|不用关|不会关|不关|取消|作废|撤销|解除|有误|错了|恢复观测|已开放", detail)
                                     or re.search(r"(?:取消|撤销|作废|解除)\s*:", clause[:match.start()]))
                    closed = bool(re.search(r"风(?:很)?大|大风|强风|风速超限", detail)
                                  and re.search(r"整[晚夜].*(?:关|不开)", detail))
                    # Ambiguous long arcs are left to feedback, not silently filled.
                    if (closed or cancelled) and 0 < (last - first) % 360 <= 180:
                        sectors.append((first, last, not cancelled))
            if len(parts) > 1:
                tonight = False
    return sectors
