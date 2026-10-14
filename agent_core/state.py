"""Survey state: the target catalogue, learned sky-quality scale, and per-target
progress. Built once from `initialize`, then updated from every `decision_request`'s
messages and `last_result`. Holds no hidden data -- only what the public protocol
hands us, plus what we infer from our own hits (never from a file).

Mirrors the structure-of-parallel-arrays design (ids/ra/dec/flux/weight/required/...,
all indexed by the same integer i) used by this project's companion TypeScript example,
so both examples solve the same problem the same way and can be compared directly.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from statistics import median
from typing import NamedTuple, Optional

from .geometry import FiberGrid, max_hour_angle_deg, parse_utc, wrap180
from .scoring import ScoringModel

ALT_MARGIN_DEG = 0.6
SKY_MEMORY_HOURS = 2.0
RECENT_SAMPLES = 60
EARLIER_SAMPLES = 60
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


class PendingPrediction(NamedTuple):
    model: float          # lunar/airmass quality model used at planning time
    band_model: float      # model / 0.95, used for program-band back-estimation
    alt: float
    az: float
    clean: bool            # true when no all-sky notice / directional block applied at plan time
    shared: bool = False   # geometric/directional eligibility for joint program feedback


class FaultEvidence(NamedTuple):
    recent_median: float
    earlier_median: float
    drop: float
    recent_samples: int
    recent_nights: int
    earlier_samples: int
    dark_checks: int
    dark_matched: int


def _mod(a: float, n: float) -> float:
    m = a % n
    return m + n if m < 0 else m


def _intersect_ranges(left, right):
    return [(max(lo, low), min(hi, high)) for lo, hi in left for low, high in right
            if max(lo, low) < min(hi, high)]


class SurveyState:
    def __init__(self, init_payload: dict):
        site = init_payload["site"]
        survey = init_payload["survey"]
        instrument = init_payload["instrument"]
        limits = init_payload.get("limits", {})

        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site.get("minimum_altitude_deg", 30.0))
        self.utc_offset_hours = float(site.get("utc_offset_hours", 0.0))
        self.sun_altitude_limit_deg = float(site.get("sun_altitude_limit_deg", -18.0))

        self.survey_start = parse_utc(survey["start_utc"])
        self.survey_end = parse_utc(survey["end_utc"])
        self.slot_seconds = int(survey.get("slot_seconds", 900))
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"]))
                       for n in survey.get("nights", [])]
        # Public night label (local calendar date) from the initialize night calendar when present.
        self.night_dates = [str(n.get("night_date") or "") for n in survey.get("nights", [])]

        self.fiber_grid = FiberGrid(instrument)
        exposure = instrument.get("exposure", {})
        self.min_exposure = int(exposure.get("min_duration_seconds", 60))
        self.max_exposure = int(exposure.get("max_duration_seconds", 3600))

        self.scoring = ScoringModel(init_payload.get("scoring", {}), site)

        reporting = init_payload.get("scoring", {}).get("reporting", {})
        self.max_consecutive_reports = int(reporting.get("max_consecutive_reports", limits.get("max_consecutive_reports", 32)))
        self.false_report_free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.false_reports_since_correct = 0
        self.response_max_bytes = int(limits.get("response_max_bytes", 524288))

        # Parallel arrays, one slot per target, in catalogue order.
        self.ids: list[str] = []
        self.ra: list[float] = []
        self.dec: list[float] = []
        self.flux: list[float] = []
        self.weight: list[float] = []
        self.required: list[bool] = []
        self.index_of: dict[str, int] = {}

        columns = init_payload.get("targets", {}).get("columns", [])
        col = {name: idx for idx, name in enumerate(columns)}
        for row in init_payload.get("targets", {}).get("rows", []):
            target_id = str(row[col["target_id"]])
            self.index_of[target_id] = len(self.ids)
            self.ids.append(target_id)
            self.ra.append(float(row[col["ra_deg"]]))
            self.dec.append(float(row[col["dec_deg"]]))
            self.flux.append(float(row[col["feature_flux"]]))
            self.weight.append(float(row[col["science_weight"]]))
            self.required.append(bool(row[col["required"]]))

        n = len(self.ids)
        self.hmax = [max_hour_angle_deg(self.dec[i], self.lat, self.min_alt + ALT_MARGIN_DEG) for i in range(n)]
        self.factor = [0.0] * n
        self.best_score = [0.0] * n
        self.misses = [0] * n
        self.attempts = [0] * n
        self.active = [i for i in range(n) if self.hmax[i] > 0.0]

        self._cells: dict[int, list[tuple[float, int]]] = {}
        self._build_index()
        self.first_night, self.last_night = self._build_windows()

        self.scale = 1.0
        self.prior_scale = 1.0
        self.program_scale = 1.0
        self._program_history = deque(maxlen=4)
        self._saturation_floor = None
        self.quality_intervals = []
        self.efficiency_certificate = None
        # Fibres in one exposure share conditions; retain exposure-level samples.
        self._samples: deque = deque(maxlen=4)
        self._all_ratios: deque = deque(maxlen=25)
        self.clean_history: list[tuple[float, int, float]] = []  # (hours, night, ratio)
        self._ex_cache = None  # incremental exposure-level view of clean_history (see _exposures)
        self.pending_night = -1
        self._band_checks: deque = deque(maxlen=60)        # (program, matched, model)
        self.force_program: Optional[str] = None
        self.pending: dict[str, PendingPrediction] = {}
        self.pending_program = "BACKUP"
        self.pending_duration = 0
        self.blocked: list[tuple[float, float]] = []       # (az, alt) where a hit scored zero
        self.notices: set[str] = set()                      # "kind|direction"
        self.terrain: set[str] = set()
        self.extra_avoid: set[str] = set()
        self.duration_scale = 1.0
        self.fast_level = 0

        # Per-observe-action ledger: (observe_action_index, target_id, factor), mirroring
        # the backend's own BestLedger so a Hard-mode state_resync can be answered exactly
        # (see _resync) instead of only from the resync message's best_scores.
        self.ledger: list[tuple[int, str, float]] = []
        self.pending_action_index: Optional[int] = None
        self.pending_instrument_fault = False

    # -- spatial index -------------------------------------------------------

    def _build_index(self) -> None:
        for i in self.active:
            key = math.floor(self.dec[i])
            self._cells.setdefault(key, []).append((self.ra[i], i))
        for band in self._cells.values():
            band.sort(key=lambda pair: pair[0])

    def neighbours(self, ra: float, dec: float, radius: float):
        """Indices within `radius` degrees of (ra, dec), using the 1-degree declination-band index."""
        found: list[int] = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = radius / cos_dec
        lo_key, hi_key = math.floor(dec - radius), math.floor(dec + radius)
        for key in range(lo_key, hi_key + 1):
            band = self._cells.get(key)
            if not band:
                continue
            spans: list[tuple[float, float]]
            lo, hi = ra - width, ra + width
            if lo < 0:
                spans = [(0.0, hi), (lo + 360.0, 360.0)]
            elif hi >= 360:
                spans = [(lo, 360.0), (0.0, hi - 360.0)]
            else:
                spans = [(lo, hi)]
            keys = [r for r, _ in band]
            for low, high in spans:
                start = bisect.bisect_left(keys, low)
                end = bisect.bisect_right(keys, high)
                for k in range(start, end):
                    found.append(band[k][1])
        return found

    def _build_windows(self):
        """First/last night index on which each target has >=20 minutes above the limit."""
        need = 20 * 60 * SIDEREAL_DEG_PER_SECOND
        spans = []
        for start, end in self.nights:
            from .geometry import local_sidereal_deg
            l0 = local_sidereal_deg(start, self.lon)
            span = (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND
            spans.append((l0, span))
        n = len(self.ra)
        first_night = [len(self.nights)] * n
        last_night = [-1] * n
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = _mod(self.ra[i] - h - l0, 360.0)
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if first_night[i] > k:
                        first_night[i] = k
                    last_night[i] = k
        return first_night, last_night

    # -- messages and results -------------------------------------------------

    def on_messages(self, messages: list[dict], latest_bulletin: Optional[dict]) -> None:
        for message in messages:
            if message.get("record_type") == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        self.terrain.add(notice.get("direction"))
            elif message.get("record_type") == "state_resync":
                self._resync(message)
        notices = (latest_bulletin or {}).get("notices", [])
        self.notices = {f"{n.get('event_kind')}|{n.get('direction')}" for n in notices
                        if n.get("event_kind") != "terrain_obstruction"}

    def _resync(self, message: dict) -> None:
        """Hard-mode state_resync (participant guide, Appendix A / section 8): a prior
        window of `observe` actions was invalidated. The message itself only gives
        `best_scores` (score, not factor) for targets with any surviving valid hit --
        the guide is explicit that it does not return each target's completion factor,
        and that an agent that needs it exactly should combine `invalidated_window`
        with its own saved valid-exposure history.

        Each ledger entry stores our inferred factor for one past observe action.
        on_result uses the returned score, public weight and an inferred program
        match; that match is not guaranteed to recover the hidden scoring state.
        Dropping entries inside the invalidated action-index window and taking each
        target's maximum consistently rebuilds our internal estimates, rather than
        proving the backend's true factors have been recovered exactly.

        A target with no ledger entry at all (e.g. this
        process restarted mid-run and lost its in-memory history) falls back to the old
        best_score/top_multiplier estimate below, same as before this change.
        """
        self._saturation_floor = None
        self.quality_intervals = []
        self.efficiency_certificate = None
        window = message.get("invalidated_window") or {}
        start, end = window.get("action_index_start"), window.get("action_index_end_exclusive")
        if start is not None and end is not None:
            self.ledger = [entry for entry in self.ledger if not (start <= entry[0] < end)]
        else:
            self.ledger = []  # no window given: nothing in the ledger can be trusted
        exact: dict[str, float] = {}
        for _, target_id, factor in self.ledger:
            if factor > exact.get(target_id, 0.0):
                exact[target_id] = factor

        best_scores = message.get("best_scores") or []
        best: dict[str, float] = {}
        if best_scores and isinstance(best_scores[0], dict):
            for row in best_scores:
                best[row.get("target_id")] = float(row.get("best_score", 0.0))
        else:
            for target_id, score in zip(message.get("observed_target_ids", []), best_scores):
                best[target_id] = float(score)
        top_multiplier = max(self.scoring.program_multipliers.values()) if self.scoring.program_multipliers else 1.2

        for i in range(len(self.ids)):
            target_id = self.ids[i]
            self.best_score[i] = best.get(target_id, 0.0)
            if target_id in exact:
                self.factor[i] = exact[target_id]
                continue
            score = best.get(target_id, 0.0)
            self.factor[i] = min(1.0, score / (self.weight[i] * top_multiplier)) if score > 0 and self.weight[i] > 0 else 0.0
        self.active = [i for i in range(len(self.ids)) if self.hmax[i] > 0.0]
        self.pending.clear()
        self.pending_action_index = None

    def site_closed(self) -> bool:
        for key in self.notices:
            kind, _, direction = key.partition("|")
            if kind in ("rain", "storm") and direction == "ALL":
                return True
        return False

    def all_sky_notice(self) -> bool:
        return any(key.partition("|")[2] == "ALL" for key in self.notices)

    def _joint_quality_intervals(self, hits, shared=False):
        """Feasible shared quality under either program interpretation.

        Saturated hits supply lower bounds. Disjoint solutions remain disjoint;
        conflicting clean hits leave inference to the existing per-target model.
        """
        intervals = [(0.0, math.inf)]
        count = 0
        multipliers = (self.scoring.program_multipliers.get(self.pending_program, 1.0),
                       self.scoring.mismatch_multiplier)
        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            score = hits.get(target_id, 0.0)
            if i is None or not (prediction.clean or (shared and prediction.shared)) or score <= 0 or self.weight[i] <= 0:
                continue
            a = self.flux[i] * self.pending_duration * prediction.model / self.scoring.f0t0
            if a <= 0:
                continue
            y = score / self.weight[i]
            # Reuse the score tolerance already used to identify saturation.
            low, high = max(0.0, y - 2e-4), y + 2e-4
            alternatives = []
            for multiplier in multipliers:
                if multiplier > 0 and low <= multiplier:
                    alternatives.append((low / (multiplier * a),
                                         high / (multiplier * a) if high < multiplier else math.inf))
            intersections = []
            for lo, hi in intervals:
                for lower, upper in alternatives:
                    left, right = max(lo, lower), min(hi, upper)
                    if left <= right:
                        intersections.append((left, right))
            intervals = []
            for lo, hi in sorted(intersections):
                if intervals and lo <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(hi, intervals[-1][1]))
                else:
                    intervals.append((lo, hi))
            if not intervals:
                return []
            count += 1
        return intervals if count >= 2 else []

    def program_band(self, model: float) -> str:
        # Completion efficiency and atmospheric program quality are distinct.
        return self.scoring.program_band(model * self.program_scale / 0.95)

    def update_program_scale(self, hours: float) -> None:
        active = [(when, ranges) for when, ranges in self._program_history
                  if when >= hours - SKY_MEMORY_HOURS]
        self._program_history = deque(active, maxlen=4)
        if not active:
            self.program_scale = 1.0
            return
        ranges = active[-1][1]
        for _, earlier in reversed(active[:-1]):
            combined = _intersect_ranges(ranges, earlier)
            if not combined:
                break  # Latest identifiable feedback takes precedence after a shift.
            ranges = combined
        choices = []
        for lower, upper in ranges:
            cap = upper - min(1e-6 * max(1.0, upper), (upper - lower) / 2) if math.isfinite(upper) else upper
            projected = max(lower, min(self.program_scale, cap))
            choices.append((abs(projected - self.program_scale), projected))
        self.program_scale = min(choices)[1]

    def calibrate_program(self, hits, hours: float) -> None:
        quality = self._joint_quality_intervals(hits, shared=True)
        ranges, labels = [(0.0, math.inf)], 0
        declared = self.scoring.program_multipliers[self.pending_program]
        mismatch = self.scoring.mismatch_multiplier
        for name, prediction in self.pending.items():
            i = self.index_of.get(name)
            if i is None or self.weight[i] <= 0 or prediction.band_model <= 0:
                continue
            y = hits.get(name, 0.0) / self.weight[i]
            matched = y > mismatch + 2e-4
            if not matched:
                a = self.flux[i] * self.pending_duration * prediction.model / self.scoring.f0t0
                if y <= 0 or a <= 0 or not quality or not (prediction.clean or prediction.shared):
                    continue
                # A match below its saturation ceiling requires this bounded
                # quality interval. Exclude it only if the whole exposure does.
                match_range = [(max(0, y - 2e-4) / (declared * a), (y + 2e-4) / (declared * a))]
                if y + 2e-4 >= declared or any(max(lo, low) <= min(hi, high)
                        for lo, hi in quality for low, high in match_range):
                    continue  # Ambiguous feedback is not a mismatch label.
            dark = self.scoring.program_bands["DARK"] / prediction.band_model
            bright = self.scoring.program_bands["BRIGHT"] / prediction.band_model
            if self.pending_program == "DARK":
                allowed = [(dark, math.inf)] if matched else [(0.0, dark)]
            elif self.pending_program == "BRIGHT":
                allowed = [(bright, dark)] if matched else [(0.0, bright), (dark, math.inf)]
            else:
                allowed = [(0.0, bright)] if matched else [(bright, math.inf)]
            ranges = _intersect_ranges(ranges, allowed)
            labels += 1
        if labels and ranges:
            self._program_history.append((hours, ranges))
        self.update_program_scale(hours)

    def on_result(self, last_result: Optional[dict], hours: float) -> None:
        action_index = self.pending_action_index
        self.pending_action_index = None
        instrument_fault = self.pending_instrument_fault
        self.pending_instrument_fault = False
        if last_result and last_result.get("action") == "report":
            if last_result.get("correct") is True:
                self.false_reports_since_correct = 0
                self.forget_quality_history()
            elif last_result.get("correct") is False:
                self.false_reports_since_correct += 1
        if not last_result or last_result.get("action") != "observe" or not self.pending:
            self.pending.clear()
            return
        hits = {h.get("target_id"): float(h.get("score", 0.0)) for h in last_result.get("hits", [])}
        self.calibrate_program(hits, hours)
        any_positive = any(score > 0 for score in hits.values())
        scoring = self.scoring
        multipliers = scoring.program_multipliers
        mismatch = scoring.mismatch_multiplier
        declared_multiplier = multipliers.get(self.pending_program, 1.0)
        f0t0 = scoring.f0t0
        censored = []
        exposure_ratios = []
        reference_ratios = []
        efficiency_uppers = []
        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            score = hits.get(target_id, 0.0)
            if i is None or self.weight[i] <= 0 or declared_multiplier <= 0:
                continue
            factor = score / (self.weight[i] * declared_multiplier)
            y_low = (score - 0.5e-6) / self.weight[i]
            factor_upper = (score + 0.5e-6) / (self.weight[i] * declared_multiplier)
            if (self.pending_program == "DARK" and prediction.shared and y_low > mismatch
                    and factor_upper < 0.97 and self.flux[i] > 0 and self.pending_duration > 0):
                quality_upper = factor_upper * f0t0 / (self.flux[i] * self.pending_duration)
                efficiency_uppers.append(quality_upper / scoring.program_bands["DARK"])
            denom = self.flux[i] * self.pending_duration * prediction.model
            # Above the mismatch ceiling, an unsaturated match is identifiable.
            if (prediction.clean and score / self.weight[i] > mismatch + 2e-4
                    and 0 < factor < 0.97 and denom > 0):
                reference_ratios.append(factor * f0t0 / denom)
        if efficiency_uppers:
            self.efficiency_certificate = (hours, min(efficiency_uppers))
        shared_scale = median(reference_ratios) if reference_ratios else None
        self.quality_intervals = self._joint_quality_intervals(hits)
        if len(self.quality_intervals) == 1 and math.isfinite(self.quality_intervals[0][1]):
            # Only a single bounded solution can resolve program ambiguity.
            shared_scale = sum(self.quality_intervals[0]) / 2.0

        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            if i is None:
                continue
            if target_id not in hits:
                if not instrument_fault:
                    self.misses[i] += 1
                continue
            score = hits[target_id]
            self.best_score[i] = max(self.best_score[i], score)
            if score <= 0.0:
                if any_positive and not instrument_fault:
                    self.blocked.append((prediction.az, prediction.alt))
                continue
            weight = self.weight[i] if self.weight[i] > 0 else 1e-9
            multiplier_seen = score / weight
            if prediction.clean:
                if abs(multiplier_seen - declared_multiplier) < 2e-4:
                    self._band_checks.append((self.pending_program, True, prediction.model))
                elif abs(multiplier_seen - mismatch) < 2e-4:
                    self._band_checks.append((self.pending_program, False, prediction.model))
            factor_if_match = score / (weight * declared_multiplier) if declared_multiplier > 0 else 0.0
            factor_if_miss = score / (weight * mismatch) if mismatch > 0 else 0.0
            ratio_match = (factor_if_match * f0t0) / (self.flux[i] * self.pending_duration * prediction.model) \
                if self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0 else 0.0
            band = self.program_band(prediction.model)
            # Above the mismatch score ceiling, factor <= 1 proves a match.
            matched = multiplier_seen > mismatch + 2e-4 or band == self.pending_program
            if (shared_scale is not None and prediction.clean and mismatch > 0
                    and multiplier_seen <= mismatch + 2e-4):
                ratio_miss = ratio_match * declared_multiplier / mismatch
                matched = abs(ratio_match - shared_scale) <= abs(ratio_miss - shared_scale)
            factor = factor_if_match if matched else factor_if_miss
            factor = min(1.0, factor)
            previous_factor = self.factor[i]
            self.factor[i] = max(previous_factor, factor)
            if action_index is not None:
                self.ledger.append((action_index, target_id, factor))
            if (not instrument_fault and self.required[i] and self.factor[i] < scoring.required_threshold
                    and (previous_factor == 0 or factor <= previous_factor)):
                # Keep prior failures, but do not add backoff for an improving retry.
                self.attempts[i] += 1
            # A saturated matching score gives a lower bound, not an exact sky sample.
            if (prediction.clean and abs(multiplier_seen - declared_multiplier) < 2e-4
                    and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0):
                censored.append(factor_if_match * f0t0 / (self.flux[i] * self.pending_duration * prediction.model))
            if factor < 0.97 and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                ratio = (factor * f0t0) / (self.flux[i] * self.pending_duration * prediction.model)
                exposure_ratios.append(ratio)
                if prediction.clean:
                    self.clean_history.append((hours, self.pending_night, ratio))
        if exposure_ratios:
            ratio = median(exposure_ratios)
            self._samples.append((hours, ratio))
            self._all_ratios.append(ratio)
        if len(self.quality_intervals) == 1 and math.isinf(self.quality_intervals[0][1]):
            censored.append(self.quality_intervals[0][0])
        self._saturation_floor = (hours, max(censored)) if censored else None
        self.pending.clear()
        self.update_scale(hours)

    def has_efficiency_deficit(self, hours: float) -> bool:
        if self.efficiency_certificate is None:
            return False
        when, upper = self.efficiency_certificate
        return 0 <= hours - when <= SKY_MEMORY_HOURS and upper < 0.8

    def has_recent_sample(self, hours: float) -> bool:
        return any(when >= hours - SKY_MEMORY_HOURS for when, _ in self._samples)

    def update_scale(self, hours: float) -> None:
        self.update_program_scale(hours)
        if len(self._all_ratios) >= 2:
            ordered = sorted(self._all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self._samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if recent else self.prior_scale
        if self._saturation_floor and self._saturation_floor[0] >= hours - SKY_MEMORY_HOURS:
            self.scale = max(self.scale, (self.scale + self._saturation_floor[1]) / 2.0)

    # -- fault diagnostics ------------------------------------------------------

    def _exposures(self):
        """Exposure rows of clean_history: [hour, night, ratios, cached median], in first-appearance order of (hour, night).

        Identical to regrouping the whole list on every call, but only entries added since the last call are
        aggregated. A different list object (repair) or a shorter one rebuilds from scratch, so nothing is truncated
        and the repair/resync behaviour is unchanged. The runtime only appends to clean_history or replaces it; an
        in-place edit of existing entries is not a supported operation and is not detected."""
        history = self.clean_history
        cache = self._ex_cache
        if cache is None or cache[0] is not history or cache[1] > len(history):
            cache = self._ex_cache = [history, 0, [], {}]
        rows, index = cache[2], cache[3]
        for k in range(cache[1], len(history)):
            hour, night, ratio = history[k]
            row = index.get((hour, night))
            if row is None:
                row = index[(hour, night)] = [hour, night, [], None]
                rows.append(row)
            row[2].append(ratio)
            row[3] = None
        cache[1] = len(history)
        return rows

    @staticmethod
    def _row_median(row):
        if row[3] is None:
            row[3] = median(row[2])
        return row[3]

    def clean_by_hour(self):
        """-> (sorted distinct hours, {hour: ratios}) over clean_history, grouped by hour as the earthquake probe needs."""
        clean = {}
        for row in self._exposures():
            clean.setdefault(row[0], []).extend(row[2])
        return sorted(clean), clean

    def fault_evidence(self, rapid: bool = False) -> Optional[FaultEvidence]:
        history = self.clean_history
        if len(history) < (24 if rapid else RECENT_SAMPLES + EARLIER_SAMPLES):
            return None
        # Fibres in one exposure share weather: count an exposure once. Select
        # recent observations by observing nights, not by a fixed number of hits.
        rows = self._exposures()
        med = self._row_median
        if rapid:
            if len(rows) < 24:
                return None
            recent = [(r[0], r[1], med(r)) for r in rows[-6:]]
            earlier = [(r[0], r[1], med(r)) for r in rows[-30:-6]]
        else:
            recent_nights = sorted({r[1] for r in rows})[-2:]
            recent = [(r[0], r[1], med(r)) for r in rows if r[1] in recent_nights]
            earlier = [(r[0], r[1], med(r)) for r in rows if r[1] not in recent_nights]
            if len(recent) < 8 or len(earlier) < 8:
                return None
        span = recent[-1][0] - recent[0][0]
        nights = len({night for _, night, _ in recent})
        if (span < 0.5 if rapid else span < 4.0 or nights < 2):
            return None
        recent_median = median(r for _, _, r in recent)
        earlier_median = median(r for _, _, r in earlier)
        dark_line = self.scoring.program_bands["DARK"] * 1.3
        dark = [c for c in list(self._band_checks)[-16:]
                if c[0] == "DARK" and (c[2] * earlier_median) / 0.95 >= dark_line]
        return FaultEvidence(
            recent_median=round(recent_median, 3),
            earlier_median=round(earlier_median, 3),
            drop=round(recent_median / max(1e-9, earlier_median), 3),
            recent_samples=len(recent),
            recent_nights=nights,
            earlier_samples=len(earlier),
            dark_checks=len(dark),
            dark_matched=sum(1 for c in dark if c[1]),
        )

    def forget_quality_history(self) -> None:
        self._saturation_floor = None
        self.quality_intervals = []
        self.efficiency_certificate = None
        self.clean_history = []
        self._ex_cache = None
        self._band_checks.clear()
        self._samples.clear()
        self._all_ratios.clear()
        self.prior_scale = 1.0

    # -- night lookup -------------------------------------------------------------

    def current_night(self, now):
        for index, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return index, start, end
        return None

    def next_night_start(self, now):
        for start, _end in self.nights:
            if start > now:
                return start
        return None
