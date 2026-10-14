"""Infer a constant alt/az pointing bias from this run's geometric hit feedback.

Finite-grid consistency search, not a certified interval enclosure or a
calibrated probability. Zero-score hits still count as geometrical hits.
"""
from collections import deque
from .geometry import tangent_offsets


class PointingCalibration:
    def __init__(self, grid):
        self.grid = grid
        self.offset = (0.0, 0.0)
        self.batches = deque(maxlen=8)
        self.pending = None
        self.exposures = 0

    def remember(self, action, samples, action_index):
        self.pending = (action_index, action["pointing"]["alt_deg"],
                        action["pointing"]["az_deg"], len(action["assignments"]), samples)

    def update(self, result):
        pending, self.pending = self.pending, None
        if pending is None or not result or result.get("action") != "observe":
            return False
        index, alt, az, assigned, samples = pending
        if (result.get("assigned_count") != assigned or
                (index is not None and result.get("observe_index", index) != index)):
            return False
        hits = {row["target_id"] for row in result.get("hits", [])}
        batch = [(fiber, ta, tz, target in hits) for fiber, target, ta, tz in samples]
        if not batch:
            return False
        self.batches.append((alt, az, batch))
        self.exposures += 1
        size = sum(len(batch) for _, _, batch in self.batches)
        if self.exposures % 4 or len(self.batches) < 3 or size < 24:
            return False

        def loss(offset):
            da, dz = offset
            errors = 0
            for ca, cz, observations in self.batches:
                actual_alt, actual_az = ca + da, (cz + dz) % 360
                for fiber, ta, tz, hit in observations:
                    delta = tangent_offsets(ta, tz, actual_alt, actual_az) if 0 <= actual_alt <= 90 else None
                    predicted = delta is not None and self.grid.classify(*delta)[0] == fiber
                    errors += predicted != hit
            return errors

        previous_loss = loss(self.offset)
        if previous_loss < 2:
            return False
        step = self.grid.pitch / 8
        candidates = [(i * step, j * step) for i in range(-8, 9) for j in range(-8, 9)]
        candidates.append(self.offset)
        # Minimum displacement breaks ambiguous ties conservatively toward zero.
        key = lambda pair: (loss(pair), pair[0] ** 2 + pair[1] ** 2)
        best = min(candidates, key=key)
        for _ in range(2):
            step /= 4
            local = [(best[0] + i * step, best[1] + j * step)
                     for i in range(-4, 5) for j in range(-4, 5)]
            local = [pair for pair in local if max(abs(pair[0]), abs(pair[1])) <= self.grid.pitch]
            best = min(local + [self.offset], key=key)
        best_loss = loss(best)
        if best_loss <= max(1, int(size * 0.05)) and previous_loss - best_loss >= 2:
            self.offset = best
            return True
        return False
