from __future__ import annotations

from typing import Iterable

import numpy as np


class SimpleCentroidTracker:
    def __init__(self, max_distance: float = 60.0) -> None:
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def update(self, bboxes: Iterable[tuple[float, float, float, float]]) -> dict[int, tuple[float, float]]:
        detections = [self._center(b) for b in bboxes]
        if not detections:
            self.objects = {}
            return {}

        if not self.objects:
            self.objects = {self.next_id + i: c for i, c in enumerate(detections)}
            self.next_id += len(detections)
            return dict(self.objects)

        existing_ids = list(self.objects.keys())
        existing_pts = np.array([self.objects[i] for i in existing_ids], dtype=np.float32)
        det_pts = np.array(detections, dtype=np.float32)

        used_det = set()
        new_objects: dict[int, tuple[float, float]] = {}
        for obj_id, obj_pt in zip(existing_ids, existing_pts):
            dists = np.linalg.norm(det_pts - obj_pt, axis=1)
            best_idx = int(np.argmin(dists))
            if best_idx in used_det or dists[best_idx] > self.max_distance:
                continue
            used_det.add(best_idx)
            new_objects[obj_id] = tuple(map(float, det_pts[best_idx]))

        for i, det in enumerate(detections):
            if i in used_det:
                continue
            new_objects[self.next_id] = det
            self.next_id += 1

        self.objects = new_objects
        return dict(self.objects)


class SimpleBallTracker:
    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []

    def update(self, detections: Iterable[tuple[float, float, float, float]]) -> tuple[float, float] | None:
        dets = list(detections)
        if not dets:
            return self.history[-1] if self.history else None
        x1, y1, x2, y2 = dets[0]
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        self.history.append(center)
        if len(self.history) > 256:
            self.history = self.history[-256:]
        return center
