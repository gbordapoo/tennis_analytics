from __future__ import annotations


class SimpleBallTracker:
    def __init__(self) -> None:
        self.next_id = 0
        self.objects = {}

    def update(self, detections):
        tracked = {}
        for det in detections:
            tracked[self.next_id] = det
            self.next_id += 1
        self.objects = tracked
        return tracked
