from collections import Counter, deque
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    total = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / total if total > 0 else 0.0

@dataclass
class Track:
    id: int
    label: str
    box: list
    confidence: float
    last_frame: int
    velocity: tuple = (0.0, 0.0)
    missed: int = 0
    hits: int = 1
    path: deque = field(default_factory=lambda: deque(maxlen=50))
    stable_side: int = 0
    directions: set = field(default_factory=set)

class Tracker:
    """Class-gated Hungarian matching with centroid motion and IoU. Not identity recognition."""
    def __init__(self, width, line_position=0.5, max_missed=8, max_distance=100, min_hits=3):
        self.width = width
        self.line_x = width * line_position
        self.margin = max(4, width * 0.01)
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 1
        self.unique = Counter()
        self.crossings = Counter()

    def _side(self, x):
        if x < self.line_x - self.margin: return -1
        if x > self.line_x + self.margin: return 1
        return 0

    def _count(self, t, frame, events):
        if t.hits == self.min_hits:
            self.unique[t.label] += 1
        if t.hits < self.min_hits:
            return
        side = self._side(center(t.box)[0])
        if side and t.stable_side and side != t.stable_side:
            direction = "left_to_right" if side == 1 else "right_to_left"
            if direction not in t.directions:
                t.directions.add(direction)
                self.crossings[(t.label, direction)] += 1
                events.append({"frame": frame, "track_id": t.id, "label": t.label, "direction": direction})
        if side:
            t.stable_side = side

    def update(self, detections, frame):
        tracks = list(self.tracks.values())
        assigned_tracks, assigned_dets = set(), set()
        events = []
        if tracks and detections:
            # Dummy columns let a track remain unmatched rather than accepting a gated pair.
            costs = np.full((len(tracks), len(detections)+len(tracks)), 1.5)
            for i, t in enumerate(tracks):
                c = center(t.box)
                dt = frame-t.last_frame
                predicted = (c[0]+t.velocity[0]*dt, c[1]+t.velocity[1]*dt)
                for j, d in enumerate(detections):
                    dc = center(d["box"])
                    distance = math.dist(predicted, dc)
                    if t.label != d["label"] or distance > self.max_distance:
                        costs[i,j] = 1000
                    else:
                        costs[i,j] = distance/self.max_distance + 0.35*(1-iou(t.box,d["box"]))
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if j >= len(detections) or costs[i,j] >= 1.5:
                    continue
                t, d = tracks[i], detections[j]
                old, new = center(t.box), center(d["box"])
                dt = max(1, frame-t.last_frame)
                t.velocity = ((new[0]-old[0])/dt, (new[1]-old[1])/dt)
                t.box, t.confidence = list(d["box"]), float(d["confidence"])
                t.last_frame, t.missed = frame, 0
                t.hits += 1
                t.path.append(new)
                self._count(t, frame, events)
                assigned_tracks.add(t.id); assigned_dets.add(j)
        for t in tracks:
            if t.id not in assigned_tracks:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[t.id]
        for j, d in enumerate(detections):
            if j not in assigned_dets:
                t = Track(self.next_id, d["label"], list(d["box"]), float(d["confidence"]), frame)
                t.path.append(center(t.box))
                self.tracks[t.id] = t
                self.next_id += 1
                self._count(t, frame, events)
        visible = [t for t in self.tracks.values() if t.missed == 0 and t.hits >= self.min_hits]
        return visible, events

    def summary(self):
        return {"unique_tracks": dict(self.unique), "crossings": [
            {"label": label, "direction": direction, "count": count}
            for (label,direction),count in sorted(self.crossings.items())]}
