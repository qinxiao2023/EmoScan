#!/usr/bin/env python3
"""
Lightweight ByteTrack-like multi-object tracker.

Notes:
- This is an "equivalent" practical implementation for this project:
  - two-threshold association using detection scores (high/low)
  - IoU-based matching
  - simple constant-velocity prediction (no external deps)
- It is not a full reference ByteTrack implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math


Box = Tuple[float, float, float, float]  # xyxy


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _greedy_match(iou_matrix: List[List[float]], iou_threshold: float) -> List[Tuple[int, int]]:
    """
    Greedy bipartite matching on IoU.
    Returns list of (row_idx, col_idx) pairs.
    """
    if not iou_matrix:
        return []
    num_rows = len(iou_matrix)
    num_cols = len(iou_matrix[0]) if num_rows > 0 else 0
    pairs: List[Tuple[int, int, float]] = []
    for r in range(num_rows):
        for c in range(num_cols):
            v = iou_matrix[r][c]
            if v >= iou_threshold:
                pairs.append((r, c, v))
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_r = set()
    used_c = set()
    out: List[Tuple[int, int]] = []
    for r, c, _ in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        out.append((r, c))
    return out


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Box
    score: float
    age: int = 0
    time_since_update: int = 0
    hits: int = 1
    confirmed: bool = False
    vx: float = 0.0
    vy: float = 0.0

    def predict(self) -> None:
        # Constant-velocity prediction of center; keep size.
        x1, y1, x2, y2 = self.bbox_xyxy
        cx, cy = _center(self.bbox_xyxy)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx2 = cx + self.vx
        cy2 = cy + self.vy
        self.bbox_xyxy = (cx2 - w / 2, cy2 - h / 2, cx2 + w / 2, cy2 + h / 2)

    def update(self, det_box: Box, det_score: float) -> None:
        prev_cx, prev_cy = _center(self.bbox_xyxy)
        new_cx, new_cy = _center(det_box)
        self.vx = new_cx - prev_cx
        self.vy = new_cy - prev_cy
        self.bbox_xyxy = det_box
        self.score = det_score
        self.hits += 1
        self.time_since_update = 0


class BYTETracker:
    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        iou_thresh: float = 0.3,
        max_time_lost: int = 30,
        min_hits: int = 3,
    ):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_time_lost = int(max_time_lost)
        self.min_hits = int(min_hits)

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, bbox: Box, score: float) -> Track:
        tid = self._next_id
        self._next_id += 1
        return Track(track_id=tid, bbox_xyxy=bbox, score=score, confirmed=False)

    def update(self, detections: Sequence[Tuple[Box, float]]) -> List[Track]:
        """
        detections: list of (bbox_xyxy, score)
        returns: active tracks (confirmed + recently updated unconfirmed)
        """
        # Predict existing tracks
        for t in self.tracks.values():
            t.age += 1
            t.time_since_update += 1
            t.predict()

        det_high: List[Tuple[Box, float]] = [(b, s) for (b, s) in detections if s >= self.high_thresh]
        det_low: List[Tuple[Box, float]] = [(b, s) for (b, s) in detections if self.low_thresh <= s < self.high_thresh]

        track_list = list(self.tracks.values())

        # Stage 1: match high score detections to all tracks
        matches1, unmatched_tracks_idx, unmatched_det_high_idx = self._associate(track_list, det_high)

        for ti, di in matches1:
            track_list[ti].update(det_high[di][0], det_high[di][1])

        # Stage 2: match remaining tracks to low score detections (recover)
        remaining_tracks = [track_list[i] for i in unmatched_tracks_idx]
        matches2, unmatched_tracks2_idx, unmatched_det_low_idx = self._associate(remaining_tracks, det_low)

        for ti, di in matches2:
            remaining_tracks[ti].update(det_low[di][0], det_low[di][1])

        # Create new tracks from unmatched high detections
        for di in unmatched_det_high_idx:
            b, s = det_high[di]
            nt = self._new_track(b, s)
            self.tracks[nt.track_id] = nt

        # Confirm tracks & remove lost ones
        to_delete = []
        for tid, t in self.tracks.items():
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.time_since_update > self.max_time_lost:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Return tracks that are alive (including unconfirmed but recently updated)
        active = [t for t in self.tracks.values() if t.time_since_update == 0 or t.confirmed]
        return active

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Tuple[Box, float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            unmatched_t = list(range(len(tracks)))
            unmatched_d = list(range(len(detections)))
            return [], unmatched_t, unmatched_d

        iou_matrix: List[List[float]] = []
        for t in tracks:
            row = []
            for (b, _s) in detections:
                row.append(_iou(t.bbox_xyxy, b))
            iou_matrix.append(row)

        matches = _greedy_match(iou_matrix, self.iou_thresh)
        matched_t = {t for t, _ in matches}
        matched_d = {d for _, d in matches}
        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
        return matches, unmatched_t, unmatched_d

