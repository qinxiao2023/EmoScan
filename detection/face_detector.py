#!/usr/bin/env python3
"""
Face detection module that outputs bounding boxes with confidence scores.

This is used to feed trackers (e.g., ByteTrack) which rely on detection scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection in pixel coordinates."""

    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float


class MediaPipeFaceDetector:
    """
    MediaPipe face detector wrapper.

    Output bboxes are clipped to image bounds and returned as integer pixels.
    """

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        # Lazy import so the rest of the system can still run without mediapipe installed,
        # though tracking will be unavailable.
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mediapipe is required for MediaPipeFaceDetector. "
                "Install it via: pip install mediapipe"
            ) from e

        # MediaPipe has two major APIs in the wild:
        # - legacy: mp.solutions.face_detection.FaceDetection
        # - tasks:  mp.tasks (requires model asset paths)
        #
        # For maximum compatibility, we support legacy solutions when present;
        # otherwise we fail with a clear message so callers can choose another detector.
        self._mp = mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build does not expose 'mediapipe.solutions'. "
                "Please install a legacy mediapipe version with solutions API, "
                "or use FERFaceDetectorWithScore instead."
            )

        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(frame_rgb)

        detections: List[Detection] = []
        if not results or not results.detections:
            return detections

        for det in results.detections:
            score = float(det.score[0]) if det.score else 0.0
            bbox = det.location_data.relative_bounding_box
            x1 = int(round(bbox.xmin * w))
            y1 = int(round(bbox.ymin * h))
            x2 = int(round((bbox.xmin + bbox.width) * w))
            y2 = int(round((bbox.ymin + bbox.height) * h))

            # Clip
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(Detection(bbox_xyxy=(x1, y1, x2, y2), score=score))

        return detections


class FERFaceDetectorWithScore:
    """
    Face detector based on the already-used FER pipeline.

    It returns:
    - bbox from FER face detection
    - score as max emotion probability (a practical proxy to drive score-based tracking)
    """

    def __init__(self, fer_model):
        self._fer = fer_model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if self._fer is None:
            return []

        results = self._fer.detect_emotions(frame_bgr)
        out: List[Detection] = []
        for r in results or []:
            box = r.get("box", None)
            emos = r.get("emotions", {}) or {}
            if box is None:
                continue
            # FER may return list/tuple or numpy array for box; normalize safely.
            if isinstance(box, np.ndarray):
                box = box.tolist()
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x, y, w, h = box
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            score = float(max(emos.values()) if emos else 0.0)
            out.append(Detection(bbox_xyxy=(x1, y1, x2, y2), score=score))
        return out


def create_face_detector(fer_model):
    """
    Create a detector that outputs bbox + score.

    Preference:
    - MediaPipe solutions API when available
    - else: FER-based detector (works without external models beyond FER)
    """
    try:
        return MediaPipeFaceDetector(min_detection_confidence=0.5, model_selection=0)
    except Exception:
        return FERFaceDetectorWithScore(fer_model=fer_model)

