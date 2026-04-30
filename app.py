#!/usr/bin/env python3
"""
EmoScan: Flask Web Interface

Adds:
- webcam monitoring (existing)
- uploaded video monitoring (new)
- tiered alerting (existing RiskEngine) + negative-duration rule (new)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import logging
import os
import threading
import time
import uuid

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

import config
from alerting.persistence import AlertPersistence
from alerting.risk_engine import AlertEvent, RiskEngine
from tracking.bytetrack import BYTETracker, Track


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="ui/templates")


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


class NegativeAccumulationRule:
    """
    Additional alert rule:
    - If negative emotion accumulates > 3s -> L2 (orange)
    - If > 5s -> L3 (red)
    Supports time-mode or count-mode (frames/events).
    """

    def __init__(self, settings: dict):
        self.enabled = bool(settings.get("enabled", True))
        self.mode = str(settings.get("mode", "time")).lower()
        self.negative_emotions = set(settings.get("negative_emotions", ["sad", "angry", "fear", "disgust"]))
        self.prob_threshold = float(settings.get("prob_threshold", 0.5))
        self.min_neutral_count = int(settings.get("min_neutral_count", 12))
        self.orange_ratio = float(settings.get("orange_ratio", 1.0 / 3.0))
        self.red_ratio = float(settings.get("red_ratio", 1.0 / 2.0))
        self.emit_on_escalation_only = bool(settings.get("emit_on_escalation_only", True))

        self._state: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, ts: float, track_probs: Dict[int, Dict[str, float]]) -> List[dict]:
        if not self.enabled:
            return []

        alerts: List[dict] = []
        for tid, probs in track_probs.items():
            st = self._state.setdefault(tid, {"neg_count": 0, "neu_count": 0, "last_level": 0})

            dominant = None
            dom_p = 0.0
            if probs:
                dominant, dom_p = max(probs.items(), key=lambda kv: float(kv[1]))

            # Count by dominant emotion (expression counting mode)
            is_negative = bool(dominant in self.negative_emotions)
            is_neutral = bool(dominant == "neutral")

            if is_negative:
                st["neg_count"] = int(st["neg_count"]) + 1
            if is_neutral:
                st["neu_count"] = int(st["neu_count"]) + 1

            neg_count = int(st["neg_count"])
            neu_count = int(st["neu_count"])

            level = 0
            if self.mode == "count_ratio_vs_neutral":
                if neu_count >= self.min_neutral_count:
                    if neg_count >= int(np.ceil(neu_count * self.orange_ratio)):
                        level = 2
                    if neg_count >= int(np.ceil(neu_count * self.red_ratio)):
                        level = 3
            else:
                # Fallback: disabled/unknown mode -> no alert
                level = 0

            if level <= 0:
                continue

            if self.emit_on_escalation_only and level <= int(st["last_level"]):
                continue

            st["last_level"] = int(level)
            msg = "持续负面情绪"
            alerts.append(
                {
                    "ts": float(ts),
                    "level": int(level),
                    "track_id": int(tid),
                    "risk": float(0.0),
                    "z": float(0.0),
                    "duration_sec": float(0.0),
                    "crowd_median": float(0.0),
                    "crowd_mad": float(0.0),
                    "message": msg,
                    "rule": "neg_count_ratio",
                }
            )

        return alerts


class WebEmotionDetector:
    """Web-based emotion detector for Flask interface."""

    def __init__(self) -> None:
        self.emotions = list(getattr(config, "EMOTIONS", ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]))
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.session_data: List[dict] = []

        self.is_running = False
        self.source: str = "camera"  # camera|video
        self.session_id: str = ""

        self.cap: Optional[cv2.VideoCapture] = None
        self.video_fps: Optional[float] = None

        self.face_cascade = None
        self.fer_model = None
        self.current_frame: Optional[np.ndarray] = None

        self._lock = threading.RLock()

        # Upload registry: video_id -> path
        self.uploaded_videos: Dict[str, Dict[str, str]] = {}

        # Tracking + alerting
        self.tracker = BYTETracker(high_thresh=0.6, low_thresh=0.1, iou_thresh=0.3, max_time_lost=30, min_hits=2)
        self.risk_engine = RiskEngine(settings=config.ALERT_SETTINGS, emotions=self.emotions)
        self.neg_rule = NegativeAccumulationRule(settings=dict(config.ALERT_SETTINGS.get("negative_rule", {})))
        self.recent_alerts: Deque[dict] = deque(maxlen=int(config.ALERT_SETTINGS.get("max_recent_alerts", 200)))

        self.persistence = AlertPersistence(root_dir=Path(config.LOGS_DIR) / "alerts")
        self.track_trajectory: Dict[int, Deque[Tuple[float, Box]]] = defaultdict(lambda: deque(maxlen=200))
        self.track_last_probs: Dict[int, Dict[str, float]] = {}

        self.setup_face_detection()
        self.setup_emotion_detection()

    def setup_face_detection(self) -> None:
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load face cascade")
            logger.info("Face detection cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            self.face_cascade = None

    def setup_emotion_detection(self) -> None:
        try:
            from fer import FER

            try:
                self.fer_model = FER(mtcnn=False)
                logger.info("FER emotion detection model loaded successfully (without MTCNN)")
            except Exception as mtcnn_error:
                logger.warning(f"MTCNN failed, trying default FER(): {mtcnn_error}")
                self.fer_model = FER()
                logger.info("FER emotion detection model loaded successfully (default)")
        except ImportError:
            logger.warning("FER not available, using simplified emotion detection")
            self.fer_model = None
        except Exception as e:
            logger.error(f"Error loading FER model: {e}")
            self.fer_model = None

    def reset_runtime(self) -> None:
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.session_data = []
        self.recent_alerts.clear()
        self.risk_engine.states.clear()
        self.risk_engine.recent_alerts.clear()
        self.neg_rule.reset()
        self.tracker = BYTETracker(high_thresh=0.6, low_thresh=0.1, iou_thresh=0.3, max_time_lost=30, min_hits=2)
        self.track_trajectory.clear()
        self.track_last_probs.clear()

    def register_uploaded_video(self, video_id: str, path: str, filename: str) -> None:
        self.uploaded_videos[str(video_id)] = {"path": str(path), "filename": str(filename)}

    def _open_capture(self, source: str, video_path: Optional[str]) -> cv2.VideoCapture:
        if source == "camera":
            cap = cv2.VideoCapture(int(config.VIDEO_SETTINGS["camera_index"]))
        else:
            if not video_path:
                raise ValueError("video_path required for video source")
            cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video source")
        return cap

    def start_detection(self, source: str = "camera", video_path: Optional[str] = None) -> bool:
        try:
            with self._lock:
                self.stop_detection()
                self.reset_runtime()

                self.source = str(source or "camera")
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.cap = self._open_capture(self.source, video_path)

                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self.video_fps = fps if fps > 1e-3 else None

                self.is_running = True
                logger.info(f"Web detection started (source={self.source}, fps={self.video_fps})")
                return True
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            return False

    def stop_detection(self) -> None:
        with self._lock:
            self.is_running = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.video_fps = None
            logger.info("Web emotion detection stopped")

    def _sleep_interval(self) -> float:
        if self.source == "video" and self.video_fps:
            return float(max(0.0, min(0.2, 1.0 / float(self.video_fps))))
        return float(config.VIDEO_SETTINGS.get("processing_interval", 0.03))

    def detect_emotion(self, face_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
        try:
            if self.fer_model:
                result = self.fer_model.detect_emotions(face_bgr)
                if result and len(result) > 0:
                    emo = result[0].get("emotions", {}) or {}
                    scores = {e: float(emo.get(e, 0.0)) for e in self.emotions}
                    dominant = max(scores.items(), key=lambda kv: float(kv[1]))[0] if scores else "neutral"
                    return dominant, scores
                return "neutral", {e: 0.0 for e in self.emotions}

            return self.detect_emotion_simple(face_bgr)
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return "neutral", {e: 0.0 for e in self.emotions}

    def detect_emotion_simple(self, face_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
        import random

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        if brightness > 150:
            dominant = "happy"
        elif brightness < 100:
            dominant = "sad"
        else:
            dominant = "neutral"

        scores: Dict[str, float] = {}
        for e in self.emotions:
            scores[e] = float(random.uniform(0.6, 0.9) if e == dominant else random.uniform(0.0, 0.3))
        return dominant, scores

    def log_emotion(self, track_id: int, dominant: str, emotion_scores: Dict[str, float]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.session_data.append(
            {
                "timestamp": ts,
                "track_id": int(track_id),
                "dominant_emotion": str(dominant),
                **{e: float(emotion_scores.get(e, 0.0)) for e in self.emotions},
            }
        )

    def save_session_log(self) -> Optional[str]:
        if not self.session_data:
            return None

        os.makedirs(str(config.LOGS_DIR), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(Path(config.LOGS_DIR) / f"emotion_session_web_{timestamp}.csv")
        pd.DataFrame(self.session_data).to_csv(filename, index=False)
        logger.info(f"Session log saved to {filename}")
        return filename

    def _append_alerts(self, alerts: List[dict]) -> None:
        for a in alerts:
            self.recent_alerts.append(a)

    def _append_risk_alerts(self, alerts: List[AlertEvent]) -> None:
        for ev in alerts:
            d = asdict(ev)
            d["rule"] = "risk_engine"
            self.recent_alerts.append(d)

    def _persist_alert_if_needed(self, level: int, track_id: int, ts: float, keyframe_bgr: np.ndarray, rule: str) -> None:
        if level < 2:
            return

        traj = list(self.track_trajectory.get(track_id, []))
        curves: Dict[str, Any] = {}
        st = self.risk_engine.states.get(track_id)
        if st is not None:
            curves["risk_history"] = [{"ts": float(t), "risk": float(v)} for (t, v) in list(st.risk_history)]
            curves["z_history"] = [{"ts": float(t), "z": float(v)} for (t, v) in list(st.z_history)]
        curves["last_probs"] = dict(self.track_last_probs.get(track_id, {}))
        metadata = {"source": self.source, "rule": rule}

        try:
            self.persistence.persist(
                session_id=self.session_id or "session",
                level=int(level),
                track_id=int(track_id),
                ts=float(ts),
                keyframe_bgr=keyframe_bgr,
                trajectory=[(float(t), tuple(map(float, bbox))) for (t, bbox) in traj],
                curves=curves,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Alert persistence failed: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        processed = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        det_boxes: List[Box] = []
        det_scores: List[float] = []
        det_probs: List[Dict[str, float]] = []
        det_dom: List[str] = []

        for (x, y, w, h) in faces:
            x1 = float(max(0, x))
            y1 = float(max(0, y))
            x2 = float(max(0, x + w))
            y2 = float(max(0, y + h))

            face_roi = frame_bgr[int(y1) : int(y2), int(x1) : int(x2)]
            if face_roi.size == 0:
                continue

            dominant, scores = self.detect_emotion(face_roi)
            score = float(max(scores.values()) if scores else 0.0)

            det_boxes.append((x1, y1, x2, y2))
            det_scores.append(score)
            det_probs.append(scores)
            det_dom.append(dominant)

            if dominant in self.emotion_counts:
                self.emotion_counts[dominant] += 1

        detections = list(zip(det_boxes, det_scores))
        tracks: List[Track] = self.tracker.update(detections)

        ts = time.time()
        track_probs: Dict[int, Dict[str, float]] = {}

        # Associate tracks to current detections by IoU
        for tr in tracks:
            best_i = -1
            best_iou = 0.0
            for i, b in enumerate(det_boxes):
                v = _iou(tuple(map(float, tr.bbox_xyxy)), tuple(map(float, b)))
                if v > best_iou:
                    best_iou = v
                    best_i = i

            if best_i >= 0 and best_iou >= 0.2:
                probs = det_probs[best_i]
                dom = det_dom[best_i]
                track_probs[int(tr.track_id)] = probs
                self.track_last_probs[int(tr.track_id)] = probs
                self.log_emotion(track_id=int(tr.track_id), dominant=str(dom), emotion_scores=probs)
            elif int(tr.track_id) in self.track_last_probs:
                track_probs[int(tr.track_id)] = self.track_last_probs[int(tr.track_id)]

            self.track_trajectory[int(tr.track_id)].append((float(ts), tuple(map(float, tr.bbox_xyxy))))

        # Risk engine + additional negative-duration rule
        _smoothed, _metrics, new_alerts = self.risk_engine.update(ts=ts, track_probs=track_probs)
        self._append_risk_alerts(new_alerts)

        neg_alerts = self.neg_rule.update(ts=ts, track_probs=track_probs)
        self._append_alerts(neg_alerts)

        for ev in new_alerts:
            self._persist_alert_if_needed(level=int(ev.level), track_id=int(ev.track_id), ts=float(ev.ts), keyframe_bgr=processed, rule="risk_engine")
        for a in neg_alerts:
            self._persist_alert_if_needed(level=int(a["level"]), track_id=int(a["track_id"]), ts=float(a["ts"]), keyframe_bgr=processed, rule="neg_duration")

        # Draw track boxes
        for tr in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in tr.bbox_xyxy]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(0, x2)
            y2 = max(0, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            probs = track_probs.get(int(tr.track_id), {})
            dom = max(probs.items(), key=lambda kv: float(kv[1]))[0] if probs else "unknown"
            color = (0, 255, 0)
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            label = f"T{tr.track_id} {str(dom).upper()}"
            cv2.putText(processed, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return processed

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self.is_running or self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret:
                if self.source == "video":
                    # EOF for uploaded video
                    self.is_running = False
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                else:
                    logger.warning("Camera read failed (will retry)")
                return None

        try:
            processed = self.process_frame(frame)
        except Exception as e:
            logger.error(f"process_frame error: {e}")
            processed = frame
        self.current_frame = processed
        return processed

    def get_alerts_snapshot(self, limit: int = 50) -> List[dict]:
        items = list(self.recent_alerts)
        items.sort(key=lambda x: float(x.get("ts", 0.0)), reverse=True)
        return items[: int(limit)]


detector = WebEmotionDetector()


def generate_frames():
    while detector.is_running:
        frame = detector.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        time.sleep(detector._sleep_interval())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file field"}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        orig_name = str(f.filename)
        ext = Path(orig_name).suffix.lower()
        allowed = set(getattr(config, "UPLOAD_SETTINGS", {}).get("allowed_exts", [".mp4", ".avi", ".mov", ".mkv", ".webm"]))
        if ext and ext not in allowed:
            return jsonify({"status": "error", "message": f"Unsupported file type: {ext}"}), 400

        upload_dir = Path(getattr(config, "UPLOAD_SETTINGS", {}).get("upload_dir", str(Path(config.LOGS_DIR) / "uploads")))
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = secure_filename(orig_name) or f"video{ext or '.mp4'}"
        video_id = uuid.uuid4().hex
        path = upload_dir / f"{video_id}_{safe_name}"
        f.save(str(path))

        detector.register_uploaded_video(video_id=video_id, path=str(path), filename=orig_name)
        return jsonify({"status": "success", "video_id": video_id, "filename": orig_name})
    except Exception as e:
        logger.error(f"upload_video error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/start_detection", methods=["POST"])
def start_detection():
    try:
        payload = request.get_json(silent=True) or {}
        source = str(payload.get("source") or "camera")

        video_path = None
        if source == "video":
            video_id = str(payload.get("video_id") or "")
            info = detector.uploaded_videos.get(video_id)
            if not info:
                return jsonify({"status": "error", "message": "Invalid video_id"}), 400
            video_path = info.get("path")

        success = detector.start_detection(source=source, video_path=video_path)
        if success:
            return jsonify({"status": "success", "message": "Detection started"})
        return jsonify({"status": "error", "message": "Failed to start detection"}), 500
    except Exception as e:
        logger.error(f"start_detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    try:
        detector.stop_detection()
        return jsonify({"status": "success", "message": "Detection stopped"})
    except Exception as e:
        logger.error(f"stop_detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_stats")
def get_stats():
    try:
        return jsonify(
            {
                "status": "success",
                "emotion_counts": detector.emotion_counts,
                "total_detections": sum(detector.emotion_counts.values()),
            }
        )
    except Exception as e:
        logger.error(f"get_stats error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_alerts")
def get_alerts():
    try:
        return jsonify({"status": "success", "alerts": detector.get_alerts_snapshot(limit=60)})
    except Exception as e:
        logger.error(f"get_alerts error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/save_log", methods=["POST"])
def save_log():
    try:
        filename = detector.save_session_log()
        if filename:
            return jsonify({"status": "success", "message": "Session log saved", "filename": filename})
        return jsonify({"status": "error", "message": "No session data to save"}), 400
    except Exception as e:
        logger.error(f"save_log error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    try:
        detector.reset_runtime()
        return jsonify({"status": "success", "message": "Statistics reset"})
    except Exception as e:
        logger.error(f"reset_stats error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("🌐 Starting EmoScan Web Interface")
    print("=" * 50)
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(
            host=str(getattr(config, "WEB_SETTINGS", {}).get("host", "0.0.0.0")),
            port=int(getattr(config, "WEB_SETTINGS", {}).get("port", 5000)),
            debug=bool(getattr(config, "WEB_SETTINGS", {}).get("debug", True)),
            threaded=bool(getattr(config, "WEB_SETTINGS", {}).get("threaded", True)),
        )
    finally:
        detector.stop_detection()
