#!/usr/bin/env python3
"""
EmoScan: Flask Web Interface
Web-based alternative to the Tkinter UI for emotion recognition
"""


from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime
import logging
import json
import config  # <-- Add this import
from pathlib import Path
from collections import deque

from detection.face_detector import create_face_detector
from tracking.bytetrack import BYTETracker
from alerting.risk_engine import RiskEngine
from alerting.persistence import AlertPersistence



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with correct template folder
app = Flask(__name__, template_folder='ui/templates')

class WebEmotionDetector:
    """Web-based emotion detector for Flask interface"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.session_data = []
        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.fer_model = None
        self.setup_emotion_detection()

        # Detection + tracking + alerting
        self.face_detector = create_face_detector(self.fer_model)
        self.tracker = BYTETracker(
            high_thresh=0.6,
            low_thresh=0.1,
            iou_thresh=0.3,
            max_time_lost=30,
            min_hits=3,
        )
        self.risk_engine = RiskEngine(settings=config.ALERT_SETTINGS, emotions=self.emotions)

        # Evidence persistence
        self.session_id = None
        self.alert_store = AlertPersistence(root_dir=Path(config.LOGS_DIR) / "alerts")
        self.persisted_alerts = deque(maxlen=200)

        # Per-track history for evidence
        self.track_bbox_history = {}  # track_id -> deque[(ts, bbox_xyxy)]
        self.track_last_probs = {}  # track_id -> probs dict
        self.track_last_metrics = {}  # track_id -> metrics dict
        self.track_last_dominant = {}  # track_id -> dominant emotion
        
    def setup_emotion_detection(self):
        """Initialize emotion detection model"""
        try:
            from fer import FER
            # Try without mtcnn first, as it can cause issues on some systems
            try:
                self.fer_model = FER(mtcnn=False)
                logger.info("FER emotion detection model loaded successfully (without MTCNN)")
            except Exception as mtcnn_error:
                logger.warning(f"MTCNN failed, trying without: {mtcnn_error}")
                self.fer_model = FER()
                logger.info("FER emotion detection model loaded successfully (default)")
        except ImportError:
            logger.warning("FER not available, using simplified emotion detection")
            self.fer_model = None
        except Exception as e:
            logger.error(f"Error loading FER model: {e}")
            self.fer_model = None
    
    def detect_emotion(self, face_img):
        """Detect emotion in a face image using FER or simplified detection"""
        try:
            if self.fer_model:
                # Use FER for emotion detection
                logger.debug(f"Attempting FER emotion detection on image of shape: {face_img.shape}")
                result = self.fer_model.detect_emotions(face_img)
                logger.debug(f"FER result: {result}")
                
                if result and len(result) > 0:
                    dominant_emotion = result[0]['emotions']
                    # Convert FER format to our format
                    emotion_scores = {}
                    for emotion in self.emotions:
                        emotion_scores[emotion] = dominant_emotion.get(emotion, 0.0)
                    
                    # Find dominant emotion
                    dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    logger.debug(f"Detected emotion: {dominant} with scores: {emotion_scores}")
                    return dominant, emotion_scores
                else:
                    logger.debug("FER returned no results, using neutral")
                    return 'neutral', {emotion: 0.0 for emotion in self.emotions}
            else:
                # Simplified emotion detection (fallback)
                logger.debug("Using simplified emotion detection")
                return self.detect_emotion_simple(face_img)
                
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            return 'neutral', {emotion: 0.0 for emotion in self.emotions}
    
    def detect_emotion_simple(self, face_img):
        """Simplified emotion detection for fallback"""
        import random
        
        # Simple heuristic-based emotion detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Simple brightness-based emotion detection
        if brightness > 150:
            dominant_emotion = 'happy'
        elif brightness < 100:
            dominant_emotion = 'sad'
        else:
            dominant_emotion = 'neutral'
        
        # Generate emotion scores
        emotion_scores = {}
        for emotion in self.emotions:
            if emotion == dominant_emotion:
                emotion_scores[emotion] = random.uniform(0.6, 0.9)
            else:
                emotion_scores[emotion] = random.uniform(0.0, 0.3)
        
        return dominant_emotion, emotion_scores
    
    def process_frame(self, frame):
        """Process a single frame for face detection, tracking, and tiered alerting"""
        ts = time.time()
        processed_frame = frame.copy()

        # Detect faces with confidence scores
        dets = self.face_detector.detect(frame)
        detections = [((float(x1), float(y1), float(x2), float(y2)), float(s)) for (x1, y1, x2, y2), s in [(d.bbox_xyxy, d.score) for d in dets]]

        # Update tracker
        tracks = self.tracker.update(detections)

        # Per-track probabilities for risk engine
        track_probs = {}

        h, w = frame.shape[:2]
        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox_xyxy
            x1i = max(0, min(w - 1, int(round(x1))))
            y1i = max(0, min(h - 1, int(round(y1))))
            x2i = max(0, min(w - 1, int(round(x2))))
            y2i = max(0, min(h - 1, int(round(y2))))
            if x2i <= x1i or y2i <= y1i:
                continue

            face_roi = frame[y1i:y2i, x1i:x2i]
            dominant_emotion, emotion_scores = self.detect_emotion(face_roi)
            track_probs[int(trk.track_id)] = emotion_scores
            self.track_last_probs[int(trk.track_id)] = emotion_scores
            self.track_last_dominant[int(trk.track_id)] = dominant_emotion

            # Update counters (counts per active track per frame)
            self.emotion_counts[dominant_emotion] += 1

            # Track history for evidence
            hist = self.track_bbox_history.get(int(trk.track_id))
            if hist is None:
                hist = deque(maxlen=300)
                self.track_bbox_history[int(trk.track_id)] = hist
            hist.append((ts, (float(x1), float(y1), float(x2), float(y2))))

        # Risk engine update + alerts
        smoothed, metrics, new_alerts = self.risk_engine.update(ts=ts, track_probs=track_probs)
        self.track_last_metrics = metrics

        # Persist evidence on new alerts
        for ev in new_alerts:
            tid = ev.track_id
            traj = list(self.track_bbox_history.get(tid, []))
            curves = {
                "ema_probs": smoothed.get(tid, {}),
                "risk_history": list(self.risk_engine.states[tid].risk_history) if tid in self.risk_engine.states else [],
                "z_history": list(self.risk_engine.states[tid].z_history) if tid in self.risk_engine.states else [],
            }
            meta = {
                "camera_id": int(config.VIDEO_SETTINGS.get("camera_index", 0)),
                "message": ev.message,
                "metrics": {
                    "risk": ev.risk,
                    "z": ev.z,
                    "duration_sec": ev.duration_sec,
                    "slope_z_per_sec": ev.slope_z_per_sec,
                    "sync_ratio": ev.sync_ratio,
                    "crowd_median": ev.crowd_median,
                    "crowd_mad": ev.crowd_mad,
                },
                "settings": config.ALERT_SETTINGS,
            }
            ref = self.alert_store.persist(
                session_id=self.session_id or "session",
                level=ev.level,
                track_id=tid,
                ts=ev.ts,
                keyframe_bgr=processed_frame,
                trajectory=traj,
                curves=curves,
                metadata=meta,
            )
            self.persisted_alerts.appendleft({"event": ev, "ref": ref})

        # Draw overlays
        for trk in tracks:
            tid = int(trk.track_id)
            x1, y1, x2, y2 = trk.bbox_xyxy
            x1i = int(round(x1))
            y1i = int(round(y1))
            x2i = int(round(x2))
            y2i = int(round(y2))

            m = metrics.get(tid, {})
            level = int(m.get("level", 0))
            dominant = self.track_last_dominant.get(tid, "neutral")

            color = (200, 200, 200)
            if level == 1:
                color = (0, 255, 255)  # yellow (BGR)
            elif level == 2:
                color = (0, 165, 255)  # orange
            elif level == 3:
                color = (0, 0, 255)  # red

            cv2.rectangle(processed_frame, (x1i, y1i), (x2i, y2i), color, 2)
            z = float(m.get("z", 0.0))
            label = f"ID{tid} {dominant.upper()} L{level} z={z:.2f}"
            cv2.putText(
                processed_frame,
                label,
                (x1i, max(0, y1i - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Log emotion data for session (aggregated per track this frame)
        for tid, probs in track_probs.items():
            self.log_emotion(self.track_last_dominant.get(tid, "neutral"), probs, track_id=tid, metrics=metrics.get(tid))

        return processed_frame, list(self.track_last_dominant.values())
    
    def log_emotion(self, dominant_emotion, emotion_scores, track_id=None, metrics=None):
        """Log emotion data to session storage"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            'timestamp': timestamp,
            'track_id': track_id,
            'dominant_emotion': dominant_emotion,
            **emotion_scores
        }
        if metrics:
            log_entry.update(
                {
                    "risk": float(metrics.get("risk", 0.0)),
                    "z": float(metrics.get("z", 0.0)),
                    "level": int(metrics.get("level", 0)),
                }
            )
        self.session_data.append(log_entry)
    
    def save_session_log(self):
        """Save session data to CSV file"""
        if not self.session_data:
            return None
        
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/emotion_session_web_{timestamp}.csv"
        
        df = pd.DataFrame(self.session_data)
        df.to_csv(filename, index=False)
        logger.info(f"Session log saved to {filename}")
        return filename
    
    def start_detection(self):
        """Start the emotion detection process"""
        try:
            self.cap = cv2.VideoCapture(int(config.VIDEO_SETTINGS['camera_index']))  # Use config camera index
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            self.is_running = True
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Web emotion detection started")
            return True
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            return False
    
    def stop_detection(self):
        """Stop the emotion detection process"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Web emotion detection stopped")
    
    def get_frame(self):
        """Get current frame for web streaming"""
        if not self.is_running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        processed_frame, emotions = self.process_frame(frame)
        self.current_frame = processed_frame
        return processed_frame

# Global detector instance
detector = WebEmotionDetector()

def generate_frames():
    """Generate video frames for web streaming"""
    while detector.is_running:
        try:
            frame = detector.get_frame()
            if frame is not None:
                # Encode frame for web streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            # Prevent a single bad frame from killing the whole MJPEG stream
            logger.error(f"Stream frame generation error: {e}", exc_info=True)
            time.sleep(0.2)
        time.sleep(0.03)

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start emotion detection"""
    try:
        success = detector.start_detection()
        if success:
            return jsonify({'status': 'success', 'message': 'Detection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start detection'})
    except Exception as e:
        logger.error(f"Error in start_detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop emotion detection"""
    try:
        detector.stop_detection()
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        logger.error(f"Error in stop_detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_stats')
def get_stats():
    """Get current emotion statistics"""
    try:
        total = sum(detector.emotion_counts.values())
        percentages = {}
        for emo, cnt in detector.emotion_counts.items():
            percentages[emo] = 0.0 if total <= 0 else float(cnt) / float(total)
        return jsonify({
            'status': 'success',
            'emotion_counts': detector.emotion_counts,
            'total_detections': total,
            'emotion_percentages': percentages
        })
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_alerts')
def get_alerts():
    """Get recent tiered alerts (with evidence paths)"""
    try:
        out = []
        for item in list(detector.persisted_alerts)[:50]:
            ev = item["event"]
            ref = item["ref"]
            out.append(
                {
                    "ts": ev.ts,
                    "level": ev.level,
                    "track_id": ev.track_id,
                    "risk": ev.risk,
                    "z": ev.z,
                    "duration_sec": ev.duration_sec,
                    "message": ev.message,
                    "keyframe_jpg": ref.keyframe_jpg,
                    "metadata_json": ref.metadata_json,
                }
            )
        return jsonify({"status": "success", "alerts": out})
    except Exception as e:
        logger.error(f"Error in get_alerts: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_risk_snapshot')
def get_risk_snapshot():
    """Get current per-track risk/z/level snapshot for UI/debug"""
    try:
        tracks = []
        for tid, m in detector.track_last_metrics.items():
            tracks.append(
                {
                    "track_id": int(tid),
                    "dominant_emotion": detector.track_last_dominant.get(int(tid), "neutral"),
                    "risk": float(m.get("risk", 0.0)),
                    "z": float(m.get("z", 0.0)),
                    "level": int(m.get("level", 0)),
                    "duration_sec": float(m.get("duration_sec", 0.0)),
                    "slope_z_per_sec": float(m.get("slope_z_per_sec", 0.0)),
                    "sync_ratio": float(m.get("sync_ratio", 0.0)),
                }
            )
        return jsonify({"status": "success", "tracks": tracks})
    except Exception as e:
        logger.error(f"Error in get_risk_snapshot: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/alert_keyframe')
def alert_keyframe():
    """Serve saved alert keyframe by path (local only)."""
    try:
        path = request.args.get("path", "")
        if not path:
            return jsonify({"status": "error", "message": "missing path"}), 400
        # Basic safety: only allow serving within logs/alerts
        base = Path(config.LOGS_DIR) / "alerts"
        p = Path(path)
        if not p.exists():
            return jsonify({"status": "error", "message": "file not found"}), 404
        if base not in p.resolve().parents and p.resolve() != base.resolve():
            return jsonify({"status": "error", "message": "forbidden"}), 403
        return send_file(str(p), mimetype="image/jpeg")
    except Exception as e:
        logger.error(f"Error in alert_keyframe: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_log', methods=['POST'])
def save_log():
    """Save session log"""
    try:
        filename = detector.save_session_log()
        if filename:
            return jsonify({
                'status': 'success', 
                'message': 'Session log saved',
                'filename': filename
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'No session data to save'
            })
    except Exception as e:
        logger.error(f"Error in save_log: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset emotion statistics"""
    try:
        detector.emotion_counts = {emotion: 0 for emotion in detector.emotions}
        detector.session_data = []
        return jsonify({'status': 'success', 'message': 'Statistics reset'})
    except Exception as e:
        logger.error(f"Error in reset_stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🌐 Starting EmoScan Web Interface")
    print("=" * 50)
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Error: {e}")
    finally:
        detector.stop_detection()
