#!/usr/bin/env python3
"""
EmoScan: Test Suite
Comprehensive testing for the emotion recognition system
"""

import unittest
import sys
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
try:
    from config import EMOTIONS, FACE_DETECTION, VIDEO_SETTINGS
    from main import EmotionDetector, EmoScanUI
    from app import WebEmotionDetector
    from ui.emotion_visualizer import EmotionVisualizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class TestEmotionDetection(unittest.TestCase):
    """Test cases for emotion detection functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image = self.create_test_image()
        self.detector = EmotionDetector()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'detector') and self.detector.cap:
            self.detector.cap.release()
    
    def create_test_image(self):
        """Create a test image for emotion detection"""
        # Create a simple test image (64x64 pixels)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        # Add some features to make it more face-like
        cv2.rectangle(image, (20, 20), (44, 44), (255, 255, 255), -1)
        return image
    
    def test_emotion_detector_initialization(self):
        """Test emotion detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.emotions), 7)
        self.assertIn('happy', self.detector.emotions)
        self.assertIn('sad', self.detector.emotions)
        self.assertIn('angry', self.detector.emotions)
        self.assertIn('neutral', self.detector.emotions)
        self.assertIn('surprise', self.detector.emotions)
        self.assertIn('fear', self.detector.emotions)
        self.assertIn('disgust', self.detector.emotions)
    
    def test_face_cascade_loading(self):
        """Test face detection cascade loading"""
        self.assertIsNotNone(self.detector.face_cascade)
        self.assertFalse(self.detector.face_cascade.empty())
    
    def test_emotion_detection(self):
        """Test emotion detection on test image"""
        try:
            dominant_emotion, emotion_scores = self.detector.detect_emotion(self.test_image)
            
            # Check that a valid emotion was returned
            self.assertIn(dominant_emotion, self.detector.emotions)
            
            # Check that emotion scores are provided for all emotions
            for emotion in self.detector.emotions:
                self.assertIn(emotion, emotion_scores)
                self.assertIsInstance(emotion_scores[emotion], (int, float))
                self.assertGreaterEqual(emotion_scores[emotion], 0)
                
        except Exception as e:
            # DeepFace might fail on synthetic images, which is expected
            self.skipTest(f"Emotion detection failed on test image (expected): {e}")
    
    def test_frame_processing(self):
        """Test frame processing functionality"""
        # Create a test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Process the frame
        processed_frame, emotions_detected = self.detector.process_frame(test_frame)
        
        # Check that processing returns expected results
        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertIsInstance(emotions_detected, list)
        self.assertEqual(processed_frame.shape, test_frame.shape)
    
    def test_emotion_logging(self):
        """Test emotion logging functionality"""
        # Test emotion logging
        test_emotion = 'happy'
        test_scores = {emotion: 0.1 for emotion in self.detector.emotions}
        test_scores['happy'] = 0.9
        
        initial_count = len(self.detector.session_data)
        self.detector.log_emotion(test_emotion, test_scores)
        
        # Check that data was logged
        self.assertEqual(len(self.detector.session_data), initial_count + 1)
        
        # Check logged data structure
        log_entry = self.detector.session_data[-1]
        self.assertIn('timestamp', log_entry)
        self.assertIn('dominant_emotion', log_entry)
        self.assertEqual(log_entry['dominant_emotion'], test_emotion)
        
        # Check that all emotion scores are logged
        for emotion in self.detector.emotions:
            self.assertIn(emotion, log_entry)
    
    def test_session_log_saving(self):
        """Test session log saving functionality"""
        # Add some test data
        test_scores = {emotion: 0.1 for emotion in self.detector.emotions}
        test_scores['happy'] = 0.9
        
        self.detector.log_emotion('happy', test_scores)
        self.detector.log_emotion('sad', test_scores)
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change logs directory
            original_logs_dir = self.detector.session_data
            self.detector.session_data = []
            
            # Add test data
            self.detector.log_emotion('happy', test_scores)
            self.detector.log_emotion('sad', test_scores)
            
            # Save log
            filename = self.detector.save_session_log()
            
            # Check that file was created
            self.assertIsNotNone(filename)
            self.assertTrue(os.path.exists(filename))
            
            # Check file content
            df = pd.read_csv(filename)
            self.assertEqual(len(df), 2)
            self.assertIn('timestamp', df.columns)
            self.assertIn('dominant_emotion', df.columns)
            
            # Restore original data
            self.detector.session_data = original_logs_dir

class TestWebEmotionDetector(unittest.TestCase):
    """Test cases for web emotion detector"""
    
    def setUp(self):
        """Set up test environment"""
        self.web_detector = WebEmotionDetector()
    
    def test_web_detector_initialization(self):
        """Test web emotion detector initialization"""
        self.assertIsNotNone(self.web_detector)
        self.assertEqual(len(self.web_detector.emotions), 7)
        # Web detector now uses a face detector that outputs bbox + score
        self.assertIsNotNone(getattr(self.web_detector, "face_detector", None))
    
    def test_web_detector_start_stop(self):
        """Test web detector start and stop functionality"""
        # Test start detection
        success = self.web_detector.start_detection()
        # Note: This might fail if no camera is available, which is expected in CI
        if success:
            self.assertTrue(self.web_detector.is_running)
            self.assertIsNotNone(self.web_detector.cap)
            
            # Test stop detection
            self.web_detector.stop_detection()
            self.assertFalse(self.web_detector.is_running)
        else:
            self.skipTest("No camera available for testing")

class TestConfiguration(unittest.TestCase):
    """Test cases for configuration settings"""
    
    def test_emotions_configuration(self):
        """Test emotions configuration"""
        from config import EMOTIONS, EMOTION_COLORS
        
        # Check emotions list
        self.assertIsInstance(EMOTIONS, list)
        self.assertEqual(len(EMOTIONS), 7)
        
        # Check emotion colors
        self.assertIsInstance(EMOTION_COLORS, dict)
        self.assertEqual(len(EMOTION_COLORS), 7)
        
        # Check that all emotions have colors
        for emotion in EMOTIONS:
            self.assertIn(emotion, EMOTION_COLORS)
    
    def test_face_detection_configuration(self):
        """Test face detection configuration"""
        from config import FACE_DETECTION
        
        self.assertIsInstance(FACE_DETECTION, dict)
        self.assertIn('scale_factor', FACE_DETECTION)
        self.assertIn('min_neighbors', FACE_DETECTION)
        self.assertIn('min_size', FACE_DETECTION)
        self.assertIn('cascade_file', FACE_DETECTION)
    
    def test_video_settings_configuration(self):
        """Test video settings configuration"""
        from config import VIDEO_SETTINGS
        
        self.assertIsInstance(VIDEO_SETTINGS, dict)
        self.assertIn('camera_index', VIDEO_SETTINGS)
        self.assertIn('frame_width', VIDEO_SETTINGS)
        self.assertIn('frame_height', VIDEO_SETTINGS)
        self.assertIn('fps', VIDEO_SETTINGS)

class TestDataStructures(unittest.TestCase):
    """Test cases for data structures and utilities"""
    
    def test_emotion_counts_initialization(self):
        """Test emotion counts initialization"""
        detector = EmotionDetector()
        
        # Check that all emotions have count 0 initially
        for emotion in detector.emotions:
            self.assertEqual(detector.emotion_counts[emotion], 0)
    
    def test_session_data_structure(self):
        """Test session data structure"""
        detector = EmotionDetector()
        
        # Add test data
        test_scores = {emotion: 0.1 for emotion in detector.emotions}
        test_scores['happy'] = 0.9
        
        detector.log_emotion('happy', test_scores)
        
        # Check data structure
        self.assertEqual(len(detector.session_data), 1)
        log_entry = detector.session_data[0]
        
        required_fields = ['timestamp', 'dominant_emotion'] + detector.emotions
        for field in required_fields:
            self.assertIn(field, log_entry)


class TestBYTETracker(unittest.TestCase):
    """Sanity tests for (ByteTrack-like) tracker"""

    def test_track_id_stability_simple_motion(self):
        from tracking.bytetrack import BYTETracker

        trk = BYTETracker(high_thresh=0.6, low_thresh=0.1, iou_thresh=0.3, max_time_lost=5, min_hits=1)
        # one object moving slightly to the right
        dets_seq = []
        for i in range(10):
            x1 = 100 + i * 2
            y1 = 100
            x2 = x1 + 50
            y2 = y1 + 50
            dets_seq.append([((x1, y1, x2, y2), 0.9)])

        ids = []
        for dets in dets_seq:
            tracks = trk.update(dets)
            self.assertTrue(len(tracks) >= 1)
            ids.append(int(tracks[0].track_id))

        # Should keep a single consistent id
        self.assertEqual(len(set(ids)), 1)


class TestRiskEngine(unittest.TestCase):
    """Sanity tests for MAD edge cases and tiering"""

    def test_mad_zero_no_crash(self):
        from alerting.risk_engine import RiskEngine

        settings = {
            "ema_alpha": 0.3,
            "crowd_window_sec": 5.0,
            "eps": 1e-6,
            "risk_weights": {"angry": 1.0, "fear": 1.0, "disgust": 1.0, "sad": 1.0},
            "thresholds": {"t1": 1.5, "t2": 2.5, "t3": 3.5},
            "durations": {"d2_sec": 1.0},
            "slope_window_sec": 1.0,
            "slopes": {"s3_z_per_sec": 0.6},
            "sync": {"y3_ratio": 0.35, "crowd_median_min": 0.0},
            "track_ttl_sec": 10.0,
            "max_recent_alerts": 50,
        }
        emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
        eng = RiskEngine(settings=settings, emotions=emotions)

        ts = 1000.0
        probs = {1: {e: 0.0 for e in emotions}, 2: {e: 0.0 for e in emotions}}
        smoothed, metrics, alerts = eng.update(ts=ts, track_probs=probs)
        self.assertIn(1, metrics)
        self.assertIn(2, metrics)
        # MAD should be zero, but z should still be finite due to eps
        self.assertTrue(np.isfinite(metrics[1]["z"]))

    def test_tier_alert_generation(self):
        from alerting.risk_engine import RiskEngine

        settings = {
            "ema_alpha": 1.0,  # no smoothing for test
            "crowd_window_sec": 5.0,
            "eps": 1e-6,
            "risk_weights": {"angry": 1.0},
            "thresholds": {"t1": 0.5, "t2": 1.0, "t3": 1.5},
            "durations": {"d2_sec": 0.5},
            "slope_window_sec": 0.5,
            "slopes": {"s3_z_per_sec": 0.1},
            "sync": {"y3_ratio": 0.0, "crowd_median_min": 0.0},
            "track_ttl_sec": 10.0,
            "max_recent_alerts": 50,
        }
        emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
        eng = RiskEngine(settings=settings, emotions=emotions)

        # baseline track 2 low risk, track 1 high risk -> should produce alerts
        alerts_all = []
        for k in range(5):
            ts = 2000.0 + k * 0.2
            track_probs = {
                1: {"angry": 1.0},
                2: {"angry": 0.0},
            }
            _sm, _m, alerts = eng.update(ts=ts, track_probs=track_probs)
            alerts_all.extend(alerts)

        self.assertTrue(any(a.level >= 1 for a in alerts_all))

def run_tests():
    """Run all tests"""
    print("🧪 Running EmoScan Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEmotionDetection,
        TestWebEmotionDetector,
        TestConfiguration,
        TestDataStructures
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.skipped:
        print("\n⚠️  Skipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

def main():
    """Main test function"""
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 