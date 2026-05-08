"""
engine.py — AHSE Core Engine
In-memory state, thread-safe, non-blocking, WebSocket-ready
"""
import time
import random
import threading
import json
import os
from collections import deque
from datetime import datetime
from behavior import BehaviorTracker
from fusion import fuse_state, compute_productivity
from sentiment import analyze_text

# real model + webcam
try:
    import joblib
    import numpy as np
    import cv2
    from skimage.feature import hog as _hog

    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
    MODEL = joblib.load(_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    USE_REAL_MODEL = True
    print("[Engine] SVM model loaded. Webcam active.")
except Exception as e:
    MODEL, cap, _face_cascade = None, None, None
    USE_REAL_MODEL = False
    print(f"[Engine] Mock mode: {e}")

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
INTERVAL = 5  # seconds between inference cycles


class EngineState:
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    PAUSED  = "PAUSED"
    STOPPED = "STOPPED"


class Session:
    def __init__(self):
        self.id         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.end_time   = None
        self.state_log: list[dict] = []
        self.recording  = False

    def log(self, snapshot: dict):
        self.state_log.append({**snapshot, "ts": time.time()})

    def summary(self) -> dict:
        duration = (self.end_time or time.time()) - self.start_time
        states = [e["state"] for e in self.state_log]
        dominant = max(set(states), key=states.count) if states else "Unknown"
        stress_spikes = sum(1 for s in states if s == "Stressed")
        focus_entries = [e for e in self.state_log if e["state"] in ("Focused", "Engaged")]
        focus_score   = round(len(focus_entries) / max(len(self.state_log), 1), 2)
        mins, secs = divmod(int(duration), 60)
        return {
            "session_id":      self.id,
            "session_time":    f"{mins}m {secs}s",
            "dominant_state":  dominant,
            "stress_spikes":   stress_spikes,
            "focus_score":     focus_score,
            "total_snapshots": len(self.state_log),
        }

    def export(self) -> dict:
        return {"summary": self.summary(), "timeline": self.state_log}


class AHSEEngine:
    def __init__(self):
        self._lock = threading.Lock()

        # Engine lifecycle
        self.engine_state = EngineState.IDLE
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread: threading.Thread | None = None

        # In-memory state (no disk I/O during runtime)
        self.latest: dict = {
            "state": "Unknown", "confidence": 0.0, "emotion": "Neutral",
            "behavior": {}, "sentiment": None, "productivity": {},
            "engine_state": EngineState.IDLE, "timestamp": None,
        }
        self.history: deque[dict] = deque(maxlen=200)

        # WebSocket broadcast callback
        self._broadcast_fn = None   # set by api.py via set_broadcast()

        # Sub-systems
        self._pending_text = {"value": None}
        self._tracker      = BehaviorTracker()
        self._session: Session | None = None
        self._sessions_dir = os.path.join(os.path.dirname(__file__), "sessions")
        os.makedirs(self._sessions_dir, exist_ok=True)
        self._last_emotion = 6

    # WebSocket hook
    def set_broadcast(self, fn):
        self._broadcast_fn = fn

    def _broadcast(self, data: dict):
        if self._broadcast_fn:
            try:
                self._broadcast_fn(data)
            except Exception:
                pass

    # Engine lifecycle
    def start(self) -> dict:
        if self.engine_state == EngineState.RUNNING:
            return {"status": "already running"}
        self._stop_event.clear()
        self._pause_event.set()
        self._session = Session()
        self._tracker.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.engine_state = EngineState.RUNNING
        self._update_latest({"engine_state": EngineState.RUNNING})
        print("[Engine] Started.")
        return {"status": "started", "session_id": self._session.id}

    def stop(self) -> dict:
        if self.engine_state == EngineState.STOPPED:
            return {"status": "already stopped"}
        self._stop_event.set()
        self._pause_event.set()
        if self._session:
            self._session.end_time = time.time()
            self._save_session()
        self._tracker.stop()
        self.engine_state = EngineState.STOPPED
        self._update_latest({"engine_state": EngineState.STOPPED})
        print("[Engine] Stopped.")
        return {"status": "stopped", "summary": self._session.summary() if self._session else {}}

    def pause(self) -> dict:
        self._pause_event.clear()
        self.engine_state = EngineState.PAUSED
        self._update_latest({"engine_state": EngineState.PAUSED})
        return {"status": "paused"}

    def resume(self) -> dict:
        self._pause_event.set()
        self.engine_state = EngineState.RUNNING
        self._update_latest({"engine_state": EngineState.RUNNING})
        return {"status": "resumed"}

    def reset(self) -> dict:
        self.stop()
        self.engine_state = EngineState.IDLE
        with self._lock:
            self.latest = {
                "state": "Unknown", "confidence": 0.0, "emotion": "Neutral",
                "behavior": {}, "sentiment": None, "productivity": {},
                "engine_state": EngineState.IDLE, "timestamp": None,
            }
            self.history.clear()
        print("[Engine] Reset.")
        return {"status": "reset"}

    def status(self) -> dict:
        with self._lock:
            return {**self.latest, "session_id": self._session.id if self._session else None}

    def set_text(self, text: str):
        self._pending_text["value"] = text

    # Session
    def get_session_summary(self) -> dict:
        if not self._session:
            return {"error": "no active session"}
        return self._session.summary()

    def get_history(self, limit: int = 60) -> list[dict]:
        with self._lock:
            return list(self.history)[-limit:]

    def start_recording(self) -> dict:
        if self._session:
            self._session.recording = True
        return {"recording": True}

    def stop_recording(self) -> dict:
        if not self._session:
            return {"error": "no session"}
        self._session.recording = False
        self._session.end_time = time.time()
        return self._session.export()

    def _save_session(self):
        if not self._session:
            return
        path = os.path.join(self._sessions_dir, f"session_{self._session.id}.json")
        with open(path, "w") as f:
            json.dump(self._session.export(), f, indent=2)
        print(f"[Engine] Session saved → {path}")

    # In-memory state update
    def _update_latest(self, patch: dict):
        with self._lock:
            self.latest.update(patch)

    # Emotion helpers
    def _get_emotion_real(self) -> tuple[int, float]:
        if cap is None:
            return self._last_emotion, 0.5
        ret, frame = cap.read()
        if not ret:
            return self._last_emotion, 0.5
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return self._last_emotion, 0.5
        x, y, w, h = faces[0]
        try:
            face     = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            features = _hog(face, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2)).reshape(1, -1)
            label    = int(MODEL.predict(features)[0])
            proba    = float(max(MODEL.predict_proba(features)[0]))
            self._last_emotion = label
            return label, proba
        except Exception:
            return self._last_emotion, 0.5

    def _mock_emotion(self) -> tuple[int, float]:
        emotions = [max(0, min(6, self._last_emotion + random.choice([-1, 0, 0, 1])))]
        self._last_emotion = emotions[0]
        return emotions[0], round(random.uniform(0.50, 0.92), 2)

    # Main loop
    def _loop(self):
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            emotion_label, emotion_conf = (
                self._get_emotion_real() if USE_REAL_MODEL else self._mock_emotion()
            )
            behavior  = self._tracker.get_features(window=INTERVAL)
            text      = self._pending_text["value"]
            self._pending_text["value"] = None
            sentiment = analyze_text(text)

            state, conf  = fuse_state(emotion_label, emotion_conf, behavior, sentiment)
            productivity = compute_productivity(state, behavior, sentiment)

            snapshot = {
                "state":        state,
                "confidence":   conf,
                "emotion":      EMOTION_NAMES[emotion_label],
                "behavior":     behavior,
                "sentiment":    sentiment,
                "productivity": productivity,
                "engine_state": self.engine_state,
                "timestamp":    time.time(),
            }

            with self._lock:
                self.latest = snapshot.copy()
                self.history.append({**snapshot, "ts": time.time()})

            if self._session:
                self._session.log(snapshot)

            # Push to WebSocket subscribers
            self._broadcast(snapshot)

            print(
                f"[{time.strftime('%H:%M:%S')}] {state:12s} | conf={conf:.2f} | "
                f"prod={productivity['productivity_score']} | "
                f"emotion={EMOTION_NAMES[emotion_label]} | idle={behavior['idle_time']}s"
            )

            self._stop_event.wait(timeout=INTERVAL)
