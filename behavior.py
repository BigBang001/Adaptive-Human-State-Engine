import time
import threading
from collections import deque

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


class BehaviorTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._keystrokes: list[float] = []
        self._mouse_events: list[float] = []
        self._last_event_time = time.time()
        self._listeners = []

    def start(self):
        if not PYNPUT_AVAILABLE:
            print("[BehaviorTracker] pynput not available — mock mode.")
            return
        try:
            kb = keyboard.Listener(on_press=self._on_key)
            ms = mouse.Listener(on_move=self._on_mouse, on_click=self._on_mouse)
            kb.start()
            ms.start()
            self._listeners = [kb, ms]
        except Exception as e:
            print(f"[BehaviorTracker] Could not start listeners: {e}")

    def stop(self):
        for l in self._listeners:
            try:
                l.stop()
            except Exception:
                pass

    def _on_key(self, key):
        now = time.time()
        with self._lock:
            self._keystrokes.append(now)
            self._last_event_time = now

    def _on_mouse(self, *args):
        now = time.time()
        with self._lock:
            self._mouse_events.append(now)
            self._last_event_time = now

    def get_features(self, window: float = 5.0) -> dict:
        now = time.time()
        cutoff = now - window
        with self._lock:
            recent_keys = [t for t in self._keystrokes if t > cutoff]
            recent_mouse = [t for t in self._mouse_events if t > cutoff]
            idle_time = now - self._last_event_time
            self._keystrokes = recent_keys
            self._mouse_events = recent_mouse

        typing_speed = round(len(recent_keys) / window, 2)
        activity_score = round((len(recent_keys) + len(recent_mouse)) / window, 2)
        return {
            "typing_speed": typing_speed,
            "activity_score": activity_score,
            "idle_time": round(idle_time, 2),
        }