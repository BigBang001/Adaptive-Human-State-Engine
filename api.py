"""
api.py — AHSE FastAPI Backend
WebSocket real-time push, in-memory state, clean routing.

Run: uvicorn api:app --reload --port 8000
"""
import json
import asyncio
import threading
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import AHSEEngine

app = FastAPI(title="AHSE", version="3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

engine = AHSEEngine()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            self._clients = [c for c in self._clients if c is not ws]

    # Called from engine thread → schedule coroutine on event loop
    def broadcast_from_thread(self, data: dict):
        if not self._loop or not self._clients:
            return
        payload = json.dumps(data, default=str)
        asyncio.run_coroutine_threadsafe(self._send_all(payload), self._loop)

    async def _send_all(self, payload: str):
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
engine.set_broadcast(manager.broadcast_from_thread)


@app.on_event("startup")
async def startup():
    manager.set_loop(asyncio.get_event_loop())


# Pages
@app.get("/", response_class=HTMLResponse)
def dashboard():
    f = STATIC_DIR / "index.html"
    return HTMLResponse(f.read_text(encoding="utf-8") if f.exists() else "<h1>index.html missing</h1>")


@app.get("/analytics", response_class=HTMLResponse)
def analytics():
    f = STATIC_DIR / "analytics.html"
    return HTMLResponse(f.read_text(encoding="utf-8") if f.exists() else "<h1>analytics.html missing</h1>")


# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send current state immediately on connect
    try:
        await ws.send_text(json.dumps(engine.latest, default=str))
        while True:
            await ws.receive_text()  # keep-alive ping/pong
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# Engine control
@app.post("/start")
def start():  return engine.start()

@app.post("/stop")
def stop():   return engine.stop()

@app.post("/pause")
def pause():  return engine.pause()

@app.post("/resume")
def resume(): return engine.resume()

@app.post("/reset")
def reset():  return engine.reset()

@app.get("/status")
def status(): return engine.status()


# Data
@app.get("/predict")
def predict():
    return engine.latest


@app.get("/history")
def history(limit: int = 60):
    """Return recent inference history for analytics charts."""
    return {"history": engine.get_history(limit)}


@app.get("/session")
def session_summary():
    return engine.get_session_summary()


@app.get("/session/history")
def session_history():
    sessions_dir = Path(__file__).parent / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    files = sorted(sessions_dir.glob("*.json"), reverse=True)[:10]
    history = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            history.append(data.get("summary", {}))
        except Exception:
            pass
    return {"sessions": history}


@app.post("/session/start_recording")
def start_rec():  return engine.start_recording()

@app.post("/session/stop_recording")
def stop_rec():   return engine.stop_recording()


class TextInput(BaseModel):
    text: str

@app.post("/text")
def submit_text(body: TextInput):
    engine.set_text(body.text)
    return {"accepted": True}


# Dev runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
