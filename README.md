# AHSE: Adaptive Human State Engine

Real-time multimodal cognitive state inference + adaptive productivity coaching engine.


## Project Structure

ahse/
├── api.py            ← FastAPI app (WebSocket + REST)
├── engine.py         ← Core inference loop (in-memory state)
├── fusion.py         ← Weighted multimodal fusion + productivity
├── behavior.py       ← Keyboard/mouse behavioral tracker
├── sentiment.py      ← VADER text sentiment
├── train.py          ← SVM model trainer (HOG + CK+ dataset)
├── model.pkl         ← Trained SVM emotion classifier (shows up after training)
├── sessions/         ← Auto-saved session JSON exports
└── static/
    ├── index.html    ← Main dashboard (WebSocket-driven)
    └── analytics.html← Analytics dashboard


## Setup

```bash
pip install -r requirements.txt
```

### Train the emotion model

Requires `ckextended.csv`(The dataset used here[CK+]) in the same directory:

```bash
python train.py
# → model.pkl and a sample.png (gets created)
```

NOTE: Without `model.pkl`, the engine runs in **mock mode** (randomised emotion data).


## Run

```bash
uvicorn api:app --reload --port 8000
```

Then open: **http://localhost:8000**


### Backend

| Area | Before | After |
|---|---|---|
| State storage | `state.json` disk writes every 5s | In-memory `engine.latest` dict |
| Frontend updates | HTTP polling every 2s | **WebSocket push** (instant) |
| History | Not exposed via API | `GET /history?limit=N` endpoint |
| WS fallback | None | HTTP poll activates if WS drops |
| Fusion confidence | Raw per-cycle value | Rolling 5-cycle average (smooth) |
| Mock emotion | Pure random | Smooth random walk |

### Frontend (Dashboard)

- **WebSocket** connection with exponential back-off reconnect
- Live connection status badge (green = connected, red = error)
- Dual-dataset productivity chart (Prod % + HSI×100 dashed line)
- **Activity bar gauges** — visual bars for typing/activity/idle
- **Sentiment scale** — visual thumb on −1→+1 track
- **Trend arrow** — ↑ ↓ → based on productivity trend
- Stressed state: ambient background pulse animation
- Coaching text color adapts per state

### Analytics

- Seeds charts from `GET /history` on load (shows current session history)
- Added **sentiment-over-time** chart
- Periodic re-sync of history every 30s
- Dual-line productivity + HSI chart


## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/analytics` | Analytics page |
| WS  | `/ws` | Real-time push stream |
| GET | `/predict` | Current state (HTTP fallback) |
| GET | `/history?limit=N` | Recent inference history |
| GET | `/status` | Engine status |
| GET | `/session` | Active session summary |
| GET | `/session/history` | Past 10 sessions |
| POST | `/start` | Start engine |
| POST | `/stop` | Stop engine |
| POST | `/pause` | Pause engine |
| POST | `/resume` | Resume engine |
| POST | `/reset` | Reset to IDLE |
| POST | `/session/start_recording` | Begin recording |
| POST | `/session/stop_recording` | End + export session |
| POST | `/text` | Submit text for sentiment |


## Cognitive States

| State | Color | Description |
|---|---|---|
| Focused | Blue | Deep work, high typing, neutral/happy |
| Engaged | Green | Active, high activity, positive |
| Calm | Teal | Low activity, positive sentiment |
| Distracted | Orange | High idle, low typing |
| Stressed | Pink/Red | Angry/fear emotion, negative text |
| Fatigued | Purple | Extended idle, sad emotion |


## Fusion Weights

Final state = Vision(0.40) + Behavior(0.35) + Sentiment(0.25)
HSI         = Focus(0.40)  + Activity(0.30) + Sentiment(0.30)

Temporal smoothing: majority vote over last 5 cycles.