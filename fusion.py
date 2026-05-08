"""
fusion.py — AHSE State Fusion & Productivity Engine
Weighted multimodal fusion with temporal smoothing.
"""
from collections import deque

EMOTION_MAP = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral",
}

STATES = ["Focused", "Distracted", "Stressed", "Calm", "Fatigued", "Engaged"]

_history: deque[str] = deque(maxlen=5)
_conf_history: deque[float] = deque(maxlen=5)


# FUSION
def fuse_state(
    emotion_label: int,
    emotion_conf: float,
    behavior: dict,
    sentiment: float | None = None,
) -> tuple[str, float]:
    typing_speed   = behavior.get("typing_speed", 0)
    activity_score = behavior.get("activity_score", 0)
    idle_time      = behavior.get("idle_time", 0)

    scores = {s: 0.0 for s in STATES}

    # Vision signal (weight 0.40)
    vw = 0.40 * emotion_conf
    vision_rules = {
        3: {"Focused": 0.35, "Engaged": 0.35, "Calm": 0.30},          # Happy
        0: {"Stressed": 0.70, "Distracted": 0.30},                     # Angry
        2: {"Stressed": 0.60, "Fatigued": 0.40},                       # Fear
        4: {"Fatigued": 0.50, "Stressed": 0.30, "Distracted": 0.20},  # Sad
        5: {"Distracted": 0.50, "Engaged": 0.50},                      # Surprise
        1: {"Stressed": 0.60, "Distracted": 0.40},                     # Disgust
        6: {"Focused": 0.40, "Calm": 0.40, "Engaged": 0.20},          # Neutral
    }
    for state, weight in vision_rules.get(emotion_label, vision_rules[6]).items():
        scores[state] += weight * vw

    # Behavior signal (weight 0.35)
    bw = 0.35

    # Idle time → fatigue / distraction
    if idle_time > 120:
        scores["Fatigued"]   += 0.70 * bw
        scores["Distracted"] += 0.30 * bw
    elif idle_time > 60:
        scores["Fatigued"]   += 0.55 * bw
        scores["Distracted"] += 0.45 * bw
    elif idle_time > 20:
        scores["Distracted"] += 0.55 * bw
        scores["Fatigued"]   += 0.20 * bw
    elif idle_time > 8:
        scores["Distracted"] += 0.30 * bw

    # Typing speed → focus signal
    if typing_speed > 4.0:
        scores["Focused"]  += 0.45 * bw
        scores["Engaged"]  += 0.30 * bw
        scores["Stressed"] += 0.10 * bw  # frantic typing may indicate stress
    elif typing_speed > 2.0:
        scores["Focused"]  += 0.35 * bw
        scores["Engaged"]  += 0.20 * bw
    elif typing_speed > 0.5:
        scores["Focused"]  += 0.20 * bw
    elif typing_speed > 0.1:
        scores["Focused"]  += 0.10 * bw

    # Activity score
    if activity_score > 8.0:
        scores["Engaged"]  += 0.35 * bw
        scores["Stressed"] += 0.20 * bw
    elif activity_score > 4.0:
        scores["Engaged"]  += 0.20 * bw
    elif activity_score < 0.1 and idle_time < 5:
        scores["Calm"]     += 0.45 * bw  # very still, not idle = calm/reading

    # Sentiment signal (weight 0.25)
    if sentiment is not None:
        sw = 0.25
        if sentiment > 0.6:
            scores["Calm"]    += 0.40 * sw
            scores["Focused"] += 0.30 * sw
            scores["Engaged"] += 0.30 * sw
        elif sentiment > 0.2:
            scores["Focused"] += 0.35 * sw
            scores["Calm"]    += 0.25 * sw
        elif sentiment > -0.2:
            scores["Focused"] += 0.20 * sw  # neutral text, slight focus signal
        elif sentiment > -0.5:
            scores["Stressed"]   += 0.35 * sw
            scores["Fatigued"]   += 0.20 * sw
        else:
            scores["Stressed"]   += 0.55 * sw
            scores["Distracted"] += 0.30 * sw

    raw_state = max(scores, key=scores.get)
    raw_conf  = round(min(scores[raw_state], 1.0), 3)

    # Temporal smoothing
    _history.append(raw_state)
    _conf_history.append(raw_conf)
    smoothed = max(set(_history), key=list(_history).count)
    # Averaged confidence prevents spiky display
    avg_conf = round(sum(_conf_history) / len(_conf_history), 3)

    return smoothed, avg_conf


# PRODUCTIVITY + HSI
def compute_productivity(
    state: str,
    behavior: dict,
    sentiment: float | None,
) -> dict:
    typing_speed   = behavior.get("typing_speed", 0)
    activity_score = behavior.get("activity_score", 0)
    idle_time      = behavior.get("idle_time", 0)

    # Focus score (0–1) based on cognitive state
    state_focus = {
        "Focused":    0.92,
        "Engaged":    0.85,
        "Calm":       0.68,
        "Distracted": 0.32,
        "Stressed":   0.42,
        "Fatigued":   0.22,
    }
    focus = state_focus.get(state, 0.50)

    # Activity norm (0–1) with idle penalty
    activity_norm = min(activity_score / 8.0, 1.0)
    if idle_time > 60:
        activity_norm *= 0.25
    elif idle_time > 30:
        activity_norm *= 0.55
    elif idle_time > 15:
        activity_norm *= 0.75

    # Sentiment norm (0–1)
    sent_norm = ((sentiment or 0.0) + 1) / 2

    # HSI = Human State Index
    hsi = round(focus * 0.40 + activity_norm * 0.30 + sent_norm * 0.30, 3)

    # Productivity 0–100
    productivity = round(hsi * 100)

    # Insights
    insights: list[str] = []
    if idle_time > 60:
        insights.append("Attention drift detected")
    if state == "Fatigued" and idle_time > 15:
        insights.append("Fatigue predicted — consider a break")
    if state == "Focused" and typing_speed > 2:
        insights.append("Deep work mode active")
    if state == "Distracted" and activity_score < 0.5:
        insights.append("Focus drop detected")
    if state == "Stressed" and typing_speed > 3:
        insights.append("High-stress activity — breathe")
    if state == "Engaged" and activity_score > 5:
        insights.append("Peak engagement")

    # Coaching tips based on state
    coaching_map = {
        "Focused":    "You're in deep work mode — protect this time.",
        "Engaged":    "Great engagement. Maintain this flow state.",
        "Calm":       "Good baseline. Ease into your next task.",
        "Distracted": "Close extra tabs. Focus on one task at a time.",
        "Stressed":   "Take a 5-minute break. Breathe slowly and reset.",
        "Fatigued":   "Step away for 10 minutes. Hydrate and stretch.",
    }
    coaching = coaching_map.get(state, "Stay present and focused.")

    # Trend (simple: compare focus to baseline)
    trend = "stable"
    if focus > 0.80:
        trend = "up"
    elif focus < 0.40:
        trend = "down"

    return {
        "productivity_score": productivity,
        "hsi":                hsi,
        "focus":              round(focus, 2),
        "activity_norm":      round(activity_norm, 2),
        "sentiment_norm":     round(sent_norm, 2),
        "insights":           insights,
        "coaching":           coaching,
        "trend":              trend,
    }
