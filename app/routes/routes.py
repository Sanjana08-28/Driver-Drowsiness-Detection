import time
import sqlite3
import datetime
import os

from flask import Blueprint, Response, jsonify, render_template
from src.video_capture import VideoCamera

main = Blueprint('main', __name__)

camera = VideoCamera()

# ── SQLite setup ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'sessions.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT,
            ear     REAL,
            mar     REAL,
            blinks  INTEGER,
            status  TEXT,
            session INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            ended_at   TEXT,
            duration_s INTEGER,
            alerts     INTEGER DEFAULT 0,
            risk       TEXT DEFAULT 'Low'
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Session state ─────────────────────────────────────────────────────────────
_session = {
    "id": None,
    "start": None,
    "alerts": 0,
}

_last_persist = 0


def _persist_event(ear, mar, blinks, status):
    global _last_persist
    now = time.time()
    if now - _last_persist < 2:
        return
    _last_persist = now
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO events (ts, ear, mar, blinks, status, session) VALUES (?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), ear, mar, blinks, status, _session["id"] or 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] persist error: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@main.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


@main.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@main.route('/api/status')
def api_status():
    detector = camera.detector

    ear = getattr(detector, '_last_ear', 0.30)
    mar = getattr(detector, '_last_mar', 0.05)

    frame_counter = detector.frame_counter
    yawn_counter  = detector.yawn_counter
    blink_counter = detector.blink_counter
    fatigue_score = detector.fatigue_score

    if frame_counter > 25:
        status   = "Very Drowsy"
        is_alert = True
    elif frame_counter > 10 or yawn_counter > 5:
        status   = "Drowsy"
        is_alert = False
    else:
        status   = "Alert"
        is_alert = False

    fatigue_conf = round(min((frame_counter / 25) * 100, 100), 1)

    # blink rate per minute
    elapsed_min = (time.time() - detector._session_start) / 60
    blink_rate  = round(blink_counter / max(elapsed_min, 0.01), 1)

    if is_alert and _session["id"]:
        _session["alerts"] += 1

    _persist_event(ear, mar, blink_counter, status)

    return jsonify({
        "ear":          round(ear, 3),
        "mar":          round(mar, 3),
        "blink_count":  blink_counter,
        "blink_rate":   blink_rate,
        "closure_dur":  round(frame_counter / 30, 2),
        "fatigue_conf": fatigue_conf,
        "yawn_count":   yawn_counter,
        "status":       status,
        "alert":        is_alert,
    })


@main.route('/api/history')
def api_history():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT ts, ear, mar, blinks, status FROM events ORDER BY ts DESC LIMIT 50"
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route('/api/session/start', methods=['POST'])
def session_start():
    conn = get_db()
    cur  = conn.execute(
        "INSERT INTO sessions (started_at, alerts, risk) VALUES (?,0,'Low')",
        (datetime.datetime.now().isoformat(),)
    )
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    _session["id"]     = sid
    _session["start"]  = time.time()
    _session["alerts"] = 0
    return jsonify({"session_id": sid})


@main.route('/api/session/end', methods=['POST'])
def session_end():
    sid   = _session.get("id")
    start = _session.get("start")
    if not sid:
        return jsonify({"error": "no active session"}), 400
    alerts   = _session.get("alerts", 0)
    duration = int(time.time() - start) if start else 0
    risk     = "High" if alerts >= 3 else ("Medium" if alerts >= 1 else "Low")
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET ended_at=?, duration_s=?, alerts=?, risk=? WHERE id=?",
        (datetime.datetime.now().isoformat(), duration, alerts, risk, sid)
    )
    conn.commit()
    conn.close()
    _session["id"] = None
    return jsonify({"session_id": sid, "duration_s": duration, "alerts": alerts, "risk": risk})


@main.route('/api/sessions')
def api_sessions():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# legacy endpoints kept for compatibility
@main.route('/status')
def status():
    return api_status()


@main.route('/history')
def history():
    detector = camera.detector
    return jsonify({"history": detector.alert_history})
