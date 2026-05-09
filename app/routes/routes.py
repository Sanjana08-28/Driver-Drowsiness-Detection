from flask import Blueprint
from flask import render_template
from flask import Response
from flask import jsonify

from src.video_capture import VideoCamera

main = Blueprint('main', __name__)

camera = VideoCamera()


@main.route('/')
def index():

    return render_template('index.html')


# ==========================================
# VIDEO STREAM
# ==========================================
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


# ==========================================
# LIVE STATUS
# ==========================================
@main.route('/status')
def status():

    detector = camera.detector

    if detector.frame_counter > 25:

        return jsonify({
            "status": "ALERT"
        })

    elif detector.yawn_counter > 5:

        return jsonify({
            "status": "YAWNING"
        })

    elif detector.frame_counter > 10:

        return jsonify({
            "status": "DROWSY"
        })

    else:

        return jsonify({
            "status": "SAFE"
        })


# ==========================================
# ALERT HISTORY
# ==========================================
@main.route('/history')
def history():

    detector = camera.detector

    return jsonify({
        "history": detector.alert_history
    })