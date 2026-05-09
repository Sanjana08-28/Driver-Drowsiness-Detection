from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",   # accessible on local network (e.g. from phone on same Wi-Fi)
        port=5000,
        debug=False,       # keep False — debug mode restarts the process and breaks the webcam thread
        threaded=True      # required for MJPEG stream + API polling to run simultaneously
    )