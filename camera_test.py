import cv2

from src.face_detection import FaceMeshDetector

class VideoCamera:

    def __init__(self):

        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.detector = FaceMeshDetector()

    def __del__(self):

        self.video.release()

    def get_frame(self):

        success, frame = self.video.read()

        if not success:
            return None

        frame = cv2.flip(frame, 1)

        frame = self.detector.detect_face_mesh(frame)

        ret, buffer = cv2.imencode('.jpg', frame)

        return buffer.tobytes()