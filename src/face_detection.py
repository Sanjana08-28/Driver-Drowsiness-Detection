<<<<<<< HEAD
=======

>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493
import cv2
import csv
import os
import time
<<<<<<< HEAD
import numpy as np
import mediapipe as mp

from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.feature_extraction import calculate_ear, calculate_mar
=======
import mediapipe as mp
import numpy as np

from datetime import datetime

from src.feature_extraction import calculate_ear
from src.feature_extraction import calculate_mar
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493
from src.alert_system import AlertSystem


class FaceMeshDetector:

    def __init__(self):

        # ==========================================
        # CSV STORAGE
        # ==========================================
        self.csv_file = 'saved_features/fatigue_features.csv'
<<<<<<< HEAD
        os.makedirs('saved_features', exist_ok=True)

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['EAR', 'MAR', 'Frames', 'Fatigue', 'Label'])

        # ==========================================
        # MEDIAPIPE TASKS API
        # ==========================================
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'face_landmarker.task'
        )
        if not os.path.exists(model_path):
            model_path = 'face_landmarker.task'

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
=======

        os.makedirs(
            'saved_features',
            exist_ok=True
        )

        if not os.path.exists(self.csv_file):

            with open(
                self.csv_file,
                mode='w',
                newline=''
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    'EAR',
                    'MAR',
                    'Frames',
                    'Fatigue',
                    'Label'
                ])

        # ==========================================
        # MEDIAPIPE
        # ==========================================
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

        # ==========================================
        # ALERT SYSTEM
        # ==========================================
        self.alert_system = AlertSystem()

        # ==========================================
        # SCREENSHOT STORAGE
        # ==========================================
        self.screenshot_folder = "outputs/screenshots"
<<<<<<< HEAD
        os.makedirs(self.screenshot_folder, exist_ok=True)
=======

        os.makedirs(
            self.screenshot_folder,
            exist_ok=True
        )

>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493
        self.last_capture_time = 0

        # ==========================================
        # ALERT HISTORY
        # ==========================================
        self.alert_history = []

        # ==========================================
        # THRESHOLDS
        # ==========================================
        self.EAR_THRESHOLD = 0.18
        self.MAR_THRESHOLD = 0.75

        # ==========================================
        # COUNTERS
        # ==========================================
        self.frame_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.fatigue_score = 0
<<<<<<< HEAD
        self._last_ear = 0.30
        self._last_mar = 0.05
        self._eye_was_closed = False
        self._session_start = time.time()
=======
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

        # ==========================================
        # EYE LANDMARKS
        # ==========================================
<<<<<<< HEAD
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
=======
        self.LEFT_EYE = [
            33, 160, 158,
            133, 153, 144
        ]

        self.RIGHT_EYE = [
            362, 385, 387,
            263, 373, 380
        ]
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

        # ==========================================
        # MOUTH LANDMARKS
        # ==========================================
<<<<<<< HEAD
        self.MOUTH = [13, 14, 78, 308]
=======
        self.MOUTH = [
            13, 14, 78, 308
        ]

>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

    # ==========================================
    # MAIN DETECTION
    # ==========================================
    def detect_face_mesh(self, frame):

        h, w, _ = frame.shape

<<<<<<< HEAD
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.face_landmarker.detect(mp_image)

        final_alert = False

        if result.face_landmarks:

            for face in result.face_landmarks:

                landmarks = face

                left_ear = calculate_ear(landmarks, self.LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, self.RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                mar = calculate_mar(landmarks, self.MOUTH, w, h)

                self._last_ear = ear
                self._last_mar = mar
=======
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb_frame)

        final_alert = False

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # ==========================================
                # EAR
                # ==========================================
                left_ear = calculate_ear(
                    landmarks,
                    self.LEFT_EYE,
                    w,
                    h
                )

                right_ear = calculate_ear(
                    landmarks,
                    self.RIGHT_EYE,
                    w,
                    h
                )

                ear = (
                    left_ear + right_ear
                ) / 2.0

                # ==========================================
                # MAR
                # ==========================================
                mar = calculate_mar(
                    landmarks,
                    self.MOUTH,
                    w,
                    h
                )
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

                # ==========================================
                # DROWSINESS
                # ==========================================
                if ear < self.EAR_THRESHOLD:
<<<<<<< HEAD
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0

                # ==========================================
                # BLINKS (count transitions open -> closed)
                # ==========================================
                if ear < self.EAR_THRESHOLD and not self._eye_was_closed:
                    self.blink_counter += 1
                    self._eye_was_closed = True
                elif ear >= self.EAR_THRESHOLD:
                    self._eye_was_closed = False
=======

                    self.frame_counter += 1

                else:

                    self.frame_counter = 0

                # ==========================================
                # BLINKS
                # ==========================================
                if ear < self.EAR_THRESHOLD:

                    self.blink_counter += 1
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

                # ==========================================
                # YAWNING
                # ==========================================
                if mar > self.MAR_THRESHOLD:
<<<<<<< HEAD
                    self.yawn_counter += 1
                else:
                    self.yawn_counter = 0

                if self.yawn_counter > 5:
                    timestamp = datetime.now().strftime("%I:%M:%S %p")
                    event = f"{timestamp} - YAWNING"
                    if len(self.alert_history) == 0 or self.alert_history[-1] != event:
                        self.alert_history.append(event)
                    self.alert_history = self.alert_history[-10:]
=======

                    self.yawn_counter += 1

                else:

                    self.yawn_counter = 0

                # ==========================================
                # YAWN TEXT
                # ==========================================
                if self.yawn_counter > 5:

                    cv2.putText(
                        frame,
                        "YAWNING DETECTED!",
                        (30, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        3
                    )

                    current_event = "YAWNING"

                    timestamp = datetime.now().strftime(
                        "%I:%M:%S %p"
                    )

                    event = (
                        f"{timestamp} - "
                        f"{current_event}"
                    )

                    if (
                        len(self.alert_history) == 0
                        or self.alert_history[-1] != event
                    ):

                        self.alert_history.append(event)

                    self.alert_history = (
                        self.alert_history[-10:]
                    )
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

                # ==========================================
                # ALERT
                # ==========================================
                if self.frame_counter > 25:
<<<<<<< HEAD
                    final_alert = True
                    self.fatigue_score += 1

                    self.alert_system.play_alert()

                    timestamp = datetime.now().strftime("%I:%M:%S %p")
                    event = f"{timestamp} - ALERT"
                    if len(self.alert_history) == 0 or self.alert_history[-1] != event:
                        self.alert_history.append(event)
                    self.alert_history = self.alert_history[-10:]

                    current_time = time.time()
                    if (current_time - self.last_capture_time) > 5:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{self.screenshot_folder}/drowsy_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        self.last_capture_time = current_time

                else:
=======

                    final_alert = True

                    self.fatigue_score += 1

                    cv2.putText(
                        frame,
                        "HYBRID DROWSINESS ALERT!",
                        (30, 550),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    self.alert_system.play_alert()

                    current_event = "ALERT"

                    timestamp = datetime.now().strftime(
                        "%I:%M:%S %p"
                    )

                    event = (
                        f"{timestamp} - "
                        f"{current_event}"
                    )

                    if (
                        len(self.alert_history) == 0
                        or self.alert_history[-1] != event
                    ):

                        self.alert_history.append(event)

                    self.alert_history = (
                        self.alert_history[-10:]
                    )

                    # ==========================================
                    # SAVE SCREENSHOT
                    # ==========================================
                    current_time = time.time()

                    if (
                        current_time -
                        self.last_capture_time
                    ) > 5:

                        timestamp = datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )

                        filename = (
                            f"{self.screenshot_folder}/"
                            f"drowsy_{timestamp}.jpg"
                        )

                        cv2.imwrite(
                            filename,
                            frame
                        )

                        self.last_capture_time = current_time

                else:

>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493
                    self.alert_system.stop_alert()

                # ==========================================
                # CONFIDENCE SCORES
                # ==========================================
<<<<<<< HEAD
                eye_confidence = int(min(max((1 - ear / self.EAR_THRESHOLD) * 100, 0), 100))
                yawn_confidence = int(min(max((mar / self.MAR_THRESHOLD) * 100, 0), 100))
                fatigue_confidence = int(min((self.frame_counter / 25) * 100, 100))
=======
                eye_confidence = int(

                    min(

                        max(

                            (
                                1 -
                                ear /
                                self.EAR_THRESHOLD
                            ) * 100,

                            0

                        ),

                        100

                    )

                )

                yawn_confidence = int(

                    min(

                        max(

                            (
                                mar /
                                self.MAR_THRESHOLD
                            ) * 100,

                            0

                        ),

                        100

                    )

                )

                fatigue_confidence = int(

                    min(

                        (
                            self.frame_counter / 25
                        ) * 100,

                        100

                    )

                )
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

                # ==========================================
                # DRIVER STATUS
                # ==========================================
                status = "SAFE"
<<<<<<< HEAD
                color = (0, 255, 0)

                if final_alert:
                    status = "ALERT"
                    color = (0, 0, 255)
                elif self.yawn_counter > 5:
                    status = "YAWNING"
                    color = (0, 165, 255)
                elif self.frame_counter > 10:
                    status = "DROWSY"
                    color = (0, 255, 255)

                # ==========================================
                # ALERT OVERLAY ONLY
                # ==========================================
                if self.yawn_counter > 5:
                    cv2.putText(frame, "YAWNING", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

                if final_alert:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
=======

                color = (0, 255, 0)

                if final_alert:

                    status = "ALERT"

                    color = (0, 0, 255)

                elif self.yawn_counter > 5:

                    status = "YAWNING"

                    color = (0, 165, 255)

                elif self.frame_counter > 10:

                    status = "DROWSY"

                    color = (0, 255, 255)

                # ==========================================
                # DISPLAY INFO
                # ==========================================
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"Frames: {self.frame_counter}",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"Blinks: {self.blink_counter}",
                    (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    frame,
                    f"Fatigue Score: "
                    f"{self.fatigue_score}",
                    (30, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"STATUS: {status}",
                    (30, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3
                )

                # ==========================================
                # AI CONFIDENCE SCORES
                # ==========================================
                cv2.putText(
                    frame,
                    f"Eye Confidence: "
                    f"{eye_confidence}%",
                    (30, 620),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Yawn Confidence: "
                    f"{yawn_confidence}%",
                    (30, 660),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Fatigue Risk: "
                    f"{fatigue_confidence}%",
                    (30, 700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493

                # ==========================================
                # SAVE FEATURES
                # ==========================================
                label = 1 if final_alert else 0
<<<<<<< HEAD
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([ear, mar, self.frame_counter, self.fatigue_score, label])

        return frame
=======

                with open(
                    self.csv_file,
                    mode='a',
                    newline=''
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        ear,
                        mar,
                        self.frame_counter,
                        self.fatigue_score,
                        label
                    ])

        return frame
>>>>>>> 58ed91b24af1ba0243f83fdd6abd449730c5e493
