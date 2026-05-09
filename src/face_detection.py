
import cv2
import csv
import os
import time
import mediapipe as mp
import numpy as np

from datetime import datetime

from src.feature_extraction import calculate_ear
from src.feature_extraction import calculate_mar
from src.alert_system import AlertSystem


class FaceMeshDetector:

    def __init__(self):

        # ==========================================
        # CSV STORAGE
        # ==========================================
        self.csv_file = 'saved_features/fatigue_features.csv'

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

        # ==========================================
        # ALERT SYSTEM
        # ==========================================
        self.alert_system = AlertSystem()

        # ==========================================
        # SCREENSHOT STORAGE
        # ==========================================
        self.screenshot_folder = "outputs/screenshots"

        os.makedirs(
            self.screenshot_folder,
            exist_ok=True
        )

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

        # ==========================================
        # EYE LANDMARKS
        # ==========================================
        self.LEFT_EYE = [
            33, 160, 158,
            133, 153, 144
        ]

        self.RIGHT_EYE = [
            362, 385, 387,
            263, 373, 380
        ]

        # ==========================================
        # MOUTH LANDMARKS
        # ==========================================
        self.MOUTH = [
            13, 14, 78, 308
        ]


    # ==========================================
    # MAIN DETECTION
    # ==========================================
    def detect_face_mesh(self, frame):

        h, w, _ = frame.shape

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

                # ==========================================
                # DROWSINESS
                # ==========================================
                if ear < self.EAR_THRESHOLD:

                    self.frame_counter += 1

                else:

                    self.frame_counter = 0

                # ==========================================
                # BLINKS
                # ==========================================
                if ear < self.EAR_THRESHOLD:

                    self.blink_counter += 1

                # ==========================================
                # YAWNING
                # ==========================================
                if mar > self.MAR_THRESHOLD:

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

                # ==========================================
                # ALERT
                # ==========================================
                if self.frame_counter > 25:

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

                    self.alert_system.stop_alert()

                # ==========================================
                # CONFIDENCE SCORES
                # ==========================================
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

                # ==========================================
                # DRIVER STATUS
                # ==========================================
                status = "SAFE"

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

                # ==========================================
                # SAVE FEATURES
                # ==========================================
                label = 1 if final_alert else 0

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