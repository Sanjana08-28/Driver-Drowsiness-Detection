import cv2
import numpy as np

from tensorflow.keras.models import load_model


class CNNModel:

    def __init__(self):

        self.model = load_model(
            'models/mobilenetv2_eye_model.h5'
        )

        self.labels = [
            'closed',
            'open'
        ]

    def predict_eye(self, eye_image):

        eye = cv2.resize(
            eye_image,
            (224, 224)
        )

        eye = eye / 255.0

        eye = np.expand_dims(
            eye,
            axis=0
        )

        prediction = self.model.predict(
            eye,
            verbose=0
        )

        prediction_value = prediction[0][0]

        if prediction_value > 0.5:

            return 'open', prediction_value

        else:

            return 'closed', prediction_value