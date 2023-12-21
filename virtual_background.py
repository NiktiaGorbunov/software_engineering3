import cv2
import math
import numpy as np
import mediapipe as mp

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def process_image(image, background_image):
    with mp.solutions.selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
        img_as_np = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(img_as_np, flags=1)

        # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bg_as_np = np.frombuffer(background_image, dtype=np.uint8)
        background_image = cv2.imdecode(bg_as_np, flags=1)

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image, background_image)

        h, w = output_image.shape[:2]
        if h < w:
            img = cv2.resize(output_image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
        else:
            img = cv2.resize(output_image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    image_bytes = cv2.imencode('.jpg', img)[1].tobytes()


    return image_bytes



