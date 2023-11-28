import cv2
import math
import numpy as np
import mediapipe as mp

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

PATH_TO_BACKGROUNDS = 'backgrounds/back_test.jpg'
PATH_TO_IMAGE = 'datasets/cloun.jpg'
PATH_TO_SAVE = 'output/cloun.jpg'


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    # Save image to 'output'
    cv2.imwrite(PATH_TO_SAVE, img)

    cv2.waitKey(0)

    # It is for removing/deleting created GUI window from screen and memory
    cv2.destroyAllWindows()

image = cv2.imread(PATH_TO_IMAGE)

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Blur the image background based on the segementation mask.
with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
    # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bg_image = cv2.imread(PATH_TO_BACKGROUNDS)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    output_image = np.where(condition, image, bg_image)

    resize_and_show(output_image)
