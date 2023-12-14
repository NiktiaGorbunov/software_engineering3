import cv2
import math
import numpy as np
import mediapipe as mp
# import matplotlib.pyplot as plt

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


# для теста вывод кртинки!!!

data_img = cv2.imread('datasets/cloun.jpg')
data_img = cv2.imencode('.jpg', data_img)[1].tobytes()

back_img = cv2.imread('backgrounds/back_test.jpg')
back_img = cv2.imencode('.jpg', back_img)[1].tobytes()

img = process_image(data_img, back_img)

nparr = np.frombuffer(img, np.byte)
img2 = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

cv2.imshow('img', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.show()
