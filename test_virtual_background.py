import unittest
import cv2
import numpy as np
from virtual_background import process_image


class TestVirtualBackground(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_process_image(self):
        # Тест: функция корректно обрабатывает изображение

        # Создаем тестовые изображения
        data_img = cv2.imread('datasets/cloun.jpg')
        data_img_bytes = cv2.imencode('.jpg', data_img)[1].tobytes()

        back_img = cv2.imread('backgrounds/back_test.jpg')
        back_img_bytes = cv2.imencode('.jpg', back_img)[1].tobytes()

        # Вызываем функцию обработки изображения
        processed_img_bytes = process_image(data_img_bytes, back_img_bytes)
        nparr = np.frombuffer(processed_img_bytes, np.byte)
        processed_img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        # Проверяем, что изображение не является пустым
        self.assertIsNotNone(processed_img)

        # Проверяем, что размеры диагоналей совпадают
        diagonal_data_img = np.linalg.norm(data_img.shape[:2])
        diagonal_back_img = np.linalg.norm(back_img.shape[:2])
        diagonal_processed_img = np.linalg.norm(processed_img.shape[:2])

        self.assertEqual(diagonal_data_img, diagonal_back_img)
        self.assertEqual(diagonal_data_img, diagonal_processed_img)


if __name__ == '__main__':
    unittest.main()
