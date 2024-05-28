import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

class Recognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, img):
        # Преобразование изображения для передачи в нейронную сеть
        img = img.resize((28, 28)).convert("L")  # изменяем размер и переводим в черно-белое
        img_array = np.array(img) / 255.0  # нормализуем значения пикселей
        img_array = img_array.reshape(1, 28, 28, 1)  # добавляем размерность батча и канала
        # Передача изображения в нейронную сеть для распознавания
        prediction = self.model.predict(img_array)
        predicted_symbol = np.argmax(prediction)
        return predicted_symbol

input("Нажмите Enter для выхода...")