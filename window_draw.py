import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()
        self.setup_bindings()
        self.model = load_model("model.h5")

    def setup_bindings(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.root.bind("<Return>", self.recognize)

    def draw(self, event):
        x, y = event.x, event.y
        r = 5  # радиус точки
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

    def recognize(self, event):
        # Получение изображения из холста
        img = self.get_image()
        # Преобразование изображения для передачи в нейронную сеть
        img = img.resize((28, 28)).convert("L")  # изменяем размер и переводим в черно-белое
        img_array = np.array(img) / 255.0  # нормализуем значения пикселей
        img_array = img_array.reshape(1, 28, 28, 1)  # добавляем размерность батча и канала
        # Передача изображения в нейронную сеть для распознавания
        prediction = self.model.predict(img_array)
        predicted_symbol = np.argmax(prediction)
        print("Predicted symbol:", predicted_symbol)

    def get_image(self):
        # Создание изображения из холста
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        img = ImageGrab.grab((x0, y0, x1, y1))
        return img

def main():
    root = tk.Tk()
    root.title("Drawing App")
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

input("Нажмите Enter для выхода...")
