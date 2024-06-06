'''
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование данных
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=2)

# Сохранение модели
model.save('model.keras')

print("Model saved as model.keras")
'''


import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Путь к вашему датасету
dataset_path = "path/to/your/dataset"

# Список классов (названия папок)
class_names = os.listdir(dataset_path)

# Загрузка изображений и меток
images = []
labels = []

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    class_index = class_names.index(class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = Image.open(image_path).convert('L')  # Преобразование в оттенки серого
        image = image.resize((28, 28))  # Изменение размера изображения до 28x28
        image_array = np.array(image).astype('float32') / 255.0  # Нормализация значений пикселей
        images.append(image_array)
        labels.append(class_index)

images = np.array(images)
labels = np.array(labels)

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Преобразование меток в one-hot формат
num_classes = len(class_names)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Добавление канала к изображениям
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Сохранение модели
model.save('model.keras')


input("Нажмите Enter для выхода...")
