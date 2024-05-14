import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных (используем MNIST для примера)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация изображений
train_images = train_images / 255.0
test_images = test_images / 255.0

# Изменение формы данных для подачи в CNN
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Преобразование меток в категориальные
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Измените число классов, если у вас другой набор данных
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels)
)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')


model.save('handwritten_math_symbol_recognition_model.h5')



import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Загрузка модели
model = load_model('handwritten_math_symbol_recognition_model.h5')

# Загрузка и предобработка нового изображения
img_path = 'path_to_new_image.png'  # Укажите путь к изображению
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Предсказание
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f'Predicted class: {predicted_class}')
