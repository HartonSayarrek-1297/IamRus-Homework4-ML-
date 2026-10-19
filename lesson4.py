# launch command:
# cmd /k D: && cd D:\__BMSTU__\Python && python lesson4.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
               'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Нормализация: пиксели 0..255 -> 0..1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Добавляем канал: (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

print('Обучающая выборка:', x_train.shape)
print('Тестовая выборка:  ', x_test.shape)

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle('Примеры из обучающей выборки')
plt.tight_layout()
plt.show()

from tensorflow.keras import layers, models, regularizers, callbacks


model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),#, kernel_regularizer=regularizers.l2(0.0001)),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    
    layers.Dropout(0.35),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    # layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model.summary()

history = model.fit(
    x_train, y_train,
    epochs=30,
    validation_split=0.1,
    batch_size=64,
    callbacks=[early_stop]
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='обучение')
plt.plot(history.history['val_accuracy'], label='валидация')
plt.xlabel('Эпоха'); plt.ylabel('Точность')
plt.legend(); plt.title('Точность')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='обучение')
plt.plot(history.history['val_loss'], label='валидация')
plt.xlabel('Эпоха'); plt.ylabel('Потери')
plt.legend(); plt.title('Потери')

plt.tight_layout()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Точность на тестовой выборке: {test_acc:.4f}')


probs = model.predict(x_test[:10])
preds = np.argmax(probs, axis=1)
confs = np.max(probs, axis=1)


plt.figure(figsize=(14, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    true_label = class_names[y_test[i]]
    pred_label = class_names[preds[i]]
    color = 'green' if preds[i] == y_test[i] else 'red'
    plt.title(f'Ист: {true_label}\nПред: {pred_label}\n{confs[i]*100:.1f}%', color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()