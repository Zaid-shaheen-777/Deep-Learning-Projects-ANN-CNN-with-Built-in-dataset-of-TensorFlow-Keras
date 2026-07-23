"""
CIFAR-10 — Deeper CNN (3 conv layers)

Plain-Python version of `04_cifar10_cnn_deeper.ipynb` (same code and flow,
just exported so it's easy to read/run without Jupyter).
"""

# %%
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %%
class_names = [
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# %%
plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')

plt.show()

# %%
model = keras.Sequential([
    # Feature extraction
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    # Flatten + Dense
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

