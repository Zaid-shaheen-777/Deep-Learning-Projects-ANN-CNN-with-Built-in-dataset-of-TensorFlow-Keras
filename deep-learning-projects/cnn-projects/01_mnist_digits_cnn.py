"""
MNIST Handwritten Digits — CNN

Plain-Python version of `01_mnist_digits_cnn.ipynb` (same code and flow,
just exported so it's easy to read/run without Jupyter).
"""

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# %%
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# %%
plt.imshow(x_test[3], cmap='gray')
plt.title(f"Label: {y_test[3]}")
plt.axis("off")
plt.show()

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %%
print(x_train.min(), x_train.max())

# %% [markdown]
# CNNs expect images to have three dimensions:
#
# - Height
# - Width
# - Channels
#
# Our images are grayscale, so they have 1 channel.

# %%
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(x_train.shape)

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.summary()

# %%
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# %%
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

# %%
predictions = model.predict(x_test)

# %%
print(predictions[0])

# %%
import numpy as np

predicted_digit = np.argmax(predictions[1])

print("Predicted:", predicted_digit)
print("Actual:", y_test[1])

# %%
plt.imshow(x_test[1].squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_digit}, Actual: {y_test[1]}")
plt.axis("off")
plt.show()

# %%
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap='gray')

    prediction = np.argmax(predictions[i])

    plt.title(f"P:{prediction}\nA:{y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

