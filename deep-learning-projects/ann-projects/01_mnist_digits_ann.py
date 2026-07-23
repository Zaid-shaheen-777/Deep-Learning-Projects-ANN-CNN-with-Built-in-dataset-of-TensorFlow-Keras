"""
MNIST Handwritten Digits — ANN (Dense Network)

Plain-Python version of `01_mnist_digits_ann.ipynb` (same code and flow,
just exported so it's easy to read/run without Jupyter).
"""

# %%
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Load Dataset

# %%
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
x_train[0].shape

# %%
y_train

# %%
# Show first image
plt.imshow(x_train[0], cmap='gray')

# Show label
plt.title(f"Label: {y_train[0]}")

# Remove axis
plt.axis('off')

# Display image
plt.show()

# %% [markdown]
# ## Shape

# %%
print(x_train.shape)
print(x_test.shape)

# %% [markdown]
# ## Normalize

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %% [markdown]
# ## Model (Dense Network)

# %%
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
    
])

# %%
model.summary()

# %%
model.compile(
    optimizer = "adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.fit(x_train,y_train, epochs = 10,validation_split=0.2)

# %%
model.evaluate(x_test,y_test)

# %%
test_image = x_test[0]

# Actual answer
actual_label = y_test[0]

# Predict
prediction = model.predict(test_image.reshape(1, 28, 28))

# Highest probability
predicted_label = np.argmax(prediction)

# =========================
# Show Image
# =========================
plt.imshow(test_image, cmap='gray')

plt.title(
    f"Actual: {actual_label} | Predicted: {predicted_label}"
)

plt.axis('off')
plt.show()

# Print probabilities
print("Prediction Probabilities:")
print(prediction)

# %% [markdown]
# ## Test on Your Own Handwritten Digit
#
# The cell below loads a local image file (`7.jpg`) and runs it through the trained model. To try this yourself:
# 1. Take a photo (or scan) of a digit you've written on paper.
# 2. Save it in this folder and update `image_path` below to point to it.
# 3. Run the cell — it will resize, invert, and normalize the image to match MNIST's format before predicting.

# %%
from PIL import Image

image_path = "7.jpg"

# Open image
img = Image.open(image_path).convert("L")

# Resize to 28x28
img = img.resize((28, 28))

# Convert to numpy array
img_array = np.array(img)

# Invert colors if needed
# (MNIST = white digit on black background)
img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0

# Show image
plt.imshow(img_array, cmap='gray')
plt.title("Your Uploaded Image")
plt.show()

# Reshape for model
img_array = img_array.reshape(1, 28, 28)

# =========================
# 7. Predict Digit
# =========================
prediction = model.predict(img_array)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)

# Show probabilities
print("Probabilities:")
print(prediction)

