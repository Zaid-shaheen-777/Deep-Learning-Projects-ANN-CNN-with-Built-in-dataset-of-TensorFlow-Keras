"""
Fashion-MNIST — CNN (with padding comparison)

Plain-Python version of `02_fashion_mnist_cnn.ipynb` (same code and flow,
just exported so it's easy to read/run without Jupyter).
"""

# %%
from tensorflow import keras
import matplotlib.pyplot as plt

# %%
fashion = keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion.load_data()

# %%
plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    
    plt.axis('off')

plt.show()

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %%
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(x_train.shape)

# %%
model = keras.Sequential([
    
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

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

# %% [markdown]
# ## Check With Padding

# %%
model = keras.Sequential([
    # First Convolutional Layer
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        input_shape=(28, 28, 1)
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Layer
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the feature maps
    keras.layers.Flatten(),

    # Fully Connected Layer
    keras.layers.Dense(64, activation='relu'),

    # Output Layer
    keras.layers.Dense(10, activation='softmax')
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

