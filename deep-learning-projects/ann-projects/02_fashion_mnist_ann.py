"""
Fashion-MNIST — ANN (Dense Network)

Plain-Python version of `02_fashion_mnist_ann.ipynb` (same code and flow,
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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

# %%

model.compile(
    optimizer = "adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.fit(x_train,y_train, epochs = 5)

# %%
model.evaluate(x_test,y_test)

# %%
index = 0

plt.imshow(x_test[index])
plt.axis('off')
plt.show()

# %%
y_pred = model.predict(x_test)

# %%
y_pred_labels = y_pred.argmax(axis=1)

# %%
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# %%
index = 0

plt.imshow(x_test[index], cmap='gray')
plt.title(f"Actual: {class_names[y_test[index]]} | Predicted: {class_names[y_pred_labels[index]]}")
plt.axis('off')
plt.show()

# %%
plt.figure(figsize=(10,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i], cmap='gray')
    
    actual = class_names[y_test[i]]
    predicted = class_names[y_pred_labels[i]]
    
    plt.title(f"A:{actual}\nP:{predicted}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %%
import numpy as np

wrong = (y_pred_labels != y_test)

wrong_indices = np.where(wrong)[0]

index = wrong_indices[0]

plt.imshow(x_test[index], cmap='gray')
plt.title(f"Actual: {class_names[y_test[index]]} | Predicted: {class_names[y_pred_labels[index]]}")
plt.axis('off')
plt.show()

