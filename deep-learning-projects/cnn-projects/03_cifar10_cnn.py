"""
CIFAR-10 — CNN with visualization

Plain-Python version of `03_cifar10_cnn.ipynb` (same code and flow,
just exported so it's easy to read/run without Jupyter).
"""

# %%
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(x_train.shape)

# %%
class_names = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

# %%
plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')

plt.show()

# %% [markdown]
# ## Normalize

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %% [markdown]
# ## FIRST CNN MODEL

# %%
model = keras.Sequential([

    keras.layers.Conv2D(32,(3,3),
                        activation='relu',
                        input_shape=(32,32,3)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

# %% [markdown]
# ## Compile

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# ## Train

# %%
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    validation_data=(x_test,y_test)
)

# %% [markdown]
# ## Training Graph (IMPORTANT)

# %%
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

# %% [markdown]
# ## Prediction Check (REAL TEST)

# %%
predictions = model.predict(x_test)

# %% [markdown]
# ## One Image Test

# %%
index = 10

plt.imshow(x_test[index])
plt.axis('off')
plt.show()

# %% [markdown]
# ## What Models output

# %%
index = 10 

predicted_label = class_names[np.argmax(predictions[index])]
true_label = class_names[y_test[index][0]]

print("Prediction:", predicted_label)
print("Actual:", true_label)

# %% [markdown]
# ## Engineer Move

# %%
for i in range(5):
    plt.imshow(x_test[i])
    plt.title(
        f"Pred: {class_names[np.argmax(predictions[i])]} | "
        f"True: {class_names[y_test[i][0]]}"
    )
    plt.axis('off')
    plt.show()

