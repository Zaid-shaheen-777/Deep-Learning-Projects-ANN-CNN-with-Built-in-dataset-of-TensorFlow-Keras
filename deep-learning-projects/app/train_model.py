"""
train_model.py
---------------
Trains a small CNN on MNIST and saves it as `mnist_cnn.keras`.

Run this once before starting the Streamlit app:

    python train_model.py

This mirrors the architecture used in `cnn-projects/01_mnist_digits_cnn.ipynb`,
just packaged as a standalone script so `app.py` has a model file to load.
"""

import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "mnist_cnn.keras"


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_model()
    model.summary()

    model.fit(
        x_train, y_train,
        epochs=5,
        validation_split=0.1,
        batch_size=128,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
