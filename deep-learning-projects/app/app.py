"""
app.py
------
A small Streamlit app that recognizes handwritten digits (0-9) using the
CNN trained in `cnn-projects/01_mnist_digits_cnn.ipynb` (packaged here via
`train_model.py`).

Run with:
    streamlit run app.py

The first run will train and cache a model automatically if
`mnist_cnn.keras` isn't found (takes ~1 minute on CPU).
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("No trained model found — training one now (~1 min)..."):
            import train_model
            train_model.main()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert a PIL image into the (1, 28, 28, 1) array the model expects."""
    img = img.convert("L").resize((28, 28))
    arr = np.array(img).astype("float32")

    # MNIST digits are white-on-black. If the uploaded/drawn image looks like
    # black-on-white (i.e. the background is bright), invert it.
    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


st.title("✏️ Handwritten Digit Recognizer")
st.write(
    "Draw a digit (0–9) below, or upload an image, and the CNN will predict "
    "what it sees. Trained on the MNIST dataset."
)

model = load_model()

tab_draw, tab_upload = st.tabs(["Draw", "Upload an image"])

image_to_predict = None

with tab_draw:
    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
            image_to_predict = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            )
    except ImportError:
        st.info(
            "Drawing canvas needs the `streamlit-drawable-canvas` package "
            "(see requirements.txt). Use the **Upload an image** tab instead."
        )

with tab_upload:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)

if image_to_predict is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(image_to_predict, width=150)

    processed = preprocess(image_to_predict)
    predictions = model.predict(processed, verbose=0)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit])

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted digit", predicted_digit, f"{confidence:.1%} confidence")

    st.subheader("Probabilities per digit")
    st.bar_chart({str(i): float(p) for i, p in enumerate(predictions)})
else:
    st.info("Draw a digit above or upload an image to get a prediction.")
