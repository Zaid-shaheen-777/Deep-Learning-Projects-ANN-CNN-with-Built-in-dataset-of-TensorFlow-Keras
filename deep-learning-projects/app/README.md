# ✏️ Handwritten Digit Recognizer (Streamlit App)

A small web app that lets you draw or upload a digit (0–9) and get a live
prediction from a CNN trained on MNIST.

## Run locally

```bash
cd app

# create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# (optional) train the model ahead of time — otherwise the app trains it
# automatically on first launch
python train_model.py

streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How it works

1. `train_model.py` trains a small CNN (2 conv layers + dense head) on MNIST
   and saves it as `mnist_cnn.keras`.
2. `app.py` loads that model, lets you draw a digit on a canvas (or upload an
   image), preprocesses it to a 28×28 grayscale array the same way MNIST
   images are formatted, and shows the predicted digit with a probability
   chart for all 10 classes.

## Notes

- If `streamlit-drawable-canvas` isn't installed, the drawing tab shows a
  message and you can still use the **Upload an image** tab.
- Uploaded images are auto-inverted if they look like dark digits on a
  light background, so both "MNIST-style" and normal photos of handwriting
  should work reasonably well.
