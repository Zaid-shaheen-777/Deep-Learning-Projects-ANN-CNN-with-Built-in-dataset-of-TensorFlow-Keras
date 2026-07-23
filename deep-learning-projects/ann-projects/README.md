# ANN Projects (Dense Neural Networks)

Fully-connected (dense) networks applied to image classification. These
were the first models built while learning deep learning fundamentals —
no convolutions yet, just `Flatten` + `Dense` layers.

| Notebook | Dataset | Architecture | Test Accuracy |
|---|---|---|---|
| `01_mnist_digits_ann.ipynb` | MNIST (handwritten digits) | Flatten → Dense(256) → Dense(128) → Dense(64) → Dense(10, softmax) | ~97-98% |
| `02_fashion_mnist_ann.ipynb` | Fashion-MNIST (clothing) | Flatten → Dense(128) → Dense(10, softmax) | ~87-88% |

## What these show

- How to load and normalize image data (pixel values 0–255 → 0–1)
- Building a `keras.Sequential` dense network
- Training with `model.fit`, evaluating with `model.evaluate`
- Visualizing predictions vs. actual labels, including misclassified
  examples (fashion notebook)
- Note: `01_mnist_digits_ann.ipynb` also includes a cell to test the model
  on your own hand-drawn digit image (`7.jpg`) — supply your own image path
  to try it.

## Run

Each notebook also has a plain-Python `.py` twin (same code, markdown notes
kept as `# %%` comments) if you'd rather run it as a script or step through
it in an editor like VS Code/PyCharm instead of Jupyter.

```bash
# from the repo root: create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r ../requirements.txt

# as a notebook
jupyter notebook 01_mnist_digits_ann.ipynb

# or as a script
python 01_mnist_digits_ann.py
```
