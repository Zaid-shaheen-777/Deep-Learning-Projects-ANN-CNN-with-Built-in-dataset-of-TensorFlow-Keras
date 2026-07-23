# CNN Projects (Convolutional Neural Networks)

The next step after ANNs — using `Conv2D` + `MaxPooling2D` layers to learn
spatial features instead of flattening images right away. Generally gives
better accuracy than the dense-only models in `ann-projects/`, especially
on more complex data like CIFAR-10.

| Notebook | Dataset | Highlights |
|---|---|---|
| `01_mnist_digits_cnn.ipynb` | MNIST (digits) | Basic 2-conv-layer CNN, reshaping grayscale images to add a channel dimension |
| `02_fashion_mnist_cnn.ipynb` | Fashion-MNIST | Same idea as above + a second experiment comparing `padding='valid'` vs `padding='same'` |
| `03_cifar10_cnn.ipynb` | CIFAR-10 (32×32 color images) | 2-conv-layer CNN, training curve (accuracy vs. val accuracy), per-image prediction visualization |
| `04_cifar10_cnn_deeper.ipynb` | CIFAR-10 | Deeper 3-conv-layer variant of the model above |

## What these show

- Preparing image data for CNNs (adding the channel dimension for grayscale
  images, since `Conv2D` expects `(height, width, channels)`)
- Building CNNs with `Conv2D` + `MaxPooling2D`, followed by `Flatten` +
  `Dense` for classification
- Comparing `padding='valid'` (default, shrinks feature maps) vs.
  `padding='same'` (keeps feature map size)
- Plotting training/validation accuracy curves to check for overfitting
- Visualizing predictions against ground truth, including CIFAR-10's 10
  object classes (airplane, car, bird, cat, deer, dog, frog, horse, ship,
  truck)

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
jupyter notebook 01_mnist_digits_cnn.ipynb

# or as a script
python 01_mnist_digits_cnn.py
```

For a runnable, interactive version of the digit recognizer, see
[`../app`](../app) — a small Streamlit app built from the same idea as
`01_mnist_digits_cnn.ipynb`.
