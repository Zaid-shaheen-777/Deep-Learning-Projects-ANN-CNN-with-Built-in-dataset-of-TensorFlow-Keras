# Deep-Learning-Projects-ANN-CNN-with-Built-in-dataset-of-TensorFlow-Keras

# Deep Learning Projects — ANN & CNN with TensorFlow/Keras

A collection of small deep learning projects built while learning the
fundamentals of neural networks — from basic dense (ANN) networks to
convolutional (CNN) architectures — using TensorFlow/Keras on classic
image datasets (MNIST, Fashion-MNIST, CIFAR-10).

Includes a runnable **Streamlit app** for handwritten digit recognition
built from the concepts in these notebooks.

## 📁 Repository Structure

```
.
├── ann-projects/          # Dense (fully-connected) neural networks
│   ├── 01_mnist_digits_ann.ipynb / .py
│   └── 02_fashion_mnist_ann.ipynb / .py
├── cnn-projects/          # Convolutional neural networks
│   ├── 01_mnist_digits_cnn.ipynb / .py
│   ├── 02_fashion_mnist_cnn.ipynb / .py
│   ├── 03_cifar10_cnn.ipynb / .py
│   └── 04_cifar10_cnn_deeper.ipynb / .py
├── notes/                 # Concept notes written while learning
│   └── deep_learning_concepts.md
├── app/                   # Streamlit app: draw a digit, get a prediction
│   ├── app.py
│   ├── train_model.py
│   └── requirements.txt
├── requirements.txt        # Dependencies for the notebooks
└── README.md
```

## 🧠 Projects

### ANN (Dense Networks) — [`ann-projects/`](ann-projects)

| Notebook | Dataset | Test Accuracy |
|---|---|---|
| MNIST digits (ANN) | 60k handwritten digit images | ~97-98% |
| Fashion-MNIST (ANN) | 60k clothing images (10 classes) | ~87-88% |

### CNN (Convolutional Networks) — [`cnn-projects/`](cnn-projects)

| Notebook | Dataset | Notes |
|---|---|---|
| MNIST digits (CNN) | Handwritten digits | 2 conv layers, higher accuracy than the ANN version |
| Fashion-MNIST (CNN) | Clothing images | Includes a `padding='same'` vs `padding='valid'` comparison |
| CIFAR-10 (CNN) | 32×32 color images, 10 classes | Training/validation accuracy curves, prediction visualization |
| CIFAR-10 (deeper CNN) | Same as above | An extra conv layer to compare depth vs. accuracy |

Across both folders, the general pattern is: **load data → normalize →
build model → compile → train → evaluate → visualize predictions**, with
CNNs consistently outperforming the plain dense networks on the same
datasets since they exploit the 2D spatial structure of images.

Every notebook has a matching `.py` script (same code, with the markdown
explanations kept as `# %%` comments) if you'd rather read or run the flow
outside of Jupyter.

## ✏️ App — Handwritten Digit Recognizer

An interactive Streamlit app (in [`app/`](app)) where you can draw a digit
or upload an image and see the model's prediction live, along with a
probability breakdown for all 10 digits.

```bash
cd app

# create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python train_model.py       # trains and saves the CNN (~1 min on CPU)
streamlit run app.py
```

## 📓 Notes

[`notes/deep_learning_concepts.md`](notes/deep_learning_concepts.md)
contains the conceptual notes taken while learning — neurons, layers,
activation functions (ReLU vs. sigmoid), and a walkthrough of the first
MNIST model — before jumping into the project notebooks above.

## 🚀 Getting Started

```bash
git clone <this-repo-url>
cd deep-learning-projects

# create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch Jupyter
jupyter notebook
```

Open any notebook in `ann-projects/` or `cnn-projects/` and run all cells.
Datasets (MNIST, Fashion-MNIST, CIFAR-10) are downloaded automatically via
`keras.datasets` on first run.

## 🛠️ Tech Stack

- Python, TensorFlow / Keras
- NumPy, Matplotlib, Pillow
- Streamlit (for the digit-recognizer app)

## 💡 What I Learned

- How neurons, layers, weights, biases, and activation functions combine
  into a trainable network
- The difference between dense (ANN) and convolutional (CNN) architectures,
  and why CNNs generalize better on image data
- The effect of padding (`'same'` vs `'valid'`) on feature map sizes
- How to evaluate a model beyond accuracy — inspecting misclassified
  examples and training/validation curves to spot overfitting
- Turning a trained model into a small interactive app

## 📌 Possible Next Steps

- Add data augmentation and dropout to reduce overfitting on CIFAR-10
- Try transfer learning (e.g. a pretrained backbone) on CIFAR-10
- Add a confusion matrix for each project's evaluation section
