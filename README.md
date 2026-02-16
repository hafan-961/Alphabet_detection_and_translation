# Alphabet_detection_and_translation
# 🖋️ NumPy Neural Engine: Digital Canvas & Translator
> **A Custom Deep Learning Model built from scratch in NumPy. Draw letters on a digital canvas and translate English words to Spanish in real-time.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Math-NumPy-orange?logo=numpy&logoColor=white)](https://numpy.org/)
[![Deployment: Render](https://img.shields.io/badge/Deployment-Render-00b3b0?logo=render&logoColor=white)](https://render.com)

---

## 🌟 Project Overview
This project is an end-to-end implementation of a **Deep Learning Model** (Multi-Layer Perceptron) created entirely without high-level frameworks like TensorFlow or PyTorch. It features an interactive **OpenCV Digital Canvas** where users draw characters. The system recognizes these characters using a custom NumPy-based engine, assembles them into words, and performs instant translation.



---

## 🧠 The Hand-Coded Architecture
Instead of using a pre-made OCR library, I engineered the mathematical foundations to handle the **EMNIST Letters dataset** (26 classes). 

### 🚀 Key Deep Learning Implementations:
| Feature | Implementation Detail |
| :--- | :--- |
| **Optimization** | **Adam & RMSprop** built from scratch to accelerate convergence and navigate complex loss landscapes. |
| **Normalization** | **Batch Normalization** implemented in hidden layers to stabilize training and allow for higher learning rates. |
| **Regularization** | **Inverted Dropout** (keep_prob=0.8) and **L2 Regularization** to ensure the model generalizes beyond training data. |
| **Initialization** | **He Initialization** utilized to prevent vanishing/exploding gradients in deep ReLU architectures. |
| **Activation Functions** | **Leaky ReLU** for hidden layers to prevent "dying neurons" and **Softmax** for multi-class probability output. |

---

## 🏗️ Project Structure
The codebase is strictly modular, separating the mathematical engine from the interface logic.

```bash
Alphabet_detection_and_translation/
├── src/
│   ├── engine.py           # Core Forward & Backward Propagation logic
│   ├── optimizers.py       # Hand-coded Adam and Momentum update logic
│   ├── layers.py           # He Initialization and Batch Norm formulas
│   ├── regularization.py   # L2 Penalty and Inverted Dropout masks
│   ├── activations.py      # Leaky ReLU and Softmax implementations
│   ├── preprocessing.py    # Data shuffling and orientation correction
│   └── translator.py       # Word formation logic and Spanish translation
├── weights/
│   └── model_params.npy    # Trained parameters (W, b, gamma, beta)
├── app.py                  # Flask Backend for Render deployment
├── canvas_translator.py    # Interactive Digital Drawing Interface (OpenCV)
└── train.py                # Main script for training the NumPy model
```
---
## Getting Started
1.Setup
```bash
git clone https://github.com/hafan-961/Alphabet_detection_and_translation.git
cd Alphabet_detection_and_translation
pip install -r requirements.txt
```
2.Training
```bash
python train.py
```

3.Usage (Canvas Mode)
```bash
python canvas_translator.py
```
---
## 🌐Deployment
This project is deployed on Render using a Flask backend. The web interface replicates the canvas experience, allowing for cross-platform testing of the NumPy engine.
