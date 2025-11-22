# MNIST — Comparative Study: CNN vs LSTM

## 1. Objective

This project presents a comparative study between two neural network architectures for handwritten digit classification using the MNIST dataset:

* **Convolutional Neural Network (CNN)**
* **Long Short-Term Memory network (LSTM)**

Both models are trained, evaluated, and compared based on test accuracy, loss, and training time. This repository includes training scripts, utilities for preparing the dataset, saved models, evaluation outputs, and plots.

---

## 2. Project Structure

```
mnist-cnn-vs-lstm/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ utils.py
│  ├─ train_cnn.py
│  ├─ train_lstm.py
│  └─ evaluate.py
├─ outputs/
│  ├─ models/
│  │  ├─ cnn_model.h5
│  │  └─ lstm_model.h5
│  ├─ figures/
│  │  ├─ cnn_accuracy.png
│  │  ├─ cnn_loss.png
│  │  ├─ lstm_accuracy.png
│  │  └─ lstm_loss.png
│  └─ results.csv
└─ LICENSE
```

---

## 3. Installation

### Clone repository

```bash
git clone <repo_url>
cd mnist-cnn-vs-lstm
```

### Create virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Dataset Preparation

Both models follow the same preprocessing steps:

* Normalize pixel values to the range **[0,1]**.
* Convert labels to **one-hot encoding**.
* Reshape images depending on the model:

  * **CNN:** `(28, 28, 1)` grayscale image.
  * **LSTM:** `(28, 28)` treating each row as a time step.

---

## 5. Training the Models

### Train CNN

```bash
python src/train_cnn.py
```

### Train LSTM

```bash
python src/train_lstm.py
```

Training outputs include loss & accuracy curves and saved `.h5` model files.

---

## 6. Model Performance

The following figures illustrate the learning curves of each model:

### **CNN Accuracy Curve**

![CNN Accuracy](https://github.com/mohammedPhd/minst/blob/main/outputs/figures/cnn_accuracy.png)

### **CNN Loss Curve**

![CNN Loss](https://github.com/mohammedPhd/minst/blob/main/outputs/figures/cnn_loss.png)

### **LSTM Accuracy Curve**

![LSTM Accuracy](https://github.com/mohammedPhd/minst/blob/main/outputs/figures/lstm_accuracy.png)

### **LSTM Loss Curve**

![LSTM Loss](https://github.com/mohammedPhd/minst/blob/main/outputs/figures/lstm_loss.png)

---

## 7. Results (from `results.csv`)

| Model | Train Time (s) | Test Loss | Test Accuracy |
| ----- | -------------- | --------- | ------------- |
| LSTM  | 141.45         | 0.0509    | 0.9845        |
| CNN   | 421.79         | 0.0289    | 0.9913        |

![Model Comparison](https://github.com/mohammedPhd/minst/blob/main/outputs/figures/comparison.png)
---

## 8. Discussion

### **1. CNN Outperforms LSTM in Accuracy**

The CNN achieves **0.9913 accuracy**, benefiting from convolution operations that capture spatial patterns.

### **2. LSTM Trains Faster**

Despite lower accuracy, the LSTM trains in **141 seconds**, significantly faster than the CNN training time.

### **3. CNN Shows Lower Loss**

The CNN generalizes better with a **lower test loss**.
