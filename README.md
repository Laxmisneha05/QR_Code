# 🔍 QR Code Authentication – Original vs. Counterfeit Detection

This notebook project is focused on detecting whether a **printed QR code is genuine or counterfeit** using image classification techniques. It explores both traditional machine learning (SVM) and deep learning (CNN) models for robust detection.

---

## 🧠 What’s Inside This Notebook?

### 1. 🚀 Setup
We begin by installing all the required Python libraries:
- `numpy`, `pandas`, `opencv-python`
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for traditional ML algorithms
- `tensorflow`, `keras`, `torch` for deep learning models

---

### 2. 📂 Data Loading
- The dataset is loaded directly from **Google Drive**.
- Folder path used: `/My Drive/new/alemeno/`
- Each subfolder represents a class (e.g., `original`, `fake`, `first print`, `second print`).
- A custom function walks through the directories, reads image paths, and assigns corresponding labels.

---

### 3. 🖼️ Sample Image Display
- Visualizes a few sample images to confirm successful loading and label distribution.
- Helps in verifying data quality before training.

---

### 4. 🧹 Image Preprocessing
- Images are resized and normalized using **OpenCV** and **NumPy**.
- Grayscale conversion may also be applied.
- This preprocessing ensures that all images are in a consistent format for both machine learning and deep learning models.

---

### 5. 🧠 Model Training

This notebook explores two different approaches:

#### 🔹 Support Vector Machine (SVM)
- Implements an SVM classifier with a linear kernel.
- Evaluates model performance using:
  - Accuracy score
  - Classification report
  - Confusion matrix
- Quick to train and provides a strong baseline for performance.

#### 🔸 Convolutional Neural Network (CNN)
- Builds a deep learning model using **Keras** or **PyTorch**.
- CNN captures intricate visual patterns from QR code images.
- Suitable for cases where image complexity or noise affects classification accuracy.

---

### 6. 📊 Evaluation
- Accuracy and classification reports are generated for both models.
- Confusion matrix is visualized using `seaborn` to show the number of correct and incorrect predictions for each class.
- Model comparisons can be done to choose the best performer.

---

## 💡 Why This Project?

Fake QR codes are a growing problem — from misleading payment links to counterfeit products. This project provides a step toward **automated authentication** of QR prints using visual recognition techniques.

---

## 🛠️ Requirements

Install all required dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tensorflow keras torch
