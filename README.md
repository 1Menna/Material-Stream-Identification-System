# Material Stream Identification System

**Team Members:** Menna Reda, Fatma Ibrahim, Sara Mohmed, Youssef, Mohammed  
**Project:** Automated Material Stream Identification using Machine Learning  
**Instructor:** Hanaa  
**TA:** Sara


---

## Project Overview

The Material Stream Identification (MSI) System is an end-to-end machine learning application designed to classify post-consumer waste into distinct material categories. This system emphasizes the complete ML pipeline: from data preprocessing and feature extraction to classifier training and real-time deployment.

The system currently classifies waste into seven classes:

- Glass
- Paper
- Cardboard
- Plastic
- Metal
- Trash
- Unknown (out-of-distribution or blurred inputs)

---

## Features

- **Data Preprocessing & Augmentation:**
  - Resize, normalize, and clean images.
  - Apply augmentation (rotation, flipping, scaling, color jitter) to increase dataset size by ≥30%.

- **Feature Extraction:**
  - Convert raw images into fixed-length numerical feature vectors.
  - Experimented with multiple feature descriptors (color, texture, histogram-based features).

- **Machine Learning Models:**
  - SVM Classifier: Trained on extracted features with hyperparameter tuning.
  - k-NN Classifier: Trained with different values of k and weighting schemes.
  - Best-performing model selected for real-time classification.

- **Real-Time Deployment:**
  - Processes live camera frames.
  - Displays the predicted class in real-time using OpenCV.

---

## Project Structure
```
Material-Stream-Identification/
│
├── dataset/                 # Original dataset (ignored in Git)
├── dataset_augmented/       # Augmented dataset (ignored in Git)
├── features/                # Feature vectors and labels
│   ├── X_features.npy
│   └── y_labels.npy
├── models/                  # Trained models
│   ├── svm_best.pkl
│   └── knn_best.pkl
├── src/                     # Training and preprocessing scripts
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── train_svm.py
│   └── train_knn.py
├── app/                     # Real-time application
│   ├── realtime_classifier.py
│   ├── model_loader.py
│   └── utils.py
├── notebooks/               # Experimentation notebooks
│   ├── feature_experiments.ipynb
│   ├── svm_testing.ipynb
│   └── knn_testing.ipynb
├── docs/                    # Project report
│   └── report.pdf
├── requirements.txt         # Python dependencies
└── main.py                  # Entry point for the project
```

---

## Installation

1. **Clone the repository:**
```bash
   git clone <your-repo-url>
   cd Material-Stream-Identification
```

2. **Create a virtual environment:**
```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

**Note:** `dataset/` and `dataset_augmented/` are not included due to size. Add your local dataset manually.

---

## Usage

### 1. Preprocessing & Augmentation
```bash
python src/preprocess.py
```

### 2. Feature Extraction
```bash
python src/extract_features.py
```

### 3. Train Models

- **Train SVM:**
```bash
  python src/train_svm.py
```

- **Train k-NN:**
```bash
  python src/train_knn.py
```

### 4. Real-Time Classification
```bash
python app/realtime_classifier.py
```

---

## Dependencies

- numpy
- pandas
- scikit-learn
- OpenCV (`opencv-python`)
- Pillow
- matplotlib
- joblib

(Full versions listed in `requirements.txt`)

---

## Contributing

- Fork the repository and create a new branch for your feature.
- Ensure your code follows the project structure and naming conventions.
- Submit a pull request for review before merging.

---

## License

This project is for academic purposes. Do not use without permission.
