import os
import cv2
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_best.pkl")
KNN_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_knn.pkl")

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
UNKNOWN_LABEL = "UNKNOWN"
CONFIDENCE_THRESHOLD = 0.5
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def predict(dataFilePath=IMAGE_FOLDER, bestModelPath=KNN_MODEL_PATH):
    knn_model = joblib.load(bestModelPath)
    knn_scaler = joblib.load(KNN_SCALER_PATH)

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    predictions = []

    for file_name in sorted(os.listdir(dataFilePath)):
        if not file_name.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(dataFilePath, file_name)
        img = cv2.imread(img_path)

        if img is None:
            predictions.append(6)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)

        features = base_model.predict(img_preprocessed, verbose=0)

        features_scaled = knn_scaler.transform(features)
        probs = knn_model.predict_proba(features_scaled)[0]
        max_prob = np.max(probs)

        if max_prob < CONFIDENCE_THRESHOLD:
            predictions.append(int(6))
        else:
            predictions.append(int(np.argmax(probs)))

    return predictions

results = predict(IMAGE_FOLDER, KNN_MODEL_PATH)
print(results)
