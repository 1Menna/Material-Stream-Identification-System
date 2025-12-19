import os
import sys
import cv2
import joblib
import pickle
import numpy as np
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

CONFIDENCE_THRESHOLD = 0.5
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_best.pkl")
SVM_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_svm.pkl")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_best.pkl")
KNN_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_knn.pkl")

def robust_load(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        return joblib.load(path)
    except:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

print("Loading Models & Feature Extractor...")
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

svm_model = robust_load(SVM_MODEL_PATH)
svm_scaler = robust_load(SVM_SCALER_PATH)
knn_model = robust_load(KNN_MODEL_PATH)
knn_scaler = robust_load(KNN_SCALER_PATH)

if svm_model is None and knn_model is None:
    print("Fatal Error: Could not load models.")
    sys.exit(1)

def get_prediction(model, scaler, features):
    if model is None: return "N/A", 0.0
    
    processed_feats = features
    if scaler is not None:
        processed_feats = scaler.transform(features)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_feats)[0]
        conf = np.max(probs)
        label = CLASSES[np.argmax(probs)].upper() if conf >= CONFIDENCE_THRESHOLD else "UNKNOWN"
        return label, conf
    return "UNKNOWN", 0.0

cap = cv2.VideoCapture(0)
print("System Ready. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    size = 250
    x1, y1 = (w - size) // 2, (h - size) // 2
    x2, y2 = x1 + size, y1 + size
    roi = frame[y1:y2, x1:x2]

    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    img_pre = preprocess_input(np.expand_dims(img_res, axis=0))
    feats = base_model.predict(img_pre, verbose=0)

    s_label, s_conf = get_prediction(svm_model, svm_scaler, feats)
    k_label, k_conf = get_prediction(knn_model, knn_scaler, feats)

    color = (0, 255, 0) if (s_conf > 0.6 or k_conf > 0.6) else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    svm_str = f"SVM: {s_label} | Conf: {s_conf*100:.1f}%"
    cv2.putText(frame, svm_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    knn_str = f"KNN: {k_label} | Conf: {k_conf*100:.1f}%"
    cv2.putText(frame, knn_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Waste Stream Identification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
