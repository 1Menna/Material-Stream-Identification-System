import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

print("Loading features...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "..", "features")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "svm_best.pkl")

X_train = np.load(os.path.join(FEATURES_DIR, "X_features.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_labels.npy"))

X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training SVM using scikit-learn...")
# You can adjust C and kernel as needed
svm = SVC(C=1.0, kernel='linear', probability=True)
svm.fit(X_train, y_train)

print("Final results:")
print("Train accuracy:", svm.score(X_train, y_train))
print("Val accuracy:", svm.score(X_test, y_test))

# Save model + scaler
with open(model_path, "wb") as f:
    pickle.dump({
        "svm": svm,
        "scaler": scaler
    }, f)

print("Model saved to:", model_path)
