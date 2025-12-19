import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

os.makedirs("models", exist_ok=True)

X_train = np.load("../features/X_features.npy")
y_train = np.load("../features/y_labels.npy")

print("Train features:", X_train.shape)
print("Train labels:", y_train.shape)

X_test = np.load("../features/X_test.npy")
y_test = np.load("../features/y_test.npy")

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

X_unknown = np.load("../features/X_unknown.npy")
y_unknown = np.load("../features/y_unknown.npy")

print("Unknown features:", X_unknown.shape)
print("Unknown labels:", y_unknown.shape)

X_eval = np.vstack((X_test, X_unknown))
y_eval = np.concatenate((y_test, y_unknown))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

print("Feature scaling done")

C_values = [0.1, 1, 10]
kernels = ["linear", "poly"]
gammas = ["scale", "auto"]

best_accuracy = 0
best_params = None
best_model = None

for C in C_values:
    for kernel in kernels:
        for gamma in gammas if kernel == "rbf" else ["scale"]:
            print(f"Training SVM: C={C}, kernel={kernel}, gamma={gamma}")

            svm = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True
            )

            svm.fit(X_train, y_train)

            y_val = svm.predict(X_test)
            acc = accuracy_score(y_test, y_val)
            print(f"Validation Accuracy: {acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = (C, kernel, gamma)
                best_model = svm

print("\nBest SVM Model:")
print("C:", best_params[0])
print("Kernel:", best_params[1])
print("Gamma:", best_params[2])
print("Best validation accuracy:", best_accuracy)

UNKNOWN_LABEL = 6
CONFIDENCE_THRESHOLD = 0.5

probs = best_model.predict_proba(X_eval)
max_probs = np.max(probs, axis=1)
pred_classes = np.argmax(probs, axis=1)

y_pred = np.where(
    max_probs < CONFIDENCE_THRESHOLD,
    UNKNOWN_LABEL,
    pred_classes
)

accuracy = accuracy_score(y_eval, y_pred)
print(f"Evaluation Accuracy with unknown: {accuracy:.4f}")

joblib.dump(best_model, "../models/svm_best.pkl")
joblib.dump(scaler, "../models/scaler_svm.pkl")

print("SVM best model saved")
