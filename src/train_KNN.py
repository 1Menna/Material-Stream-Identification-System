import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import os
os.makedirs("models", exist_ok=True)

X_train = np.load("../features/X_features.npy")
y_train = np.load("../features/y_labels.npy")
print("train features: ", X_train.shape)
print("train labels: ", y_train.shape)

X_test = np.load("../features/X_test.npy")
y_test = np.load("../features/y_test.npy")

print ("Test features:", X_test.shape)
print ("Test labels:", y_test.shape)

X_unknown = np.load("../features/X_unknown.npy")
y_unknown = np.load("../features/y_unknown.npy")

print ("Unknown features:", X_unknown.shape)
print ("Unknown labels:", y_unknown.shape)

X_eval = np.vstack((X_test, X_unknown))
y_evel = np.concatenate([y_test, y_unknown])


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)
print("Feature scaling done")

k_values = [1,3,5,7,9,11,13,15,17]
weighting_schemes = ["uniform", "distance"]
best_accuracy = 0
best_k = None
best_weight = None
best_model = None

for k in k_values:
    for weight in weighting_schemes:
        print(f"Training k-NN with k={k}, weights={weight}")
        knn = KNeighborsClassifier(n_neighbors=k,weights=weight,metric="cosine")

        knn.fit(X_train,y_train)

        y_val = knn.predict(X_test)
        acc = accuracy_score(y_test, y_val)
        print(f"Validation Accuracy: {acc:.4f}")

        if acc >best_accuracy:
            best_accuracy = acc
            best_k = k
            best_weight = weight
            best_model = knn
print("\nBest k-NN Model:")
print("Best k:", best_k)
print("Best weighting:", best_weight)
print("Best validation accuracy:", best_accuracy)

Unknown_label = 6
DISTANCE_THRESHOLD = 2.0
VOTE_THRESHOLD = 0.5

distances, neighbors = knn.kneighbors(X_eval)
neighbors_labels = y_train[neighbors]

def predict_with_threshold(distances, neighbor_labels):
    predictions = []

    for i in range(len(distances)):
        mean_distance = np.mean(distances[i])
        votes = Counter(neighbor_labels[i])
        t_class, t_count = votes.most_common(1)[0]
        v_ratio = t_count / len(neighbor_labels[i])
        if mean_distance > DISTANCE_THRESHOLD or v_ratio < VOTE_THRESHOLD:
            predictions.append(Unknown_label)
        else:
            predictions.append(t_class)
    return np.array(predictions)

y_pred = predict_with_threshold(distances, neighbors_labels)

accuracy = accuracy_score(y_evel, y_pred)
print(f"Evaluation Accuracy with unknown: {accuracy:.4f}")

joblib.dump(knn,"../models/knn_best.pkl")
joblib.dump(scaler,"../models/scaler_knn.pkl")

print("KNN best model saved")
