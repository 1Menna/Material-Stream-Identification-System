import cv2
import numpy as np
from skimage.feature import hog
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import os

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_cnn_features(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    features = base_model.predict(img, verbose=0)
    return features.flatten()

classes = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash"
]

X_features = []
y_labels = []

dataset_path = "D:\Coding things\Material-Stream-Identification-System-master\dataset"
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if class_name not in classes:
        continue
    label = classes.index(class_name)
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        try:
            feature_vector = extract_cnn_features(img_path)
            X_features.append(feature_vector)
            y_labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

X_features = np.array(X_features, dtype=np.float32)
y_labels = np.array(y_labels)


np.save("../features/X_features.npy", X_features)
np.save("../features/y_labels.npy", y_labels)

print("Done! Features and labels saved.")
print("X_features shape:", X_features.shape)
print("y_labels shape:", y_labels.shape)

print("Min value:", X_features.min())
print("Max value:", X_features.max())

print("Min label:", y_labels.min())
print("Max label:", y_labels.max())

classes = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash"
]

X_features = []
y_labels = []

dataset_path = r"D:\Coding things\Material-Stream-Identification-System-master\test_data"
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if class_name not in classes:
        continue
    label = classes.index(class_name)
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        try:
            feature_vector = extract_cnn_features(img_path)
            X_features.append(feature_vector)
            y_labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

X_features = np.array(X_features, dtype=np.float32)
y_labels = np.array(y_labels)


np.save("../features/X_test.npy", X_features)
np.save("../features/y_test.npy", y_labels)

print("Done! Features and labels saved.")
print("X_test shape:", X_features.shape)
print("y_test shape:", y_labels.shape)

UNKNOWN_LABEL = 6

X_unknown = []
y_unknown = []

unknown_path = r"D:\Coding things\Material-Stream-Identification-System-master\unknown_images"

for img_file in os.listdir(unknown_path):
    img_path = os.path.join(unknown_path, img_file)

    try:
        feature_vector = extract_cnn_features(img_path)
        X_unknown.append(feature_vector)
        y_unknown.append(UNKNOWN_LABEL)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
X_unknown = np.array(X_unknown, dtype=np.float32)
y_unknown =np.array(y_unknown)
np.save("../features/X_unknown.npy", X_unknown)
np.save("../features/y_unknown.npy", y_unknown)

print("Done! Unknown features and labels saved.")
print("X_unknown shape:", X_unknown.shape)
print("y_unknown shape:", y_unknown.shape)
