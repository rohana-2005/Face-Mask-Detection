import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.ensemble import RandomForestClassifier

# Consistent category order
categories = ['with_mask', 'without_mask']
data_dir = "dataset"

data = []
labels = []

print("Loading and processing images...")

# Load and label images
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten())
            labels.append(label)

print(f"Total images: {len(data)}")

# Convert and scale
X = np.array(data)
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save model and scaler
joblib.dump(model, "face_mask_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
