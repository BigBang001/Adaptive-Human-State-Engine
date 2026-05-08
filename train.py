import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 1. Load Dataset
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'ckextended.csv')
if not os.path.exists(CSV_PATH):
    CSV_PATH = 'ckextended.csv'

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
print("Columns found:", list(df.columns))
print("Total rows:", len(df))

# 2. Preprocessing
def parse_pixels(pixel_str):
    try:
        arr = np.array(str(pixel_str).split(), dtype=np.uint8)
        if arr.size != 2304:
            return None
        return arr.reshape(48, 48)
    except:
        return None

emotion_col = df.columns[0]
images, labels = [], []

for _, row in df.iterrows():
    img = parse_pixels(row['pixels'])
    if img is not None:
        images.append(img)
        labels.append(int(row[emotion_col]))

images = np.array(images)
labels = np.array(labels)
print(f"Valid samples: {len(images)}")

if len(images) < 50:
    raise ValueError("Too few valid samples.")

# 3. HOG Feature Extraction
def extract_hog(img):
    return hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

print("Extracting HOG features...")
X = np.array([extract_hog(img) for img in images])
y = labels
print("Feature shape:", X.shape)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print("Classes:", np.unique(y_train))

# 5. Train SVM
print("Training SVM (RBF)...")
model = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Visualise sample
plt.figure(figsize=(4, 4))
plt.imshow(images[0], cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.axis('off')
plt.savefig("sample.png", bbox_inches='tight')
plt.close()
print("Sample saved → sample.png")

# 8. Save model
joblib.dump(model, "model.pkl")
print("Model saved → model.pkl")