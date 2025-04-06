import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# CONFIG
# -----------------------------
FALL_DIR = 'datasets'
NON_FALL_DIR = 'nfdatasets'
MHI_DURATION = 30
FRAME_STEP = 2  # To skip frames and reduce computation

# -----------------------------
# FUNCTIONS
# -----------------------------

def load_frame_paths(frame_folder):
    files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder)
                    if f.lower().endswith(('.jpg', '.png'))])
    return files[::FRAME_STEP]  # Step frames

def enhance_contrast(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def compute_mhi_from_folder(frame_folder, duration=MHI_DURATION):
    frame_paths = load_frame_paths(frame_folder)
    if len(frame_paths) < 2:
        return None

    frames = [cv2.resize(cv2.imread(p), (128, 128)) for p in frame_paths]
    frames = [enhance_contrast(f) for f in frames]

    h, w = frames[0].shape[:2]
    mhi = np.zeros((h, w), dtype=np.float32)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)

        # Update MHI: decay old motion, boost new motion
        mhi = np.where(motion_mask == 1, duration, np.maximum(0, mhi - 1))
        prev_gray = gray

    # Normalize MHI to 0–255
    mhi_normalized = np.uint8((mhi / duration) * 255)
    return mhi_normalized

def extract_features_and_labels():
    X, y = [], []

    # Fall
    for folder in os.listdir(FALL_DIR):
        path = os.path.join(FALL_DIR, folder)
        if os.path.isdir(path):
            mhi = compute_mhi_from_folder(path)
            if mhi is not None:
                X.append(mhi.flatten())
                y.append(1)

    # Non-Fall
    for folder in os.listdir(NON_FALL_DIR):
        path = os.path.join(NON_FALL_DIR, folder)
        if os.path.isdir(path):
            mhi = compute_mhi_from_folder(path)
            if mhi is not None:
                X.append(mhi.flatten())
                y.append(0)

    return np.array(X), np.array(y)

# -----------------------------
# MAIN
# -----------------------------
print("Extracting features...")
X, y = extract_features_and_labels()

print(f"Total samples: {len(X)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "k-NN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

print("\nTraining and evaluating models...\n")
for name, model in models.items():
    print(f"--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


import joblib

# Save the best performing model (Random Forest)
joblib.dump(models["Random Forest"], "fall_detector_model.pkl")


