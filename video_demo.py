import cv2
import numpy as np
import joblib

# === Configuration ===
MODEL_PATH = "fall_detector_model.pkl"
VIDEO_PATH = "videoplayback.mp4"
FRAME_WIDTH, FRAME_HEIGHT = 128, 128
WINDOW_SIZE = 30
STEP = 2
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 320
save_output = True  # Set to True to save output video

# === Load trained model ===
model = joblib.load(MODEL_PATH)

def enhance_contrast(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def compute_mhi(frames, duration=30):
    h, w = frames[0].shape[:2]
    mhi = np.zeros((h, w), dtype=np.float32)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
        mhi = np.where(motion_mask == 1, duration, np.maximum(0, mhi - 1))
        prev_gray = gray

    mhi_normalized = np.uint8((mhi / duration) * 255)
    return mhi_normalized

# === Open video and prepare writer (optional) ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_buffer = []

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('fall_detection_output.avi', fourcc, 20.0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = enhance_contrast(frame)
    frame_buffer.append(frame)

    if len(frame_buffer) >= WINDOW_SIZE:
        window_frames = frame_buffer[-WINDOW_SIZE::STEP]
        mhi = compute_mhi(window_frames)
        feature = mhi.flatten().reshape(1, -1)
        predicted_label = model.predict(feature)[0]

        mhi_bgr = cv2.cvtColor(mhi, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mhi_bgr))

        label_text = "Fall Detected!" if predicted_label == 1 else "Normal Activity"
        color = (0, 0, 255) if predicted_label == 1 else (0, 255, 0)
        cv2.putText(combined, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Resize for display
        output_display = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    else:
        output_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("Fall Detection", output_display)

    if save_output:
        out.write(output_display)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
