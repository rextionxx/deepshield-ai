import os
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

# ---- Constants (must match what the model was trained with) ----
IMG_SIZE = 224
FACE_SIZE = 224
FACE_MARGIN = 0.2
FRAMES_PER_VIDEO = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_3_best.keras")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__, static_folder=None)

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Model loaded. Ready to serve predictions.")


# ---------------------------------------------------------------------------
# Shared prediction logic (same as predict_image / predict_video in the notebook)
# ---------------------------------------------------------------------------

def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    return img[y0:y1, x0:x1]


def predict_image_path(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read the uploaded image")

    crop = crop_face(img)
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    input_tensor = np.expand_dims(crop_rgb.astype("float32"), axis=0)
    prob_fake = float(model.predict(input_tensor, verbose=0)[0][0])

    label = "Fake" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake if label == "Fake" else 1 - prob_fake
    return label, confidence * 100.0


def predict_video_path(path, n_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read the uploaded video")

    n_frames = min(n_frames, total)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frame_probs = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        crop = crop_face(frame)
        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prob_fake = float(
            model.predict(
                np.expand_dims(crop_rgb.astype("float32"), axis=0), verbose=0
            )[0][0]
        )
        frame_probs.append(prob_fake)

    cap.release()
    if not frame_probs:
        raise ValueError("No faces could be detected in the uploaded video")

    avg_prob_fake = float(np.mean(frame_probs))
    label = "Fake" if avg_prob_fake >= 0.5 else "Real"
    confidence = avg_prob_fake if label == "Fake" else 1 - avg_prob_fake
    return label, confidence * 100.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if ext in VIDEO_EXTS:
            label, confidence = predict_video_path(tmp_path)
        elif ext in IMAGE_EXTS:
            label, confidence = predict_image_path(tmp_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        return jsonify({"label": label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
