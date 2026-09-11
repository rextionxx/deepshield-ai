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
MODEL_GAN_PATH = os.path.join(BASE_DIR, "models", "model_gan_best.keras")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__, static_folder=None)

print("Loading face-swap model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading GAN/diffusion-image model from:", MODEL_GAN_PATH)
gan_model = tf.keras.models.load_model(MODEL_GAN_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Both models loaded. Ready to serve predictions.")


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

    # --- Face-swap model: uses the face-crop-with-margin pipeline ---
    crop = crop_face(img)
    faceswap_input = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    faceswap_input_rgb = cv2.cvtColor(faceswap_input, cv2.COLOR_BGR2RGB)
    faceswap_prob = float(
        model.predict(
            np.expand_dims(faceswap_input_rgb.astype("float32"), axis=0),
            verbose=0,
        )[0][0]
    )

    # --- GAN model: uses a plain resize of the whole image (its own training pipeline) ---
    gan_input = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gan_input_rgb = cv2.cvtColor(gan_input, cv2.COLOR_BGR2RGB)
    gan_prob = float(
        gan_model.predict(
            np.expand_dims(gan_input_rgb.astype("float32"), axis=0),
            verbose=0,
        )[0][0]
    )

    faceswap_flag = faceswap_prob >= 0.5
    gan_flag = gan_prob >= 0.5

    if faceswap_flag and gan_flag:
        label = "Fake"
        confidence = max(faceswap_prob, gan_prob)
        detail = "Signals of both face-swap manipulation and full AI generation detected."
    elif faceswap_flag:
        label = "Fake"
        confidence = faceswap_prob
        detail = "Likely a face-swap deepfake."
    elif gan_flag:
        label = "Fake"
        confidence = gan_prob
        detail = "Likely a fully AI-generated (GAN-style) image."
    else:
        label = "Real"
        # confidence in "real" = how strongly BOTH models agree it's real
        confidence = 1 - max(faceswap_prob, gan_prob)
        detail = "No face-swap or full-generation signals detected — likely authentic."

    return label, confidence * 100.0, detail


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
    detail = (
        "Likely a face-swap deepfake (video analysis covers face-swap only)."
        if label == "Fake"
        else "No face-swap signals detected across sampled frames — likely authentic."
    )
    return label, confidence * 100.0, detail


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
            label, confidence, detail = predict_video_path(tmp_path)
        elif ext in IMAGE_EXTS:
            label, confidence, detail = predict_image_path(tmp_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        return jsonify({"label": label, "confidence": confidence, "detail": detail})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
