"""
Standalone test: loads the saved GAN detector checkpoint directly and runs
it on one image, using the SAME preprocessing as the notebook's training
pipeline (tf.io.read_file + decode_jpeg), NOT app.py's cv2-based version.

This exists purely to check whether app.py's ported preprocessing logic
matches the original notebook logic, or whether there's a real bug in the
translation from notebook -> Flask backend.

Usage:
    python test_gan_model.py /path/to/your/test_image.jpg
"""

import sys
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "models/model_gan_best.keras"


def main(image_path):
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Exact same preprocessing as load_gan_image() in the notebook's Part 2
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # add batch dimension

    prob_fake = float(model.predict(img, verbose=0)[0][0])
    label = "Fake" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake if label == "Fake" else 1 - prob_fake

    print(f"\nImage: {image_path}")
    print(f"Raw prob_fake (sigmoid output): {prob_fake:.6f}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_gan_model.py /path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
