# DeepShield AI

Deepfake detection & media authentication system — Flask backend serving a
trained EfficientNetB0 face-swap detector, wired up to the DeepShield AI
frontend.

## Project structure

```
deepshield-ai/
├── app.py                        # Flask backend (serves the site + /predict API)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition for deployment
├── index.html                    # Frontend (upload UI, results display)
├── assets/
│   └── logo_deepshield_ai.png
└── models/
    └── model_3_best.keras        # Trained model checkpoint (NOT included — see below)
```

## Before running: add your model file

`models/model_3_best.keras` is **not included in this repo** (model
checkpoints are large binary files and shouldn't be committed casually to
Git). Download it from the Google Drive folder your notebook saves to
(`Deepfake_Project/models/model_3_best.keras`) and place it at
`models/model_3_best.keras` in this project before running or deploying.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.

## Deployment

See the full step-by-step deployment guide (GitHub + Hugging Face Spaces)
provided alongside this project.

## Notes

- The backend expects images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) or
  videos (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
- If model loading fails with a version-mismatch error, check what
  TensorFlow version your Colab notebook used to train/save the model
  (`import tensorflow as tf; print(tf.__version__)`) and update the pinned
  version in `requirements.txt` to match.
