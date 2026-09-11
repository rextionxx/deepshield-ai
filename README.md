# DeepShield AI

Deepfake Detection & Media Authentication System — a capstone project that
detects two different kinds of fake media: face-swap deepfakes (video and
images) and fully AI-generated faces (GAN-style images).

**Live demo:** https://deepshield-ai-359228671407.us-central1.run.app

**Team 8:** Mona Fahad (S2026_756) · Rayan Alzaabi (S2026_741) · Aqsa Khan (S2026_759)

---

## What it does

Upload a photo or video and the site tells you whether it looks real or
fake — and if it's fake, *what kind* of fake it thinks it found:

- **Face-swap deepfake** — someone's face swapped onto existing footage
  (the kind of manipulation in datasets like Celeb-DF)
- **Fully AI-generated image** — a face that was never a real photo to
  begin with (GAN/StyleGAN-style generation)

Two separate models run on every image upload, and their results are
combined into one verdict. Video is checked for face-swap manipulation
only (see Limitations in the notebook for why).

## How it was built

The full training pipeline, data cleaning, model comparison, and
evaluation is documented in `Deepfake_Detection___Media_Authentication_System.ipynb`.
Short version:

- **Face-swap detector:** EfficientNetB0 fine-tuned on Celeb-DF v2
  (1,203 videos, face-cropped frames). Three versions trained at 50/100/150
  epochs; best one picked by validation AUC. Test accuracy 92.75%, AUC 0.9872.
- **GAN-image detector:** same architecture, trained separately on the
  [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
  dataset (real FFHQ photos vs. StyleGAN-generated faces). Test accuracy
  99.60%, AUC 0.9997.

Full results, confusion matrices, and the literature review this project
is based on are in the notebook and project proposal.

## Project structure

```
deepshield-ai/
├── app.py                        # Flask backend — serves the site + /predict API
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition for deployment
├── index.html                    # Frontend (upload UI, results display)
├── assets/
│   └── logo_deepshield_ai.png
└── models/
    ├── model_3_best.keras        # Face-swap detector (NOT included — see below)
    └── model_gan_best.keras      # GAN-image detector (NOT included — see below)
```

## Before running: add your model files

Neither `.keras` file is committed to this repo — they're large binary
checkpoints and don't belong in Git history. Download both from the
`Deepfake_Project/models/` folder in Google Drive and place them at:

- `models/model_3_best.keras`
- `models/model_gan_best.keras`

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860`.

## Deployment

Deployed on **Google Cloud Run**, built straight from the `Dockerfile`:

```bash
gcloud run deploy deepshield-ai --source . --region us-central1 --allow-unauthenticated --memory 2Gi
```

We originally tried Hugging Face Spaces and Render before landing on
Cloud Run — Render's free tier only gives 512MB RAM, which isn't enough
for two loaded TensorFlow models, so Cloud Run's configurable memory
was the deciding factor.

## Known limitations

- Human faces only — not built for animals, objects, or other subjects
  (the face detector itself is trained specifically on human faces)
- Video is checked for face-swap manipulation only, not full AI generation
- The GAN detector was trained on StyleGAN output specifically, not
  diffusion models (Stable Diffusion, Midjourney) — a reasonable proxy
  for "fully synthetic face," not a guarantee against every generator
- Audio deepfakes are out of scope

Full discussion of limitations and future work is in the notebook's
conclusion section.
