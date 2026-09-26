---
title: Mental Health Signal Demo (NLP + Speech)
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Mental Health Signal Demo — NLP + Speech

A multimodal demo detecting emotional-distress signals from **text** and
**speech**, built from an MSc AI & Robotics project on early markers of
anxiety/depression in user-generated content.

> This is a research/portfolio demo, not a diagnostic or clinical tool. If you
> or someone you know is struggling, please contact a mental health
> professional or a crisis line (UK: Samaritans 116 123; US: 988).

## What this demonstrates

- **Original feature-engineering pipeline** (`src/audio_preprocess.py`,
  `src/text_preprocess.py`): MFCC + delta/delta-delta extraction, prosodic
  features (ZCR, RMS, tempo), silence trimming, and text cleaning — all
  written for the original training pipeline and unchanged here.
- **Multimodal framing**: a shared distressed/neutral grouping (derived from
  the original RAVDESS `EMO_MAP`) applied consistently across both the text
  and speech classifiers.
- **Honest labelling of provenance**: the classification layer currently uses
  public pretrained models (see below), not the author's own trained weights,
  because the original training corpora (RAVDESS audio, a Reddit mental-health
  text dataset) weren't available for retraining in the environment this demo
  was rebuilt in. The **About** tab in the app states this explicitly.

## Models used

| Component | Source | Status |
|---|---|---|
| Audio feature extraction (MFCC, deltas, prosody) | Original code | Author's own, unmodified |
| Text cleaning | Original code | Author's own, unmodified |
| Text emotion classification | `j-hartmann/emotion-english-distilroberta-base` | Pretrained (public) |
| Speech emotion classification | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Pretrained (public, trained on RAVDESS among other corpora) |

## Roadmap

- [ ] Retrain the original TF-IDF + SVM text classifier on the full Reddit
      mental-health corpus and swap it in behind the same `predict_text.py`
      interface already written for it.
- [ ] Retrain the original MFCC + SVM audio classifier on the full RAVDESS
      corpus (all 8 emotion classes, all actors) via `audio_split_features.py`
      / `predict_audio.py`, and swap it in.
- [ ] Add the late-fusion model combining text + speech signals described in
      the original project write-up.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploying to Render (free, no card required)

Hugging Face Spaces now requires a PRO subscription to run Gradio apps on
compute. Render's free web service tier doesn't, so that's the default path
here — `render.yaml` is already set up for it.

1. Push this folder to a GitHub repo (public or private).
2. Go to render.com, sign up free, click **New > Blueprint**, and point it at
   the repo. Render reads `render.yaml` automatically and configures the
   service (Python, free plan, correct start command).
3. Click **Apply** / **Create Web Service**. First build takes a few minutes
   while it installs `requirements.txt` and downloads the two pretrained
   models on first use.
4. Once live, your URL looks like `https://mental-health-signal-demo.onrender.com`.

Note: the free tier spins down after ~15 minutes of inactivity. The first
request after that takes 30-60 seconds to wake up — fine for a portfolio
link, so open it a few minutes early if you're demoing it live in an
interview.

## Deploying to Hugging Face Spaces (needs PRO, $9/month)

1. Create a new Space at huggingface.co/new-space, SDK: **Gradio**, hardware:
   **CPU basic**. This step now requires a PRO subscription.
2. Upload `app.py`, `requirements.txt`, this `README.md`, and the `src/` folder.
3. The Space installs dependencies and starts automatically — first load will
   be slower while the two pretrained models download and cache.
