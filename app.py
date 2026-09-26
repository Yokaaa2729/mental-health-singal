"""
Mental Health Signal Demo — NLP + Speech
-----------------------------------------
Portfolio demo built on an original feature-engineering pipeline (MFCC/prosodic
audio features, TF-IDF-style text cleaning) developed for an MSc AI & Robotics
project on detecting early emotional-distress markers from text and speech.

Status note (read this before you read the code):
The classification *heads* in this demo are public pretrained models, not the
author's own trained weights — the original training corpora (RAVDESS audio,
Reddit text) were not available for retraining in this environment. The
feature-extraction code (audio_preprocess.py, text_preprocess.py) is the
author's original, unmodified implementation and is shown in the UI to make
that split transparent. See README.md for the full explanation.

This is a research/portfolio demo, not a diagnostic or clinical tool.
"""

import os
import tempfile

import gradio as gr
from transformers import pipeline

from src.audio_preprocess import extract_features, SER_LABEL_TO_GROUP
from src.text_preprocess import basic_clean

# ---------------------------------------------------------------------------
# Lazy-loaded pretrained models (downloaded on first call, then cached)
# ---------------------------------------------------------------------------
_text_pipe = None
_audio_pipe = None


def get_text_pipe():
    global _text_pipe
    if _text_pipe is None:
        _text_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
    return _text_pipe


def get_audio_pipe():
    global _audio_pipe
    if _audio_pipe is None:
        _audio_pipe = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        )
    return _audio_pipe


DISTRESS_EMOTIONS = {"sadness", "fear", "anger", "disgust", "sad", "angry", "fearful"}

CRISIS_NOTE = (
    "This tool flags emotional tone in text and speech using general-purpose "
    "pretrained models. It is **not** a diagnostic or clinical instrument and "
    "should not be used to assess real risk. If you or someone you know is "
    "struggling, please reach out to a mental health professional, or a crisis "
    "line such as Samaritans (UK: 116 123) or 988 Suicide & Crisis Lifeline (US)."
)


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------
def analyze_text(raw_text: str):
    if not raw_text or not raw_text.strip():
        return "Enter some text above to analyse.", {}, ""

    cleaned = basic_clean(raw_text)
    pipe = get_text_pipe()
    scores = pipe(cleaned)[0]  # list of {"label":..., "score":...}
    scores_dict = {s["label"]: round(float(s["score"]), 4) for s in scores}

    distress_mass = sum(v for k, v in scores_dict.items() if k in DISTRESS_EMOTIONS)
    top_emotion = max(scores_dict, key=scores_dict.get)

    if distress_mass >= 0.5:
        verdict = f"⚠️ Elevated distress signal (combined negative-emotion score: {distress_mass:.2f})"
    else:
        verdict = f"No strong distress signal detected (combined negative-emotion score: {distress_mass:.2f})"

    pipeline_note = (
        f"**Cleaned text (original `basic_clean`):** `{cleaned}`\n\n"
        f"**Top predicted emotion:** {top_emotion}"
    )

    return verdict, scores_dict, pipeline_note


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------
def analyze_audio(audio_path: str):
    if not audio_path:
        return "Upload or record audio above to analyse.", {}, ""

    # 1) Run the ORIGINAL feature-extraction pipeline (MFCCs, deltas, prosody)
    try:
        feats, sr = extract_features(audio_path)
    except Exception as e:
        return f"Feature extraction failed: {e}", {}, ""

    feat_summary = (
        f"**Original feature pipeline output** (sample rate {feats['sr']} Hz, "
        f"duration {feats['duration_sec']:.2f}s)\n\n"
        f"- Zero-crossing rate: {feats['zcr']:.4f}\n"
        f"- RMS energy: {feats['rms']:.4f}\n"
        f"- Tempo: {feats['tempo']:.1f} BPM\n"
        f"- MFCC-1 mean / std: {feats['mfcc1_mean']:.2f} / {feats['mfcc1_std']:.2f}\n"
        f"- 13 MFCCs + delta + delta-delta extracted ({sum(1 for k in feats if 'mfcc' in k)} values total)"
    )

    # 2) Run the pretrained speech-emotion classifier for the actual prediction
    pipe = get_audio_pipe()
    preds = pipe(audio_path)  # list of {"label":..., "score":...}
    scores_dict = {p["label"]: round(float(p["score"]), 4) for p in preds}

    distress_mass = sum(
        v for k, v in scores_dict.items() if SER_LABEL_TO_GROUP.get(k.lower(), "neutral") == "distressed"
    )
    top_label = max(scores_dict, key=scores_dict.get)

    if distress_mass >= 0.5:
        verdict = f"⚠️ Elevated distress signal (combined score: {distress_mass:.2f}, top: {top_label})"
    else:
        verdict = f"No strong distress signal detected (combined score: {distress_mass:.2f}, top: {top_label})"

    return verdict, scores_dict, feat_summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Mental Health Signal Demo — NLP + Speech") as demo:
    gr.Markdown(
        "# Mental Health Signal Demo — NLP + Speech\n"
        "Multimodal distress-signal detection built on an original feature-engineering "
        "pipeline (MFCC/prosodic audio features + text preprocessing) from an MSc AI & "
        "Robotics project, with pretrained models providing the classification layer. "
        "See the **About** tab for the full breakdown of what's original vs. pretrained."
    )
    gr.Markdown(f"> {CRISIS_NOTE}")

    with gr.Tab("Text"):
        text_in = gr.Textbox(
            label="Enter text",
            placeholder="Type or paste a sentence or short passage...",
            lines=4,
        )
        text_btn = gr.Button("Analyse text", variant="primary")
        text_verdict = gr.Markdown()
        text_scores = gr.Label(label="Emotion probabilities")
        text_pipeline_note = gr.Markdown()
        text_btn.click(
            analyze_text,
            inputs=text_in,
            outputs=[text_verdict, text_scores, text_pipeline_note],
        )

    with gr.Tab("Speech"):
        audio_in = gr.Audio(label="Upload or record audio", type="filepath")
        audio_btn = gr.Button("Analyse speech", variant="primary")
        audio_verdict = gr.Markdown()
        audio_scores = gr.Label(label="Emotion probabilities")
        audio_feat_note = gr.Markdown()
        audio_btn.click(
            analyze_audio,
            inputs=audio_in,
            outputs=[audio_verdict, audio_scores, audio_feat_note],
        )

    with gr.Tab("About"):
        gr.Markdown(
            "## What's original vs. pretrained\n"
            "**Original (author's own code, unmodified):**\n"
            "- `src/audio_preprocess.py` — MFCC + delta/delta-delta extraction, "
            "zero-crossing rate, RMS energy, tempo, silence trimming\n"
            "- `src/text_preprocess.py` — text cleaning used ahead of the original "
            "TF-IDF training pipeline\n"
            "- The RAVDESS `EMO_MAP` distress/neutral grouping, reused here to keep "
            "the framing consistent with the original project\n\n"
            "**Pretrained (public Hugging Face models, classification layer only):**\n"
            "- Text emotion: `j-hartmann/emotion-english-distilroberta-base`\n"
            "- Speech emotion: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` "
            "(trained on RAVDESS among other corpora)\n\n"
            "**Why the split:** the original trained model weights (TF-IDF + SVM "
            "text classifier trained on a Reddit dataset; MFCC + SVM audio classifier "
            "trained on RAVDESS) required corpora not available in the environment "
            "this demo was rebuilt in. Rather than fabricate results, the real "
            "feature-engineering code is kept and shown, and pretrained models fill "
            "the classification role until the original models are retrained and "
            "swapped in.\n\n"
            f"{CRISIS_NOTE}"
        )

if __name__ == "__main__":
    # Render (and most free PaaS hosts) assign the port via $PORT and require
    # binding to 0.0.0.0. Falls back to Gradio's normal defaults if unset,
    # so this still works unchanged if you deploy to HF Spaces later.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
