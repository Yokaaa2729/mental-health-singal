# src/audio_preprocess.py
# Original feature-engineering pipeline, unchanged from the author's RAVDESS training code.
# Used in the demo to show the real MFCC/prosodic feature extraction, independent of
# which classifier (trained or pretrained) sits on top of it.

import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

EMO_MAP = {
    "01": "neutral",
    "02": "neutral",    # calm
    "03": "neutral",    # happy
    "04": "distressed",  # sad
    "05": "distressed",  # angry
    "06": "distressed",  # fearful
    "07": "distressed",  # disgust
    "08": "neutral",    # surprised
}

# Same grouping applied to the pretrained speech-emotion model's raw output labels,
# so the demo's "distressed / neutral" framing stays consistent with the original project.
SER_LABEL_TO_GROUP = {
    "angry": "distressed",
    "disgust": "distressed",
    "fear": "distressed",
    "fearful": "distressed",
    "sad": "distressed",
    "sadness": "distressed",
    "calm": "neutral",
    "neutral": "neutral",
    "happy": "neutral",
    "surprise": "neutral",
    "surprised": "neutral",
}


def extract_features(wav_path: str, sr_target: int = 16000, n_mfcc: int = 13) -> Tuple[dict, int]:
    y, sr = sf.read(wav_path, always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)  # mono
    if sr != sr_target:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat):
        return np.mean(mat, axis=1), np.std(mat, axis=1)

    mean_mfcc, std_mfcc = stats(mfcc)
    mean_d1, std_d1 = stats(mfcc_delta)
    mean_d2, std_d2 = stats(mfcc_delta2)

    # Simple prosodic features
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    feat = {
        "zcr": zcr,
        "rms": rms,
        "tempo": float(tempo),
        "duration_sec": len(y) / sr,
        "sr": sr,
    }
    for i, (m, s) in enumerate(zip(mean_mfcc, std_mfcc), start=1):
        feat[f"mfcc{i}_mean"] = float(m)
        feat[f"mfcc{i}_std"] = float(s)
    for i, (m, s) in enumerate(zip(mean_d1, std_d1), start=1):
        feat[f"dmfcc{i}_mean"] = float(m)
        feat[f"dmfcc{i}_std"] = float(s)
    for i, (m, s) in enumerate(zip(mean_d2, std_d2), start=1):
        feat[f"ddmfcc{i}_mean"] = float(m)
        feat[f"ddmfcc{i}_std"] = float(s)
    return feat, sr
