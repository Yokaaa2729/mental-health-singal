# src/text_preprocess.py
# Original text-cleaning function, unchanged from the author's TF-IDF training pipeline.

import re


def basic_clean(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove urls
    text = re.sub(r"[^a-z0-9' ]+", " ", text)       # keep simple chars
    text = re.sub(r"\s+", " ", text).strip()
    return text
