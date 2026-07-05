"""Shared dataset loader for the formal HuggingFaceNerRecognizer benchmarks.

Pulls real, varied sentences from ai4privacy/pii-masking-200k (English
subset) instead of repeating one hardcoded string - repeating identical
text skews chunking/batching timing versus realistic mixed-length input.

Not part of the test suite - delete when done.
"""

import json

from huggingface_hub import hf_hub_download


def load_sentences(n: int) -> list:
    """Return the first n `source_text` sentences from the dataset."""
    path = hf_hub_download(
        repo_id="ai4privacy/pii-masking-200k",
        filename="english_pii_43k.jsonl",
        repo_type="dataset",
    )
    sentences = []
    with open(path) as f:
        for line in f:
            if len(sentences) >= n:
                break
            row = json.loads(line)
            text = row["source_text"].strip()
            if text:
                sentences.append(text)
    return sentences


def load_long_text(num_sentences: int = 200) -> str:
    """Join real sentences into one long text to exercise chunking."""
    sentences = load_sentences(num_sentences)
    return " ".join(sentences)
