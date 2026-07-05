"""Throwaway benchmark for the three merged HuggingFaceNerRecognizer features:

1. Backend: torch vs ort (CPU and CoreML execution provider)
2. Batch inference: sequential analyze() vs batch_analyze() with inference_batch_size
3. Text chunking: CharacterBasedTextChunker vs TokenizerBasedTextChunker

Run with:
    python benchmark_speed.py

Not part of the test suite - delete when done.
"""

import statistics
import time

from presidio_analyzer.chunkers import (
    CharacterBasedTextChunker,
    TokenizerBasedTextChunker,
)
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

MODEL_NAME = "dslim/bert-base-NER"

SHORT_TEXTS = [
    "Dr. Sarah Chen treated the patient on Monday in New York.",
    "John Smith works at Microsoft in Seattle.",
    "Please contact Maria Garcia regarding the Berlin office.",
    "Ahmed Al-Farsi will visit the Tokyo branch next week.",
    "Emily Johnson from Google called about the London contract.",
] * 100  # 500 short texts

LONG_TEXT = (
    "Dr. Sarah Chen treated the patient on Monday in New York. "
    "John Smith works at Microsoft in Seattle. "
    "Please contact Maria Garcia regarding the Berlin office. "
) * 80  # ~ a few thousand characters, forces chunking


def timed(fn, repeats=3):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def bench_backend():
    print("\n=== 1. Backend: torch vs ort ===")

    torch_rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="torch",
        aggregation_strategy="simple",
    )
    torch_rec.load()
    median, _ = timed(lambda: [torch_rec.analyze(t, []) for t in SHORT_TEXTS], repeats=1)
    print(f"torch (CPU):            {median:.3f}s for {len(SHORT_TEXTS)} texts")

    ort_cpu_rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="ort",
        aggregation_strategy="simple",
        export=True,  # convert the torch checkpoint to ONNX on the fly
    )
    ort_cpu_rec.load()
    median, _ = timed(lambda: [ort_cpu_rec.analyze(t, []) for t in SHORT_TEXTS])
    print(f"ort (CPUExecutionProvider): {median:.3f}s for {len(SHORT_TEXTS)} texts")

    try:
        ort_coreml_rec = HuggingFaceNerRecognizer(
            model_name=MODEL_NAME,
            backend="ort",
            aggregation_strategy="simple",
            export=True,
            provider="CoreMLExecutionProvider",
        )
        ort_coreml_rec.load()
        median, _ = timed(lambda: [ort_coreml_rec.analyze(t, []) for t in SHORT_TEXTS])
        print(f"ort (CoreMLExecutionProvider): {median:.3f}s for {len(SHORT_TEXTS)} texts")
    except Exception as e:
        print(f"ort (CoreMLExecutionProvider): skipped ({e})")


def bench_batch_inference():
    print("\n=== 2. Batch inference: sequential vs batched ===")

    rec_seq = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="torch",
        aggregation_strategy="simple",
        inference_batch_size=1,
    )
    rec_seq.load()
    median, _ = timed(lambda: [rec_seq.analyze(t, []) for t in SHORT_TEXTS], repeats=1)
    print(f"sequential analyze():         {median:.3f}s for {len(SHORT_TEXTS)} texts")

    for batch_size in (8, 32):
        rec_batch = HuggingFaceNerRecognizer(
            model_name=MODEL_NAME,
            backend="torch",
            aggregation_strategy="simple",
            inference_batch_size=batch_size,
        )
        rec_batch.load()
        median, _ = timed(
            lambda: rec_batch.batch_analyze(
                SHORT_TEXTS, [], [None] * len(SHORT_TEXTS)
            )
        )
        print(
            f"batch_analyze(batch_size={batch_size}): "
            f"{median:.3f}s for {len(SHORT_TEXTS)} texts"
        )


def bench_chunker():
    print("\n=== 3. Chunking: character-based vs tokenizer-based ===")

    char_rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="torch",
        aggregation_strategy="simple",
        text_chunker=CharacterBasedTextChunker(chunk_size=400, chunk_overlap=20),
    )
    char_rec.load()
    median, char_results = timed(lambda: char_rec.analyze(LONG_TEXT, []), repeats=3)
    print(
        f"CharacterBasedTextChunker: {median:.3f}s, "
        f"{len(char_results)} entities on {len(LONG_TEXT)} chars"
    )

    tok_rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="torch",
        aggregation_strategy="simple",
        text_chunker=TokenizerBasedTextChunker(max_tokens=200, overlap_tokens=20),
    )
    tok_rec.load()
    median, tok_results = timed(lambda: tok_rec.analyze(LONG_TEXT, []), repeats=3)
    print(
        f"TokenizerBasedTextChunker: {median:.3f}s, "
        f"{len(tok_results)} entities on {len(LONG_TEXT)} chars"
    )


if __name__ == "__main__":
    bench_backend()
    bench_batch_inference()
    bench_chunker()
