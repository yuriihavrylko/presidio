"""Default (pre-improvement) HuggingFaceNerRecognizer benchmark.

Must be run with PYTHONPATH pointing at the baseline worktree
(main @ 0aa9f8e4, before the onnx/batch-inference/tokenizer-chunker
branches were merged), and benchmark_dataset.py copied alongside it
(it has no dependency on presidio_analyzer), e.g.:

    PYTHONPATH=/tmp/presidio-baseline/presidio-analyzer python benchmark_baseline.py <model_name>

Uses the default torch backend, sequential analyze(), and the default
CharacterBasedTextChunker - i.e. no explicit config, matching what a
caller got before any of the three feature branches landed.

Not part of the test suite - delete when done.
"""

import sys
import time

from benchmark_dataset import load_long_text, load_sentences
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "dslim/bert-base-NER"

TEXTS = load_sentences(500)
LONG_TEXT = load_long_text(200)


def main():
    import presidio_analyzer

    print(f"Model: {MODEL_NAME}")
    print(f"Using presidio_analyzer from: {presidio_analyzer.__file__}")

    rec = HuggingFaceNerRecognizer(model_name=MODEL_NAME, aggregation_strategy="simple")
    rec.load()

    start = time.perf_counter()
    results = [rec.analyze(t, []) for t in TEXTS]
    elapsed = time.perf_counter() - start
    total_entities = sum(len(r) for r in results)
    print(
        f"BASELINE (torch, sequential, char-chunker): {elapsed:.3f}s "
        f"for {len(TEXTS)} texts, {total_entities} entities"
    )

    start = time.perf_counter()
    long_results = rec.analyze(LONG_TEXT, [])
    elapsed_long = time.perf_counter() - start
    print(
        f"BASELINE long text ({len(LONG_TEXT)} chars): {elapsed_long:.3f}s, "
        f"{len(long_results)} entities"
    )


if __name__ == "__main__":
    main()
