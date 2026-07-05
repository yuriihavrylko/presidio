"""All-improvements-combined HuggingFaceNerRecognizer benchmark.

Run from the current repo (no PYTHONPATH override) to pair with
benchmark_baseline.py. Uses the ort backend, batch inference
(batch_size=32), and TokenizerBasedTextChunker together - i.e. every
knob added by the three merged feature branches, set to its fastest
setting from the earlier per-feature runs.

Uses the exact same TEXTS/LONG_TEXT as benchmark_baseline.py (both
loaders pull the same first-N rows from the same dataset file) so the
two numbers are directly comparable.

Usage: python benchmark_combined.py <model_name> [provider]
  provider defaults to CPUExecutionProvider; pass CUDAExecutionProvider
  on a GPU machine to get the GPU version of "all improvements".
  Applies the same provider-reset workaround as benchmark_coreml.py /
  benchmark_cuda.py (optimum's pipeline() construction overwrites
  whatever provider from_pretrained() was given).

Not part of the test suite - delete when done.
"""

import sys
import time

from benchmark_dataset import load_long_text, load_sentences
from presidio_analyzer.chunkers import TokenizerBasedTextChunker
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "dslim/bert-base-NER"
PROVIDER = sys.argv[2] if len(sys.argv) > 2 else "CPUExecutionProvider"

TEXTS = load_sentences(500)
LONG_TEXT = load_long_text(200)


def main():
    import presidio_analyzer

    print(f"Model: {MODEL_NAME}")
    print(f"Provider: {PROVIDER}")
    print(f"Using presidio_analyzer from: {presidio_analyzer.__file__}")

    rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="ort",
        export=True,
        aggregation_strategy="simple",
        provider=PROVIDER,
        inference_batch_size=32,
        text_chunker=TokenizerBasedTextChunker(max_tokens=200, overlap_tokens=20),
    )
    rec.load()

    # Workaround for the provider-reset bug (see benchmark_coreml.py docstring).
    session = rec.ner_pipeline.model.model
    if session.get_providers()[0] != PROVIDER:
        session.set_providers([PROVIDER, "CPUExecutionProvider"])
    assert session.get_providers()[0] == PROVIDER, session.get_providers()

    start = time.perf_counter()
    results = rec.batch_analyze(TEXTS, [], [None] * len(TEXTS))
    elapsed = time.perf_counter() - start
    total_entities = sum(len(r) for r in results)
    print(
        f"ALL IMPROVEMENTS (ort + batch=32 + tokenizer-chunker): {elapsed:.3f}s "
        f"for {len(TEXTS)} texts, {total_entities} entities"
    )

    start = time.perf_counter()
    long_results = rec.analyze(LONG_TEXT, [])
    elapsed_long = time.perf_counter() - start
    print(
        f"ALL IMPROVEMENTS long text ({len(LONG_TEXT)} chars): {elapsed_long:.3f}s, "
        f"{len(long_results)} entities"
    )


if __name__ == "__main__":
    main()
