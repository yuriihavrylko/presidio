"""ORT CPUExecutionProvider vs CoreMLExecutionProvider benchmark.

Apple Silicon has no CUDA, so this is the closest available proxy for
"does a GPU/accelerator change the batching story" - CoreML routes
through the Mac's GPU/ANE instead of the CPU. Not equivalent to the
docs' NVIDIA CUDA/TensorRT scenarios, just the one real accelerator
path this machine has.

Compares, per execution provider: sequential analyze() vs
batch_analyze(batch_size=32), on the same 500 real dataset sentences
used in the earlier formal benchmark.

BUG FOUND: passing provider="CoreMLExecutionProvider" to the recognizer
has no effect - optimum's `pipeline()` factory calls ORTModel.to(device)
during pipeline construction, which re-derives an execution provider
from the torch-style device (cpu/cuda) and overwrites whatever provider
from_pretrained() was given. Verified: right after `rec.load()`, the
ORT session shows `['CPUExecutionProvider']` even though provider=
"CoreMLExecutionProvider" was passed - it silently ran on CPU the whole
time (same likely applies to OpenVINO/ROCm providers documented in
huggingface_ner_inference.md). This script works around it by forcing
the providers back after load(), purely to get a real measurement;
it is not a fix for the recognizer itself.

Usage: python benchmark_coreml.py <model_name>

Not part of the test suite - delete when done.
"""

import sys
import time

from benchmark_dataset import load_sentences
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "dslim/bert-base-NER"
BATCH_SIZE = 32

TEXTS = load_sentences(500)


def bench(provider: str):
    rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="ort",
        export=True,
        aggregation_strategy="simple",
        provider=provider,
        inference_batch_size=BATCH_SIZE,
    )
    rec.load()

    # Workaround for the provider-reset bug described above.
    session = rec.ner_pipeline.model.model
    if session.get_providers()[0] != provider:
        session.set_providers([provider, "CPUExecutionProvider"])
    assert session.get_providers()[0] == provider, session.get_providers()

    start = time.perf_counter()
    seq_results = [rec.analyze(t, []) for t in TEXTS]
    seq_elapsed = time.perf_counter() - start
    seq_entities = sum(len(r) for r in seq_results)

    start = time.perf_counter()
    batch_results = rec.batch_analyze(TEXTS, [], [None] * len(TEXTS))
    batch_elapsed = time.perf_counter() - start
    batch_entities = sum(len(r) for r in batch_results)

    print(
        f"{provider} sequential: {seq_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{seq_entities} entities"
    )
    print(
        f"{provider} batch={BATCH_SIZE}:  {batch_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{batch_entities} entities "
        f"(batching speedup: {seq_elapsed / batch_elapsed:.2f}x)"
    )


def main():
    print(f"Model: {MODEL_NAME}")
    bench("CPUExecutionProvider")
    bench("CoreMLExecutionProvider")


if __name__ == "__main__":
    main()
