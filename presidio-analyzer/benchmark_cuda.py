"""GPU-only benchmark: torch CUDA and ort CUDAExecutionProvider.

Run this on a machine with an NVIDIA GPU (e.g. a Colab GPU runtime).
Uses the same 500 real dataset sentences as the CPU/CoreML benchmarks
so the numbers are directly comparable.

Applies the same provider-reset workaround as benchmark_coreml.py:
optimum's pipeline() construction calls ORTModel.to(device), which
re-derives an execution provider from the torch-style device and
overwrites whatever provider from_pretrained() was given - so
provider="CUDAExecutionProvider" silently no-ops back to CPU unless
forced back after load(). See benchmark_coreml.py's docstring for the
full mechanism.

The ort(CUDAExecutionProvider) run is wrapped in a try/except: on
Colab this has been blocked by a cuDNN ABI conflict between the
system-wide cuDNN (linked by the preinstalled TensorFlow) and the one
onnxruntime-gpu needs - an environment/packaging issue, not a
presidio bug. If it fails, the script reports it and still gives you
the torch(cuda) numbers.

Usage: python benchmark_cuda.py <model_name>

Not part of the test suite - delete when done.
"""

import sys
import time

import torch
from benchmark_dataset import load_sentences
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "dslim/bert-base-NER"
BATCH_SIZE = 32

TEXTS = load_sentences(500)


def bench_torch(device: str):
    rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="torch",
        aggregation_strategy="simple",
        device=device,
        inference_batch_size=BATCH_SIZE,
    )
    rec.load()

    start = time.perf_counter()
    seq_results = [rec.analyze(t, []) for t in TEXTS]
    seq_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    batch_results = rec.batch_analyze(TEXTS, [], [None] * len(TEXTS))
    batch_elapsed = time.perf_counter() - start

    print(
        f"torch({device}) sequential: {seq_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{sum(len(r) for r in seq_results)} entities"
    )
    print(
        f"torch({device}) batch={BATCH_SIZE}:  {batch_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{sum(len(r) for r in batch_results)} entities "
        f"(batching speedup: {seq_elapsed / batch_elapsed:.2f}x)"
    )


def bench_ort(provider: str):
    rec = HuggingFaceNerRecognizer(
        model_name=MODEL_NAME,
        backend="ort",
        export=True,
        aggregation_strategy="simple",
        provider=provider,
        inference_batch_size=BATCH_SIZE,
    )
    rec.load()

    # Workaround for the provider-reset bug described in the module docstring.
    session = rec.ner_pipeline.model.model
    if session.get_providers()[0] != provider:
        session.set_providers([provider, "CPUExecutionProvider"])
    assert session.get_providers()[0] == provider, session.get_providers()

    start = time.perf_counter()
    seq_results = [rec.analyze(t, []) for t in TEXTS]
    seq_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    batch_results = rec.batch_analyze(TEXTS, [], [None] * len(TEXTS))
    batch_elapsed = time.perf_counter() - start

    print(
        f"ort({provider}) sequential: {seq_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{sum(len(r) for r in seq_results)} entities"
    )
    print(
        f"ort({provider}) batch={BATCH_SIZE}:  {batch_elapsed:.3f}s for {len(TEXTS)} texts, "
        f"{sum(len(r) for r in batch_results)} entities "
        f"(batching speedup: {seq_elapsed / batch_elapsed:.2f}x)"
    )


def main():
    print(f"Model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device visible - check the Colab runtime type (Runtime > Change runtime type > GPU).")
        return

    bench_torch("cuda")

    try:
        bench_ort("CUDAExecutionProvider")
    except Exception as e:
        print(f"ort(CUDAExecutionProvider): FAILED - {e}")


if __name__ == "__main__":
    main()
