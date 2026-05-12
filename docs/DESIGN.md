# Design

## Goal

Show AI/ML depth without a large codebase:

1. Generate a repeatable signal dataset.
2. Train a small PyTorch classifier.
3. Export to ONNX.
4. Quantize with ONNX Runtime.
5. Benchmark float32 vs int8.

## Dataset

Synthetic 1D sensor windows with four classes:

- `idle`
- `step`
- `vibration`
- `stumble`

This keeps the project close to embedded/sensor work without needing external datasets.

## Model

A compact MLP over fixed-length signal windows. It is intentionally small so latency, size, and quantization effects are visible.

## Quantization

The first implementation uses ONNX Runtime dynamic quantization:

- fast to run
- no calibration dataset required
- good baseline for CPU inference

Future upgrade: static int8 quantization with calibration data.

