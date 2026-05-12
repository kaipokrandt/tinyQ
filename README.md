# TinyQ

TinyQ is a hardware-aware AI/ML lab for testing how a small classifier behaves after export and quantization.

The project trains a compact PyTorch signal model, exports it to ONNX, dynamically quantizes it with ONNX Runtime, and benchmarks float32 vs int8 inference on CPU.

No API keys. No hosted model calls. No token cost.

## Why it exists

Most small ML demos stop at accuracy. TinyQ keeps going into the parts that matter for deployment:

- model size
- CPU inference latency
- export compatibility
- quantization impact
- reproducible benchmark artifacts

That makes it closer to embedded, edge, and systems-oriented ML work than a notebook-only classifier.

## Pipeline

```text
synthetic sensor windows
  -> PyTorch training
  -> ONNX export
  -> ONNX Runtime int8 quantization
  -> CPU benchmark report
```

The synthetic dataset contains four signal classes: `idle`, `step`, `vibration`, and `stumble`.

## Tech stack

- **PyTorch** for model definition and training.
- **ONNX** for portable inference export.
- **ONNX Runtime** for CPU execution and dynamic int8 quantization.
- **NumPy** for deterministic signal generation.
- **pytest** for correctness checks.

TinyQ uses PyTorch's stable TorchScript-based ONNX exporter because the generated graph quantizes cleanly for this compact MLP.

## Quickstart

Use Python 3.11 or 3.12.

```powershell
git clone https://github.com/kaipokrandt/tinyQ.git
cd tinyQ
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run the full experiment:

```powershell
tinyq all
```

Run individual stages:

```powershell
tinyq train
tinyq export
tinyq quantize
tinyq benchmark
tinyq report
```

## Outputs

Generated files are written to `artifacts/`:

| File | Purpose |
| --- | --- |
| `model.pt` | PyTorch checkpoint |
| `model.onnx` | float32 ONNX export |
| `model.int8.onnx` | quantized ONNX model |
| `metrics.json` | training and test metrics |
| `benchmark.json` | accuracy, latency, and size comparison |
| `report.md` | short generated summary |
| `latency.png` | benchmark plot |

## Current result shape

On the default synthetic workload, the model trains to high accuracy and produces a much smaller int8 ONNX artifact. Exact latency varies by CPU and background load, so the benchmark should be rerun on the target machine.

```powershell
tinyq all
Get-Content artifacts\benchmark.json
```

## Development

```powershell
python -m pytest -q
```

If a virtual environment was previously created with another Python version, delete it before reinstalling dependencies. Compiled packages such as NumPy, PyTorch, and ONNX Runtime are tied to the Python ABI.

## Resume framing

Built TinyQ, a hardware-aware quantized inference lab that trains a compact PyTorch signal classifier, exports it to ONNX, quantizes it with ONNX Runtime, and benchmarks float32 vs int8 CPU inference across accuracy, latency, and model size.

## Next upgrades

- Static int8 quantization with calibration data.
- Real sensor or audio windows instead of synthetic signals.
- C or microcontroller-oriented inference export.
- CI-published benchmark artifacts.
