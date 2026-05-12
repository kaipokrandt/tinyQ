from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tinyq.config import TinyQConfig


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _accuracy(session: ort.InferenceSession, x: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for sample, label in zip(x.astype(np.float32), y):
        pred = session.run(None, {"signal": sample.reshape(1, -1)})[0].argmax(axis=1)[0]
        correct += int(pred == label)
    return correct / len(y)


def _latency_ms(session: ort.InferenceSession, x: np.ndarray, runs: int) -> float:
    sample = x[:1].astype(np.float32)
    for _ in range(20):
        session.run(None, {"signal": sample})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"signal": sample})
    return (time.perf_counter() - start) * 1000.0 / runs


def benchmark(cfg: TinyQConfig) -> dict[str, dict[str, float | int]]:
    data = np.load(cfg.artifacts_dir / "dataset.npz")
    x_test = data["x_test"]
    y_test = data["y_test"]

    fp32_path = cfg.artifacts_dir / "model.onnx"
    int8_path = cfg.artifacts_dir / "model.int8.onnx"
    fp32 = _session(fp32_path)
    int8 = _session(int8_path)

    result = {
        "fp32": {
            "accuracy": _accuracy(fp32, x_test, y_test),
            "latency_ms": _latency_ms(fp32, x_test, cfg.benchmark_runs),
            "size_bytes": fp32_path.stat().st_size,
        },
        "int8": {
            "accuracy": _accuracy(int8, x_test, y_test),
            "latency_ms": _latency_ms(int8, x_test, cfg.benchmark_runs),
            "size_bytes": int8_path.stat().st_size,
        },
    }
    result["delta"] = {
        "accuracy": result["int8"]["accuracy"] - result["fp32"]["accuracy"],
        "latency_ms": result["int8"]["latency_ms"] - result["fp32"]["latency_ms"],
        "size_bytes": result["int8"]["size_bytes"] - result["fp32"]["size_bytes"],
    }

    (cfg.artifacts_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
