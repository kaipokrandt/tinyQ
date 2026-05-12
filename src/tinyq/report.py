from __future__ import annotations

import json

import matplotlib.pyplot as plt

from tinyq.config import TinyQConfig


def write_report(cfg: TinyQConfig) -> str:
    metrics = json.loads((cfg.artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    bench = json.loads((cfg.artifacts_dir / "benchmark.json").read_text(encoding="utf-8"))

    labels = ["fp32", "int8"]
    latencies = [bench["fp32"]["latency_ms"], bench["int8"]["latency_ms"]]
    sizes_kb = [bench["fp32"]["size_bytes"] / 1024.0, bench["int8"]["size_bytes"] / 1024.0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(labels, latencies, color=["#7c3aed", "#d8b4fe"])
    axes[0].set_title("Latency (ms)")
    axes[1].bar(labels, sizes_kb, color=["#7c3aed", "#d8b4fe"])
    axes[1].set_title("Model size (KB)")
    fig.tight_layout()
    fig.savefig(cfg.artifacts_dir / "latency.png", dpi=160)
    plt.close(fig)

    report = f"""# TinyQ Report

## Summary

TinyQ trains a compact signal classifier, exports it to ONNX, quantizes it with ONNX Runtime, and benchmarks float32 vs int8 inference on CPU.

## Training

- train accuracy: {metrics["train_accuracy"]:.4f}
- test accuracy: {metrics["test_accuracy"]:.4f}
- parameters: {metrics["parameters"]}

## Inference

| model | accuracy | latency ms | size bytes |
| --- | ---: | ---: | ---: |
| fp32 | {bench["fp32"]["accuracy"]:.4f} | {bench["fp32"]["latency_ms"]:.4f} | {bench["fp32"]["size_bytes"]} |
| int8 | {bench["int8"]["accuracy"]:.4f} | {bench["int8"]["latency_ms"]:.4f} | {bench["int8"]["size_bytes"]} |

## Takeaway

Dynamic int8 quantization gives a deployment-style model artifact and a concrete accuracy/latency/size tradeoff to discuss.
"""
    out = cfg.artifacts_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    return str(out)

