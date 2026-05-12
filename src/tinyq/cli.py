from __future__ import annotations

import argparse
import json

from tinyq.benchmark import benchmark
from tinyq.config import TinyQConfig
from tinyq.export_onnx import export_onnx
from tinyq.quantize import quantize_model
from tinyq.report import write_report
from tinyq.train import train_model


def _print(value: object) -> None:
    print(json.dumps(value, indent=2) if isinstance(value, dict) else value)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyQ quantized inference lab")
    parser.add_argument(
        "command",
        choices=["train", "export", "quantize", "benchmark", "report", "all"],
        help="pipeline command",
    )
    args = parser.parse_args()
    cfg = TinyQConfig()

    if args.command in {"train", "all"}:
        _print(train_model(cfg))
    if args.command in {"export", "all"}:
        _print(export_onnx(cfg))
    if args.command in {"quantize", "all"}:
        _print(quantize_model(cfg))
    if args.command in {"benchmark", "all"}:
        _print(benchmark(cfg))
    if args.command in {"report", "all"}:
        _print(write_report(cfg))


if __name__ == "__main__":
    main()

