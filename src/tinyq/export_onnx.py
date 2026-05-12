from __future__ import annotations

import sys

import torch

from tinyq.config import TinyQConfig
from tinyq.train import load_model


def export_onnx(cfg: TinyQConfig) -> str:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg.artifacts_dir / "model.pt", cfg)
    sample = torch.zeros(1, cfg.input_dim, dtype=torch.float32)
    out = cfg.artifacts_dir / "model.onnx"

    torch.onnx.export(
        model,
        (sample,),
        str(out),
        input_names=["signal"],
        output_names=["logits"],
        dynamic_axes={"signal": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return str(out)
