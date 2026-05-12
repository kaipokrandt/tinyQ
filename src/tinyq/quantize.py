from __future__ import annotations

from onnxruntime.quantization import QuantType, quantize_dynamic

from tinyq.config import TinyQConfig


def quantize_model(cfg: TinyQConfig) -> str:
    src = cfg.artifacts_dir / "model.onnx"
    dst = cfg.artifacts_dir / "model.int8.onnx"
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return str(dst)

