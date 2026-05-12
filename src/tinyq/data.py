from __future__ import annotations

import math

import numpy as np

from tinyq.config import TinyQConfig


def make_dataset(cfg: TinyQConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, cfg.window, dtype=np.float32)

    for label, name in enumerate(cfg.classes):
        for _ in range(cfg.samples_per_class):
            noise = rng.normal(0.0, 0.055, size=cfg.window).astype(np.float32)

            if name == "idle":
                signal = rng.normal(0.0, 0.035, size=cfg.window).astype(np.float32)
            elif name == "step":
                freq = rng.uniform(1.4, 2.4)
                signal = np.sin(2 * math.pi * freq * t).astype(np.float32)
            elif name == "vibration":
                freq = rng.uniform(9.0, 14.0)
                signal = 0.55 * np.sin(2 * math.pi * freq * t).astype(np.float32)
            elif name == "stumble":
                center = rng.uniform(0.35, 0.68)
                width = rng.uniform(0.025, 0.055)
                pulse = np.exp(-((t - center) ** 2) / (2 * width**2)).astype(np.float32)
                signal = 0.35 * np.sin(2 * math.pi * 2.0 * t).astype(np.float32) + 1.35 * pulse
            else:
                raise ValueError(f"unknown class {name}")

            scale = rng.uniform(0.85, 1.15)
            offset = rng.uniform(-0.08, 0.08)
            xs.append((scale * signal + offset + noise).astype(np.float32))
            ys.append(np.array(label, dtype=np.int64))

    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def train_test_split(
    x: np.ndarray, y: np.ndarray, cfg: TinyQConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(y) * (1.0 - cfg.test_fraction))
    return x[:split], y[:split], x[split:], y[split:]

