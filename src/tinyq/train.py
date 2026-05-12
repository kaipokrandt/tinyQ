from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tinyq.config import TinyQConfig
from tinyq.data import make_dataset, train_test_split
from tinyq.model import SignalMLP, count_parameters


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_model(cfg: TinyQConfig) -> dict[str, float | int]:
    torch.manual_seed(cfg.seed)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    x, y = make_dataset(cfg)
    x_train, y_train, x_test, y_test = train_test_split(x, y, cfg)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = SignalMLP(cfg.input_dim, cfg.hidden, cfg.output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(x_test))
        train_logits = model(torch.from_numpy(x_train))
        metrics = {
            "train_accuracy": accuracy(train_logits, torch.from_numpy(y_train)),
            "test_accuracy": accuracy(test_logits, torch.from_numpy(y_test)),
            "parameters": count_parameters(model),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
        }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__ | {"artifacts_dir": str(cfg.artifacts_dir)},
            "classes": cfg.classes,
        },
        cfg.artifacts_dir / "model.pt",
    )
    np.savez(cfg.artifacts_dir / "dataset.npz", x_test=x_test, y_test=y_test)
    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(path: Path, cfg: TinyQConfig) -> SignalMLP:
    model = SignalMLP(cfg.input_dim, cfg.hidden, cfg.output_dim)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

