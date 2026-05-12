from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TinyQConfig:
    seed: int = 42
    samples_per_class: int = 900
    window: int = 64
    classes: tuple[str, ...] = ("idle", "step", "vibration", "stumble")
    hidden: int = 48
    epochs: int = 45
    batch_size: int = 96
    learning_rate: float = 1e-3
    test_fraction: float = 0.2
    benchmark_runs: int = 2000
    artifacts_dir: Path = Path("artifacts")

    @property
    def input_dim(self) -> int:
        return self.window

    @property
    def output_dim(self) -> int:
        return len(self.classes)

