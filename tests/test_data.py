import numpy as np

from tinyq.config import TinyQConfig
from tinyq.data import make_dataset, train_test_split


def test_dataset_shape_and_labels():
    cfg = TinyQConfig(samples_per_class=8, window=32)
    x, y = make_dataset(cfg)
    assert x.shape == (32, 32)
    assert y.shape == (32,)
    assert x.dtype == np.float32
    assert set(y.tolist()) == {0, 1, 2, 3}


def test_split_sizes():
    cfg = TinyQConfig(samples_per_class=10, test_fraction=0.25)
    x, y = make_dataset(cfg)
    x_train, y_train, x_test, y_test = train_test_split(x, y, cfg)
    assert len(x_train) == len(y_train) == 30
    assert len(x_test) == len(y_test) == 10

