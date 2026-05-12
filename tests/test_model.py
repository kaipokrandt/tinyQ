import torch

from tinyq.model import SignalMLP, count_parameters


def test_model_forward_shape():
    model = SignalMLP(input_dim=64, hidden=16, output_dim=4)
    y = model(torch.zeros(3, 64))
    assert tuple(y.shape) == (3, 4)
    assert count_parameters(model) > 0

