"""Tests for neural model forward passes and gradient flow."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import _REGISTRY, build_model, count_parameters

BATCH = 8
WINDOW = 30
N_FEATURES = 14


@pytest.mark.parametrize("model_name", list(_REGISTRY.keys()))
def test_forward_output_shape(model_name):
    model = build_model(model_name, n_features=N_FEATURES)
    x = torch.randn(BATCH, WINDOW, N_FEATURES)
    out = model(x)
    assert out.shape == (BATCH,), (
        f"{model_name}: expected output shape ({BATCH},), got {out.shape}"
    )


@pytest.mark.parametrize("model_name", list(_REGISTRY.keys()))
def test_gradients_flow(model_name):
    model = build_model(model_name, n_features=N_FEATURES)
    x = torch.randn(BATCH, WINDOW, N_FEATURES)
    out = model(x)
    loss = out.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{model_name}: no gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"{model_name}: NaN gradient in {name}"


@pytest.mark.parametrize("model_name", list(_REGISTRY.keys()))
def test_parameter_count_nonzero(model_name):
    model = build_model(model_name, n_features=N_FEATURES)
    n = count_parameters(model)
    assert n > 0, f"{model_name}: expected nonzero parameter count"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_architecture", n_features=N_FEATURES)
