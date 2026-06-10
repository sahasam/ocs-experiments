"""Tests for src/model.py."""
import pytest
import torch

from src.model import TinyGPT, build_model


def test_build_model_from_config(tiny_config):
    model = build_model(tiny_config)
    assert isinstance(model, TinyGPT)
    n_params = sum(p.numel() for p in model.parameters())
    # Tiny config should produce a ~5M-param model
    assert 1_000_000 < n_params < 20_000_000, f"got {n_params} params"


def test_forward_shape(tiny_config):
    model = build_model(tiny_config)
    B, T = 2, 8
    idx = torch.randint(0, tiny_config["vocab_size"], (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, tiny_config["vocab_size"])
    assert loss is None


def test_forward_with_targets_returns_finite_loss(tiny_config):
    model = build_model(tiny_config)
    B, T = 2, 8
    idx = torch.randint(0, tiny_config["vocab_size"], (B, T))
    targets = torch.randint(0, tiny_config["vocab_size"], (B, T))
    _, loss = model(idx, targets)
    assert loss is not None
    assert torch.isfinite(loss), f"loss = {loss.item()}"
    # Random init + random targets -> loss near log(vocab_size) ~= 6.93
    assert 4.0 < loss.item() < 11.0, f"unexpected loss {loss.item()}"


def test_backward_populates_grads(tiny_config):
    model = build_model(tiny_config)
    B, T = 2, 8
    idx = torch.randint(0, tiny_config["vocab_size"], (B, T))
    targets = torch.randint(0, tiny_config["vocab_size"], (B, T))
    _, loss = model(idx, targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_sequence_too_long_raises(tiny_config):
    model = build_model(tiny_config)
    T_too_long = tiny_config["block_size"] + 1
    idx = torch.randint(0, tiny_config["vocab_size"], (1, T_too_long))
    with pytest.raises(AssertionError, match="exceeds block_size"):
        model(idx)
