"""Tests for src/mock_dataset.py."""
import torch

from src.mock_dataset import build_dataset


def test_length(tiny_config):
    ds = build_dataset(tiny_config, num_samples=512)
    assert len(ds) == 512


def test_item_shape_and_dtype(tiny_config):
    ds = build_dataset(tiny_config, num_samples=4)
    x, y = ds[0]
    assert x.shape == (tiny_config["block_size"],)
    assert y.shape == (tiny_config["block_size"],)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_token_ids_in_range(tiny_config):
    ds = build_dataset(tiny_config, num_samples=8)
    for i in range(len(ds)):
        x, y = ds[i]
        assert (x >= 0).all() and (x < tiny_config["vocab_size"]).all()
        assert (y >= 0).all() and (y < tiny_config["vocab_size"]).all()


def test_deterministic_across_calls(tiny_config):
    ds = build_dataset(tiny_config, num_samples=4)
    x1, y1 = ds[2]
    x2, y2 = ds[2]
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_different_indices_give_different_data(tiny_config):
    ds = build_dataset(tiny_config, num_samples=4)
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert not torch.equal(x0, x1)


def test_input_target_shifted(tiny_config):
    """target[i] is the next token after input[i] — classic LM setup."""
    ds = build_dataset(tiny_config, num_samples=4)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1])
