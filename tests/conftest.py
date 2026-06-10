"""Shared pytest fixtures."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Reset RNG state before every test so failures are reproducible."""
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture
def tiny_config() -> dict:
    """In-memory copy of configs/toy_model.yaml. Tests don't read the YAML."""
    return {
        "vocab_size": 1024,
        "n_layer": 6,
        "n_head": 4,
        "n_embd": 256,
        "block_size": 128,
        "micro_batch_size": 1,
        "global_batch_size": 4,
        "warmup_steps": 2,
        "profile_steps": 3,
    }


@pytest.fixture
def tmp_trace_dir(tmp_path):
    """Writable temp directory for profiler outputs, auto-cleaned by pytest."""
    return tmp_path
