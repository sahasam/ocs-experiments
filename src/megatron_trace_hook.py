"""Monkey-patches Megatron's training_step to wrap the profile window in capture_trace().

Loaded via PYTHONSTARTUP in scripts/launch_megatron.sh, which runs this once
per interpreter (i.e. once per rank under torchrun) before pretrain_gpt.py
imports anything.

Why a monkey-patch and not a fork of pretrain_gpt.py: Megatron's mainline
moves fast, and we want this scaffold to track upstream without merge pain.
The training_step boundary is stable across versions.

Profile window is hard-coded to iteration 5 (matching --profile-step-start/end
in launch_megatron.sh) since this hook reads no config of its own.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROFILE_START = 5
_PROFILE_END = 6


def _install_hook() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from megatron.training import training as _mt
    from src.trace_capture import capture_trace

    rank = int(os.environ.get("RANK", "0"))
    output_dir = root / "traces"

    _original_train_step = _mt.train_step

    def _wrapped_train_step(*args, **kwargs):
        iteration = kwargs.get("iteration")
        if iteration is None and len(args) >= 5:
            iteration = args[4]

        if iteration is not None and _PROFILE_START <= iteration < _PROFILE_END:
            with capture_trace(output_dir, rank):
                return _original_train_step(*args, **kwargs)
        return _original_train_step(*args, **kwargs)

    _mt.train_step = _wrapped_train_step


try:
    _install_hook()
except ImportError:
    # Megatron not yet importable (e.g., running tests on a machine without it).
    # The hook silently no-ops; Megatron's own --profile flag still fires.
    pass
