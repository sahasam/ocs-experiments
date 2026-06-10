"""End-to-end pipeline test: torchrun -> chakra_converter -> .et files exist.

Marked slow; opt-in via `pytest -m slow`. Uses 2 ranks (not 4) to stay cheap.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TORCHRUN = REPO_ROOT / ".venv" / "bin" / "torchrun"
CHAKRA_CONVERTER = REPO_ROOT / ".venv" / "bin" / "chakra_converter"


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    if not CHAKRA_CONVERTER.exists():
        pytest.skip(
            f"chakra_converter not installed at {CHAKRA_CONVERTER}; "
            f"run scripts/install_chakra.sh"
        )
    if not TORCHRUN.exists():
        pytest.skip(f"torchrun not found at {TORCHRUN}")

    nproc = 2
    config = tmp_path / "config.yaml"
    config.write_text(
        "vocab_size: 256\n"
        "n_layer: 2\n"
        "n_head: 4\n"
        "n_embd: 64\n"
        "block_size: 32\n"
        "micro_batch_size: 1\n"
        "global_batch_size: 2\n"
        "warmup_steps: 1\n"
        "profile_steps: 1\n"
    )
    out = tmp_path / "traces"
    out.mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["GLOO_SOCKET_IFNAME"] = "lo0"

    # Port chosen to avoid colliding with the manual launch_train.sh default (29500)
    result = subprocess.run(
        [
            str(TORCHRUN),
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "--rdzv-backend=static",
            "--master-addr=127.0.0.1",
            "--master-port=29555",
            "--node-rank=0",
            "src/train.py",
            "--config", str(config),
            "--output-dir", str(out),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"torchrun exit {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    for rank in range(nproc):
        et_in = out / f"et_rank{rank}.json"
        assert et_in.exists() and et_in.stat().st_size > 0, f"missing {et_in}"

        et_out = out / f"chakra_workload.{rank}.et"
        result = subprocess.run(
            [
                str(CHAKRA_CONVERTER),
                "--log-filename", str(tmp_path / f"chakra_log_{rank}.log"),
                "PyTorch",
                "--input", str(et_in),
                "--output", str(et_out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"chakra_converter rank {rank} failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert et_out.exists() and et_out.stat().st_size > 0
