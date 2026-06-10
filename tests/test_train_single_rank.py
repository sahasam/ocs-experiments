"""Smoke test for src/train.py in single-process mode (no torchrun)."""
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_single_rank_train_smoke(tmp_path):
    """Run train.py end-to-end in 1 process. Should exit 0 and emit trace files."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "vocab_size: 256\n"
        "n_layer: 2\n"
        "n_head: 4\n"
        "n_embd: 64\n"
        "block_size: 32\n"
        "micro_batch_size: 1\n"
        "global_batch_size: 1\n"
        "warmup_steps: 1\n"
        "profile_steps: 1\n"
    )
    out = tmp_path / "traces"

    env = os.environ.copy()
    # Belt-and-suspenders: ensure single-process path is exercised
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"train.py exit {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (out / "et_rank0.json").exists(), (
        f"missing et_rank0.json\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (out / "kineto_rank0.json").exists()
    assert (out / "et_rank0.json").stat().st_size > 0
    assert (out / "kineto_rank0.json").stat().st_size > 0
