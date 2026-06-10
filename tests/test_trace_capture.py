"""Tests for src/trace_capture.py — single-process, no DDP."""
import json
from pathlib import Path

import torch

from src.trace_capture import capture_trace


def _do_some_work() -> None:
    """Some matmul + nonlinearity ops, enough to populate the trace."""
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    for _ in range(5):
        c = torch.matmul(a, b)
        c = torch.nn.functional.gelu(c)
        c.sum().backward = None
        a = c.detach()


def test_capture_trace_produces_both_files(tmp_trace_dir):
    with capture_trace(tmp_trace_dir, rank=0):
        _do_some_work()

    et_path = Path(tmp_trace_dir) / "et_rank0.json"
    kineto_path = Path(tmp_trace_dir) / "kineto_rank0.json"

    assert et_path.exists(), f"missing {et_path}"
    assert kineto_path.exists(), f"missing {kineto_path}"
    assert et_path.stat().st_size > 0
    assert kineto_path.stat().st_size > 0


def test_kineto_json_parses_and_has_cpu_ops(tmp_trace_dir):
    with capture_trace(tmp_trace_dir, rank=0):
        _do_some_work()

    kineto_path = Path(tmp_trace_dir) / "kineto_rank0.json"
    with open(kineto_path) as f:
        data = json.load(f)

    assert "traceEvents" in data, "Kineto JSON missing traceEvents"
    events = data["traceEvents"]
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    assert len(cpu_ops) > 0, "no cpu_op events captured"


def test_et_json_parses_and_has_nodes(tmp_trace_dir):
    with capture_trace(tmp_trace_dir, rank=0):
        _do_some_work()

    et_path = Path(tmp_trace_dir) / "et_rank0.json"
    with open(et_path) as f:
        data = json.load(f)

    assert "nodes" in data, f"ET JSON missing 'nodes', keys: {list(data.keys())}"
    assert len(data["nodes"]) > 0, "ET has zero nodes"


def test_different_ranks_get_separate_files(tmp_trace_dir):
    with capture_trace(tmp_trace_dir, rank=0):
        _do_some_work()
    with capture_trace(tmp_trace_dir, rank=3):
        _do_some_work()

    for r in (0, 3):
        assert (Path(tmp_trace_dir) / f"et_rank{r}.json").exists()
        assert (Path(tmp_trace_dir) / f"kineto_rank{r}.json").exists()
