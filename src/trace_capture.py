"""Wraps torch.profiler + ExecutionTraceObserver into a single context manager.

Produces two files needed by `chakra_trace_link`:
- et_rank{N}.json     - op DAG from ExecutionTraceObserver
- kineto_rank{N}.json - per-op timing from torch.profiler (Kineto)
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from torch.profiler import (
    ExecutionTraceObserver,
    ProfilerActivity,
    profile,
)


@contextmanager
def capture_trace(output_dir: str | Path, rank: int) -> Iterator[profile]:
    """Profile every op inside the with-block; emit ET + Kineto on exit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    et_path = str(output_dir / f"et_rank{rank}.json")
    kineto_path = str(output_dir / f"kineto_rank{rank}.json")

    et_obs = ExecutionTraceObserver()
    et_obs.register_callback(et_path)

    def _export_kineto(prof: profile) -> None:
        prof.export_chrome_trace(kineto_path)

    with profile(
        activities=[ProfilerActivity.CPU],
        execution_trace_observer=et_obs,
        on_trace_ready=_export_kineto,
        record_shapes=True,
    ) as prof:
        yield prof
