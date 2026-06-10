"""Toy training entrypoint. Single-process OR DDP via torchrun."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.mock_dataset import build_dataset
from src.model import build_model
from src.trace_capture import capture_trace


def _set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(0)

    if _is_distributed():
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    is_main = rank == 0
    if is_main:
        print(f"[rank {rank}/{world_size}] config={args.config} output={output_dir}")

    model = build_model(config)
    if _is_distributed():
        model = DistributedDataParallel(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = build_dataset(config, num_samples=1024)

    if _is_distributed():
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        loader = DataLoader(
            dataset, batch_size=config["micro_batch_size"], sampler=sampler
        )
    else:
        loader = DataLoader(
            dataset, batch_size=config["micro_batch_size"], shuffle=False
        )

    data_iter = iter(loader)

    def _step() -> torch.Tensor:
        x, y = next(data_iter)
        _, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.detach()

    for i in range(config["warmup_steps"]):
        loss = _step()
        if is_main:
            print(f"[warmup {i}] loss={loss.item():.4f}")

    with capture_trace(output_dir, rank=rank):
        for i in range(config["profile_steps"]):
            loss = _step()
            if is_main:
                print(f"[profile {i}] loss={loss.item():.4f}")

    if _is_distributed():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
