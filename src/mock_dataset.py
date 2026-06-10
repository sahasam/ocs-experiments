"""Random-token dataset for trace generation. No corpus, hermetic."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        block_size: int,
        vocab_size: int,
        seed: int = 1337,
    ):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        tokens = torch.randint(
            0,
            self.vocab_size,
            (self.block_size + 1,),
            generator=gen,
            dtype=torch.long,
        )
        return tokens[:-1], tokens[1:]


def build_dataset(config: dict, num_samples: int = 1024) -> RandomTokenDataset:
    return RandomTokenDataset(
        num_samples=num_samples,
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
    )
