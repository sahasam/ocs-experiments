"""Roofline FLOP counter for a Llama-style transformer block on H100 BF16.

Emits per-phase compute times in microseconds, for use as input to
chakra text_converter (which expects integer microseconds per phase).

Phases:
  fwd     : forward pass of one transformer block
  bwd_ig  : input-gradient (data-grad) backward — roughly equal to fwd
  bwd_wg  : weight-gradient backward — roughly equal to fwd

Total per-block training FLOPs ≈ 3x forward FLOPs (the classic 6N for
training vs 2N for inference per token).
"""
import argparse

from gpu_constants import BF16_PEAK_TFLOPS, ACHIEVED_FRACTION_OF_PEAK


def block_fwd_flops(d_model: int, d_ff: int, seq: int, batch: int) -> int:
    """FLOPs for one transformer block forward pass.

    Counts the four big matmuls: QKV projection (3*d^2 per token),
    attention output projection (d^2 per token), and SwiGLU MLP
    (3*d*d_ff per token for the three projections up/gate/down).
    Softmax + activations are O(seq^2 * d) — included as the attention
    score matmul (2 * seq * d per token).
    """
    tokens = batch * seq

    qkv = 2 * tokens * 3 * d_model * d_model
    out_proj = 2 * tokens * d_model * d_model
    attn_scores = 2 * tokens * seq * d_model  # QK^T
    attn_apply = 2 * tokens * seq * d_model   # softmax(QK^T) @ V
    mlp = 2 * tokens * 3 * d_model * d_ff     # up + gate + down (SwiGLU)

    return qkv + out_proj + attn_scores + attn_apply + mlp


def flops_to_us(flops: int) -> int:
    """Convert FLOPs to microseconds on H100 BF16 at the achieved-fraction rate."""
    achieved_flops_per_sec = BF16_PEAK_TFLOPS * 1e12 * ACHIEVED_FRACTION_OF_PEAK
    seconds = flops / achieved_flops_per_sec
    return max(1, int(round(seconds * 1e6)))


def compute_times(d_model: int, d_ff: int, seq: int, batch: int) -> tuple[int, int, int]:
    """Returns (fwd_us, bwd_ig_us, bwd_wg_us) for one transformer block."""
    fwd_flops = block_fwd_flops(d_model, d_ff, seq, batch)
    fwd_us = flops_to_us(fwd_flops)
    # Backward roughly doubles forward work; split half-and-half between
    # the data-grad and weight-grad branches.
    bwd_ig_us = fwd_us
    bwd_wg_us = fwd_us
    return fwd_us, bwd_ig_us, bwd_wg_us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--d-ff", type=int, required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    args = p.parse_args()
    fwd, bwd_ig, bwd_wg = compute_times(args.d_model, args.d_ff, args.seq, args.batch)
    print(f"{fwd} {bwd_ig} {bwd_wg}")


if __name__ == "__main__":
    main()
