"""H100 SXM5 constants for cost/power proxies.

Sources:
- $/hr: ~$2-4 spot, ~$8-12 on-demand across clouds (mid-2026); we pick $4.
- TDP: 700 W per SXM5 module (NVIDIA datasheet).
- BF16 peak: 989 TFLOPs/s with sparsity off (NVIDIA whitepaper).

Achieved-fraction defaults to 0.5 — typical sustained BF16 throughput on
real transformer kernels is around 45-60% of peak.
"""

DOLLARS_PER_GPU_HOUR = 4.0
WATTS_PER_GPU = 700
BF16_PEAK_TFLOPS = 989
ACHIEVED_FRACTION_OF_PEAK = 0.5
