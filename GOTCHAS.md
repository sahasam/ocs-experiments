# Gotchas

Things that bit us during initial setup. Each entry: **symptom -> root cause -> fix**.

## 1. macOS torchrun `--standalone` hangs at rendezvous

**Symptom**

```
torch.distributed.DistNetworkError: The client socket has timed out after 300000ms
while trying to connect to (1.0.0.0.0.0.0.0...ip6.arpa, 56061).
```

That reverse-DNS-looking address is IPv6 `::1`.

**Cause**

`torchrun --standalone` lets `_this_node.addr` resolve via `socket.getfqdn()`,
which on macOS returns an IPv6 loopback. The TCPStore binds there, workers
can't reach it, rendezvous times out at 5 minutes.

**Fix**

Don't use `--standalone`. Use static rendezvous with explicit IPv4 loopback:

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv-backend=static \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  --node-rank=0 \
  src/train.py ...
```

See `scripts/launch_train.sh`.

## 2. `pip install chakra @ git+...` hangs forever

**Symptom**

`pip install` runs for 10+ minutes with near-zero CPU, output buffered to nothing.
Killing it shows the last line was downloading `jupyterlab-4.x.x-py3-none-any.whl`.

**Cause**

`chakra`'s install_requires lists `HolisticTraceAnalysis @ git+...`, which in
turn depends on `jupyterlab -> nbconvert -> beautifulsoup -> ...`. On Python 3.14
the resolver struggles and the download stalls. We only need `chakra_converter`
for our pipeline, not `chakra_trace_link` (which is what needs HolisticTraceAnalysis).

**Fix**

```bash
git clone --depth 1 https://github.com/mlcommons/chakra.git third_party/chakra
.venv/bin/pip install --no-deps ./third_party/chakra
.venv/bin/pip install protobuf
```

Encapsulated in `scripts/install_chakra.sh`. `chakra_converter` works on the
raw `ExecutionTraceObserver` JSON directly when there's no GPU device trace
to merge, so `chakra_trace_link` is unnecessary for our path.

## 3. AstraSim binary won't run natively on macOS

**Symptom**

```
$ /Users/sahas/workplace/astra-sim/build/.../AstraSim_Analytical_Congestion_Aware
zsh: exec format error: .../AstraSim_Analytical_Congestion_Aware
```

**Cause**

`file` reports `ELF 64-bit LSB pie executable, ARM aarch64, ... for GNU/Linux 3.7.0`.
It was built inside the `astra-sim:latest` Docker image, not for darwin.

**Fix**

Always invoke through Docker. Mount both the astra-sim repo (for the binary +
example configs) and your project dir (for our chakra ETs):

```bash
docker run --rm \
  -v /Users/sahas/workplace/astra-sim:/app/astra-sim \
  -v /Users/sahas/workplace/llm-parallelism:/app/llm-parallelism \
  astra-sim:latest \
  /app/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware \
    --workload-configuration=... --system-configuration=... ...
```

See `astrasim/run_reference.sh` and `astrasim/run_our_traces.sh`.

## 4. `chakra_converter --log-filename` rejected by subcommand

**Symptom**

```
chakra_converter: error: unrecognized arguments: --log-filename /tmp/x.log
```

...even though `--help` shows the flag exists.

**Cause**

`--log-filename` is defined on the **parent** parser, not the `PyTorch` subparser.
argparse requires it to appear before the subcommand name.

**Fix**

```bash
# WRONG
chakra_converter PyTorch --input X --output Y --log-filename Z

# RIGHT
chakra_converter --log-filename Z PyTorch --input X --output Y
```

## 5. Chakra ETs from CPU runs lack collectives (`gloo:` not recognized)

**Symptom**

`tools/inspect_chakra_et.py traces/chakra_workload.0.et` reports `Total nodes: 20503`
but `No collective ops found.` — even though we ran DDP and should have allreduces.

**Cause**

`chakra/src/converter/pytorch_node.py::is_nccl_op` is literally:

```python
return "nccl:" in self.name
```

Our gloo backend produces `gloo:all_reduce`, which doesn't match. The ops survive
in the Chakra ET as plain `COMP` nodes; AstraSim sees no communication to simulate.

**Workaround**

None worth shipping for the toy. Confirm collectives WERE captured by torch
(in the Kineto trace):

```bash
.venv/bin/python tools/inspect_kineto.py traces/kineto_rank0.json | grep -i 'allreduce\|gloo'
```

**Real fix**

Path A: real NCCL traces on GPU. The same converter will then recognize the ops.

## 6. AstraSim crashes on our Chakra ETs (METADATA node, dangling deps)

**Symptom**

```
[statistics] [critical] Invalid node_type, node.id=3, node.type=1
AstraSim_Analytical_Congestion_Aware: .../Statistics.cc:72:
  ... Assertion `false' failed.
```

After filtering METADATA nodes:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Node 3 in data_dep graph, but not found in index, file might be corrupted
```

**Cause**

On CPU, chakra's PyTorch converter keeps the raw ExecutionTrace including a
`METADATA_NODE` (type=1, e.g. `## process_group:init ##`). AstraSim's statistics
layer asserts on unknown node types. Stripping the metadata leaves dangling
parent IDs in the data-dep graph.

The real GPU path strips these during host+device merge.

**Workaround**

Use `astrasim/run_reference.sh` to confirm the binary works; accept that our
own traces won't simulate end-to-end on Path C. Resolved by Path A.

## 7. Python 3.14 + torch wheel availability

Worth knowing rather than a gotcha that bit us: torch 2.12 ships a `cp314`
macOS arm64 wheel, so Python 3.14 works fine. Older torch versions (≤2.9)
do not — if you downgrade torch, also downgrade Python.

## 8. Megatron-LM is GPU+Linux only; don't try to validate it locally on macOS

**Symptom**

`scripts/install_megatron.sh` may install on macOS, but `scripts/launch_megatron.sh`
fails immediately — either on the NCCL backend init (no CUDA) or on a
`transformer_engine` import (no H100 kernels available).

**Cause**

Megatron's training path requires NCCL (CUDA-only) and the `--transformer-impl
transformer_engine` flag pulls in H100-specific fused kernels. Neither is
available off-cluster.

**Fix**

Use the toy pipeline (`scripts/launch_train.sh` → `src/train.py`) for any
local CPU validation; it produces a structurally compatible Chakra ET (just
without real NCCL collectives — see gotcha #5). Reserve Megatron for the
actual H100 nodes. `tests/test_llama70b_configs.py` validates the scaffold's
configs and shell scripts without invoking Megatron itself, so CPU CI still
exercises the new code paths.
