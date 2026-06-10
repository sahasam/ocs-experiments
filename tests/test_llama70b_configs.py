"""Validation tests for the Llama-3 70B / 64-GPU scaffold.

These run without GPUs and without Megatron installed. They check that:
- The YAML and JSON config files parse.
- Internal numeric consistency holds (TP*PP*DP == 64; topology depth matches
  implementation-array length; npus_count product == total GPUs).
- The new shell scripts are syntactically valid bash.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def llama70b_yaml() -> dict:
    with open(ROOT / "configs" / "llama3_70b.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def network_yml() -> dict:
    with open(ROOT / "astrasim" / "llama70b" / "network.yml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def system_json() -> dict:
    with open(ROOT / "astrasim" / "llama70b" / "system.json") as f:
        return json.load(f)


def test_llama70b_yaml_has_required_keys(llama70b_yaml):
    required = {
        "n_layer", "n_head", "n_kv_head", "n_embd", "ffn_hidden",
        "seq_len", "vocab_size",
        "tensor_parallel", "pipeline_parallel", "data_parallel",
        "micro_batch_size", "global_batch_size",
        "profile_step_start", "profile_step_end",
    }
    assert required <= set(llama70b_yaml.keys())


def test_llama70b_parallelism_product_matches_64_gpus(llama70b_yaml):
    tp = llama70b_yaml["tensor_parallel"]
    pp = llama70b_yaml["pipeline_parallel"]
    dp = llama70b_yaml["data_parallel"]
    assert tp * pp * dp == 64


def test_llama70b_layers_divisible_by_pp(llama70b_yaml):
    assert llama70b_yaml["n_layer"] % llama70b_yaml["pipeline_parallel"] == 0


def test_llama70b_heads_divisible_by_tp(llama70b_yaml):
    assert llama70b_yaml["n_head"] % llama70b_yaml["tensor_parallel"] == 0
    assert llama70b_yaml["n_kv_head"] <= llama70b_yaml["n_head"]


def test_llama70b_profile_window_is_one_step(llama70b_yaml):
    assert llama70b_yaml["profile_step_end"] - llama70b_yaml["profile_step_start"] == 1


def test_network_yml_topology_matches_64_npus(network_yml):
    topology = network_yml["topology"]
    npus = network_yml["npus_count"]
    assert len(topology) == len(npus)
    product = 1
    for n in npus:
        product *= n
    assert product == 64


def test_network_yml_bandwidth_latency_per_tier(network_yml):
    depth = len(network_yml["topology"])
    assert len(network_yml["bandwidth"]) == depth
    assert len(network_yml["latency"]) == depth


def test_system_json_implementation_arrays_match_topology_depth(system_json, network_yml):
    depth = len(network_yml["topology"])
    for key in (
        "all-reduce-implementation",
        "all-gather-implementation",
        "reduce-scatter-implementation",
        "all-to-all-implementation",
    ):
        assert len(system_json[key]) == depth, f"{key} length must equal topology depth ({depth})"


def test_system_json_active_chunks_matches_multitier(system_json, network_yml):
    if len(network_yml["topology"]) > 1:
        assert system_json["active-chunks-per-dimension"] >= 2


@pytest.mark.parametrize("script", [
    "scripts/install_megatron.sh",
    "scripts/launch_megatron.sh",
    "astrasim/llama70b/run.sh",
])
def test_shell_script_syntax(script):
    bash = shutil.which("bash")
    assert bash, "bash not on PATH"
    result = subprocess.run(
        [bash, "-n", str(ROOT / script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{script} failed syntax check: {result.stderr}"
