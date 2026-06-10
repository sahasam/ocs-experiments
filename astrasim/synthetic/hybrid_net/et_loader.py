"""Read Chakra ET files into Python.

Adds the local chakra checkout to sys.path so we can use the same protobuf
schema that text_converter wrote. The chakra package lives outside this repo
at /Users/sahas/workplace/astra-sim/extern/graph_frontend/chakra (the astra-sim
submodule).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

CHAKRA_PARENT = Path(
    os.environ.get("CHAKRA_ROOT",
                   "/Users/sahas/workplace/astra-sim/extern/graph_frontend"))
if str(CHAKRA_PARENT) not in sys.path:
    sys.path.insert(0, str(CHAKRA_PARENT))

from chakra.schema.protobuf import et_def_pb2                       # noqa: E402
from chakra.src.third_party.utils.protolib import decodeMessage     # noqa: E402


@dataclass
class EtNode:
    """A single Chakra ET node, flattened to the fields we use.

    The first six fields are the original set the PerLayer path consumes; the
    rest were added for the STAGE general-DAG path (dag_sim.py) and default to
    empty/None so existing callers are unaffected.
    """
    id:          int
    name:        str
    type:        int                # et_def_pb2.NodeType enum
    duration_us: int                # duration_micros from the proto
    comm_type:   int | None         # CollectiveCommType (None if not a coll)
    comm_size:   int | None         # bytes (None if not a coll/send/recv)
    # --- DAG fields (STAGE) ---
    data_deps:   list[int] = field(default_factory=list)
    ctrl_deps:   list[int] = field(default_factory=list)
    num_ops:     int | None = None  # COMP: FLOPs for roofline
    tensor_size: int | None = None  # COMP: bytes touched, for roofline OI
    pg_name:     str | None = None  # COMM_COLL: process-group id (-> comm_group.json)
    comm_src:    int | None = None  # RECV: source rank
    comm_dst:    int | None = None  # SEND: destination rank
    comm_tag:    int | None = None  # SEND/RECV: matching tag


def _get_attr_int(node, name: str) -> int | None:
    for a in node.attr:
        if a.name == name:
            # int64_val / uint64_val are the fields we care about; comm_type
            # is stored as int64_val in text_converter's output but ProtoLib
            # may have decoded it as double for legacy reasons -- accept both.
            if a.HasField("uint64_val"):
                return int(a.uint64_val)
            if a.HasField("int64_val"):
                return int(a.int64_val)
            if a.HasField("int32_val"):
                return int(a.int32_val)
            if a.HasField("uint32_val"):
                return int(a.uint32_val)
            if a.HasField("double_val"):
                return int(a.double_val)
    return None


def _get_attr_str(node, name: str) -> str | None:
    for a in node.attr:
        if a.name == name and a.HasField("string_val"):
            return a.string_val
    return None


def load_et(path: Path) -> list[EtNode]:
    """Parse one .et file -> ordered list of EtNodes.

    Always captures id/name/type/duration plus data/ctrl deps. Type-specific
    attributes are read per node: COMP -> num_ops/tensor_size; COMM_COLL ->
    comm_type/comm_size/pg_name; SEND -> comm_size/comm_dst/comm_tag; RECV ->
    comm_size/comm_src/comm_tag.
    """
    nodes: list[EtNode] = []
    with open(path, "rb") as f:
        md = et_def_pb2.GlobalMetadata()
        decodeMessage(f, md)
        while True:
            raw = et_def_pb2.Node()
            if not decodeMessage(f, raw):
                break
            comm_type = comm_size = None
            num_ops = tensor_size = None
            pg_name = comm_src = comm_dst = comm_tag = None
            if raw.type == et_def_pb2.COMP_NODE:
                num_ops = _get_attr_int(raw, "num_ops")
                tensor_size = _get_attr_int(raw, "tensor_size")
            elif raw.type == et_def_pb2.COMM_COLL_NODE:
                comm_type = _get_attr_int(raw, "comm_type")
                comm_size = _get_attr_int(raw, "comm_size")
                pg_name = _get_attr_str(raw, "pg_name")
            elif raw.type == et_def_pb2.COMM_SEND_NODE:
                comm_size = _get_attr_int(raw, "comm_size")
                comm_dst = _get_attr_int(raw, "comm_dst")
                comm_tag = _get_attr_int(raw, "comm_tag")
            elif raw.type == et_def_pb2.COMM_RECV_NODE:
                comm_size = _get_attr_int(raw, "comm_size")
                comm_src = _get_attr_int(raw, "comm_src")
                comm_tag = _get_attr_int(raw, "comm_tag")
            nodes.append(EtNode(
                id=raw.id,
                name=raw.name,
                type=raw.type,
                duration_us=int(raw.duration_micros),
                comm_type=comm_type,
                comm_size=comm_size,
                data_deps=list(raw.data_deps),
                ctrl_deps=list(raw.ctrl_deps),
                num_ops=num_ops,
                tensor_size=tensor_size,
                pg_name=pg_name,
                comm_src=comm_src,
                comm_dst=comm_dst,
                comm_tag=comm_tag,
            ))
    return nodes


# Re-export the enum constants callers will compare against.
COMP_NODE = et_def_pb2.COMP_NODE
COMM_COLL_NODE = et_def_pb2.COMM_COLL_NODE
COMM_SEND_NODE = et_def_pb2.COMM_SEND_NODE
COMM_RECV_NODE = et_def_pb2.COMM_RECV_NODE
ALL_REDUCE = et_def_pb2.ALL_REDUCE
ALL_GATHER = et_def_pb2.ALL_GATHER
REDUCE_SCATTER = et_def_pb2.REDUCE_SCATTER
ALL_TO_ALL = et_def_pb2.ALL_TO_ALL
