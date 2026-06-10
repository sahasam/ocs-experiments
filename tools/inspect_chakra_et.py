"""Pretty-print a Chakra ET protobuf: node count, op-type histogram, collectives.

Usage: python tools/inspect_chakra_et.py traces/chakra_workload.0.et
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from chakra.schema.protobuf import et_def_pb2
from chakra.src.third_party.utils.protolib import decodeMessage


# Map enum int -> name for the node types we care about
_NODE_TYPE_NAMES = {
    et_def_pb2.INVALID_NODE: "INVALID",
    et_def_pb2.METADATA_NODE: "METADATA",
    et_def_pb2.MEM_LOAD_NODE: "MEM_LOAD",
    et_def_pb2.MEM_STORE_NODE: "MEM_STORE",
    et_def_pb2.COMP_NODE: "COMP",
    et_def_pb2.COMM_SEND_NODE: "COMM_SEND",
    et_def_pb2.COMM_RECV_NODE: "COMM_RECV",
    et_def_pb2.COMM_COLL_NODE: "COMM_COLL",
}

_COLL_TYPE_NAMES = {
    et_def_pb2.ALL_REDUCE: "all_reduce",
    et_def_pb2.ALL_GATHER: "all_gather",
    et_def_pb2.ALL_TO_ALL: "all_to_all",
    et_def_pb2.REDUCE_SCATTER: "reduce_scatter",
    et_def_pb2.BROADCAST: "broadcast",
}


def _get_attr(node, name):
    for a in node.attr:
        if a.name == name:
            if a.HasField("int64_val"):
                return a.int64_val
            if a.HasField("uint64_val"):
                return a.uint64_val
            if a.HasField("string_val"):
                return a.string_val
    return None


def summarize(path: Path) -> None:
    with open(path, "rb") as f:
        md = et_def_pb2.GlobalMetadata()
        decodeMessage(f, md)

        type_counts: Counter[str] = Counter()
        coll_counts: Counter[str] = Counter()
        coll_sizes: list[tuple[str, int]] = []
        total_nodes = 0

        while True:
            node = et_def_pb2.Node()
            if not decodeMessage(f, node):
                break
            total_nodes += 1
            type_name = _NODE_TYPE_NAMES.get(node.type, f"?{node.type}")
            type_counts[type_name] += 1
            if node.type == et_def_pb2.COMM_COLL_NODE:
                coll_type = _get_attr(node, "comm_type")
                coll_name = _COLL_TYPE_NAMES.get(coll_type, f"?{coll_type}")
                coll_counts[coll_name] += 1
                size = _get_attr(node, "comm_size")
                if size is not None:
                    coll_sizes.append((coll_name, int(size)))

    print(f"=== {path} ===")
    print(f"Total nodes:    {total_nodes}")
    print()
    print("Nodes by type:")
    for t, n in type_counts.most_common():
        print(f"  {n:6d}  {t}")
    print()
    if coll_counts:
        print("Collectives by type:")
        for t, n in coll_counts.most_common():
            print(f"  {n:6d}  {t}")
        print()
        print(f"Collective message sizes (bytes), first 10:")
        for name, size in coll_sizes[:10]:
            print(f"  {name:<16s} {size:>12d}")
        total_coll_bytes = sum(s for _, s in coll_sizes)
        print(f"Total collective bytes: {total_coll_bytes:,}")
    else:
        print("No collective ops found.")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    summarize(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
