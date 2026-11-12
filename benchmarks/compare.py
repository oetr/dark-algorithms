"""Regression gate: compare a pull request's benchmark results with its base branch.

    python benchmarks/compare.py base.json head.json

Fails (exit 1) when, on the cases both runs share:
  * any case's circuit is no longer equivalent to its network, or
  * the total LUT count grows, or
  * the worst-case depth grows.

A change that trades one metric for the other has to be argued for by a person, which is
the point: autonomous changes may only move the frontier forward.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(base_path: str, head_path: str) -> int:
    base = {r["case"]: r for r in json.loads(Path(base_path).read_text())["results"]}
    head = {r["case"]: r for r in json.loads(Path(head_path).read_text())["results"]}
    shared = sorted(base.keys() & head.keys())
    failures = [f"{c}: circuit is not equivalent" for c in head if not head[c]["equivalent"]]

    print("| case | LUTs base -> head | depth base -> head |")
    print("|---|---:|---:|")
    for c in shared:
        b, h = base[c], head[c]
        print(f"| {c} | {b['lut_count']} -> {h['lut_count']} | {b['depth']} -> {h['depth']} |")

    if shared:
        base_luts = sum(base[c]["lut_count"] for c in shared)
        head_luts = sum(head[c]["lut_count"] for c in shared)
        base_depth = max(base[c]["depth"] for c in shared)
        head_depth = max(head[c]["depth"] for c in shared)
        print(
            f"\nTotal LUTs: {base_luts} -> {head_luts}; worst depth: {base_depth} -> {head_depth}"
        )
        if head_luts > base_luts:
            failures.append(f"total LUT count regressed: {base_luts} -> {head_luts}")
        if head_depth > base_depth:
            failures.append(f"worst-case depth regressed: {base_depth} -> {head_depth}")

    for f in failures:
        print(f"FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
