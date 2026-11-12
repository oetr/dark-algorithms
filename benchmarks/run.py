"""Run the benchmark suite, print a summary, and optionally export results as metrics.

    python benchmarks/run.py --output results.json [--export-otlp]

With --export-otlp, every result is sent as OpenTelemetry gauges to the endpoint in the
standard OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS environment variables
(for Dash0: the OTLP/HTTP ingress URL and "Authorization=Bearer <token>"). Install the
`telemetry` extra for this. Each commit on main produces one data point per case, which is
what the progress dashboard charts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpga_baseline import load_fpga_baseline  # noqa: E402
from suite import SUITE  # noqa: E402
from yosys_baseline import load_baseline  # noqa: E402

from dark_algorithms.ternary_lut import evaluate  # noqa: E402

ALGORITHM = "ternary-nn-to-lut"


def git_revision() -> str:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_suite() -> list[dict]:
    """Benchmark every case and attach the stored Yosys results for the same network."""
    yosys = load_baseline()
    fpga = load_fpga_baseline()
    results = []
    for case in SUITE:
        net = case.network()
        r = {"case": case.name, "k": case.k, **evaluate(net, k=case.k).to_dict()}
        y = yosys.get(case.name)
        r["yosys_lut_count"] = y["lut_count"] if y else None
        r["yosys_depth"] = y["depth"] if y else None
        # Below 1.0 means fewer LUTs than the open-source Yosys/ABC flow on the same network.
        r["lut_ratio_vs_yosys"] = r["lut_count"] / y["lut_count"] if y else None
        # Real-FPGA reference: Yosys's vendor flow (Xilinx for k=6, ECP5 for k=4) with carry
        # chains. Our k-LUTs map one to one onto the target's LUT primitives with no carry
        # cells, so our LUT sites there are our LUT count.
        f = fpga.get(case.name)
        net_side = f["network"] if f else None
        r["fpga_target"] = f["target"] if f else None
        r["fpga_yosys_lut_sites"] = net_side["lut_sites"] if f else None
        r["fpga_yosys_carry_cells"] = net_side["carry_cells"] if f else None
        r["lut_ratio_vs_fpga"] = r["lut_count"] / net_side["lut_sites"] if f else None
        results.append(r)
    return results


def versus_yosys(results: list[dict]) -> tuple[int, int] | None:
    """Total LUTs of ours and of Yosys/ABC over the cases that have a Yosys result."""
    covered = [r for r in results if r["yosys_lut_count"]]
    if not covered:
        return None
    return sum(r["lut_count"] for r in covered), sum(r["yosys_lut_count"] for r in covered)


def summary_table(results: list[dict]) -> str:
    lines = [
        "| case | equivalent | proven | LUTs | depth | Yosys/ABC LUTs | Yosys depth | vs Yosys |"
        " dense baseline LUTs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        check = ("yes (exhaustive)" if r["exhaustive"] else "yes") if r["equivalent"] else "**NO**"
        proven = f"yes ({r['proof_seconds']:.1f} s)" if r["proven"] else "no"
        if r["yosys_lut_count"]:
            yl, yd = r["yosys_lut_count"], r["yosys_depth"]
            vs = f"{r['lut_ratio_vs_yosys'] - 1:+.1%}"
        else:
            yl = yd = vs = "n/a"
        lines.append(
            f"| {r['case']} | {check} | {proven} | {r['lut_count']} | {r['depth']} | {yl} | {yd} |"
            f" {vs} |"
            f" {r['baseline_lut_count']} |"
        )
    total = sum(r["lut_count"] for r in results)
    lines.append(f"\nTotal LUTs: {total}")
    proven = sum(r["proven"] for r in results)
    lines.append(f"Formally proven equivalent: {proven} of {len(results)} cases")
    vy = versus_yosys(results)
    if vy:
        ours, theirs = vy
        lines.append(
            f"Versus Yosys/ABC on the cases it finished: {ours} vs {theirs} LUTs "
            f"({ours / theirs - 1:+.1%})"
        )
    lines.append("\n" + fpga_table(results))
    return "\n".join(lines)


def fpga_table(results: list[dict]) -> str:
    """Our circuits against Yosys's vendor flows, which also use the FPGA's carry chains."""
    covered = [r for r in results if r["fpga_target"]]
    if not covered:
        return "Real-FPGA reference: none stored."
    lines = [
        "Real-FPGA reference (Yosys vendor flow with carry chains; ours uses no carry cells):",
        "",
        "| case | target | our LUTs | Yosys LUT sites | Yosys carry cells | vs Yosys |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in covered:
        lines.append(
            f"| {r['case']} | {r['fpga_target']} | {r['lut_count']} | {r['fpga_yosys_lut_sites']}"
            f" | {r['fpga_yosys_carry_cells']} | {r['lut_ratio_vs_fpga'] - 1:+.1%} |"
        )
    ours = sum(r["lut_count"] for r in covered)
    theirs = sum(r["fpga_yosys_lut_sites"] for r in covered)
    carry = sum(r["fpga_yosys_carry_cells"] for r in covered)
    lines.append(
        f"\nVersus the vendor flows on the {len(covered)} cases they finished: {ours} vs {theirs}"
        f" LUT sites ({ours / theirs - 1:+.1%}), and 0 vs {carry} carry cells"
    )
    return "\n".join(lines)


def export_otlp(results: list[dict], revision: str) -> bool:
    """Send results as gauges in one OTLP request. Returns False if the backend rejected them."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricExportResult
    from opentelemetry.sdk.resources import Resource

    # Keep every resource attribute identical across commits: a changing value (commit SHA,
    # random instance id, SDK version) would start a new series, and dashboards summing the
    # latest value per series would then add up all commits instead of showing the newest.
    resource = Resource(
        {
            "service.name": "dark-algorithms-benchmark",
            "service.instance.id": "ci-main",
            "vcs.repository.url.full": os.environ.get(
                "BENCHMARK_REPOSITORY_URL", "https://github.com/oetr/dark-algorithms"
            ),
        }
    )
    reader = InMemoryMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = provider.get_meter("dark_algorithms.benchmark")
    gauges = {
        "lut_count": meter.create_gauge("dark_algorithms.benchmark.lut_count", unit="{lut}"),
        "depth": meter.create_gauge("dark_algorithms.benchmark.depth", unit="{level}"),
        "baseline_lut_count": meter.create_gauge(
            "dark_algorithms.benchmark.baseline_lut_count", unit="{lut}"
        ),
        "lut_reduction": meter.create_gauge("dark_algorithms.benchmark.lut_reduction", unit="1"),
        "equivalent": meter.create_gauge("dark_algorithms.benchmark.equivalent", unit="1"),
        "proven": meter.create_gauge("dark_algorithms.benchmark.proven", unit="1"),
        "compile_seconds": meter.create_gauge(
            "dark_algorithms.benchmark.compile_duration", unit="s"
        ),
    }
    # The commit lives on its own metric so the result series stay stable across commits.
    meter.create_gauge("dark_algorithms.benchmark.run_info", unit="1").set(
        1.0, {"vcs.ref.head.revision": revision, "algorithm": ALGORITHM}
    )
    yosys_gauges = {
        "yosys_lut_count": meter.create_gauge(
            "dark_algorithms.benchmark.yosys_lut_count", unit="{lut}"
        ),
        "yosys_depth": meter.create_gauge("dark_algorithms.benchmark.yosys_depth", unit="{level}"),
        "lut_ratio_vs_yosys": meter.create_gauge(
            "dark_algorithms.benchmark.lut_ratio_vs_yosys", unit="1"
        ),
        "fpga_yosys_lut_sites": meter.create_gauge(
            "dark_algorithms.benchmark.fpga_yosys_lut_sites", unit="{lut}"
        ),
        "fpga_yosys_carry_cells": meter.create_gauge(
            "dark_algorithms.benchmark.fpga_yosys_carry_cells", unit="{cell}"
        ),
        "lut_ratio_vs_fpga": meter.create_gauge(
            "dark_algorithms.benchmark.lut_ratio_vs_fpga", unit="1"
        ),
    }
    for r in results:
        attrs = {"benchmark.case": r["case"], "algorithm": ALGORITHM, "lut.k": r["k"]}
        for key, gauge in gauges.items():
            gauge.set(float(r[key]), attrs)
        for key, gauge in yosys_gauges.items():
            if r[key] is not None:
                gauge.set(float(r[key]), attrs)
    data = reader.get_metrics_data()
    exporter = OTLPMetricExporter()
    try:
        result = exporter.export(data)
    finally:
        exporter.shutdown()
        provider.shutdown()
    return result == MetricExportResult.SUCCESS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write results as JSON to this path")
    parser.add_argument("--export-otlp", action="store_true", help="export results as metrics")
    args = parser.parse_args()

    revision = git_revision()
    results = run_suite()
    print(summary_table(results))
    if args.output:
        args.output.write_text(
            json.dumps({"revision": revision, "algorithm": ALGORITHM, "results": results}, indent=2)
        )
    if args.export_otlp:
        if not export_otlp(results, revision):
            print("FAIL: the OTLP backend rejected the metrics; check endpoint and token.")
            return 1
        print("Exported benchmark metrics over OTLP.")
    return 0 if all(r["equivalent"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
