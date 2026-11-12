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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from suite import SUITE  # noqa: E402

from dark_algorithms.ternary_lut import evaluate, random_network  # noqa: E402

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
    results = []
    for case in SUITE:
        net = random_network(case.layer_sizes, case.sparsity, np.random.default_rng(case.seed))
        r = evaluate(net, k=case.k)
        results.append({"case": case.name, "k": case.k, **r.to_dict()})
    return results


def summary_table(results: list[dict]) -> str:
    lines = [
        "| case | equivalent | LUTs | depth | dense baseline LUTs | LUT reduction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        check = ("yes (exhaustive)" if r["exhaustive"] else "yes") if r["equivalent"] else "**NO**"
        lines.append(
            f"| {r['case']} | {check} | {r['lut_count']} | {r['depth']} | "
            f"{r['baseline_lut_count']} | {r['lut_reduction']:.1%} |"
        )
    total = sum(r["lut_count"] for r in results)
    lines.append(f"\nTotal LUTs: {total}")
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
        "compile_seconds": meter.create_gauge(
            "dark_algorithms.benchmark.compile_duration", unit="s"
        ),
    }
    # The commit lives on its own metric so the result series stay stable across commits.
    meter.create_gauge("dark_algorithms.benchmark.run_info", unit="1").set(
        1.0, {"vcs.ref.head.revision": revision, "algorithm": ALGORITHM}
    )
    for r in results:
        attrs = {"benchmark.case": r["case"], "algorithm": ALGORITHM, "lut.k": r["k"]}
        for key, gauge in gauges.items():
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
