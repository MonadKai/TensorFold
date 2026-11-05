#!/usr/bin/env python3
"""Export only measured counters and public kernel names from Nsight Compute raw CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = (
    "gpu__time_duration.sum", "lts__t_bytes.sum", "lts__t_sectors_op_read.sum",
    "lts__t_sector_hit_rate.pct", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)


def summarize(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 2:
        raise ValueError("no profiled kernels")
    units = rows[0]
    kernels = []
    for row in rows[1:]:
        counters = {}
        for key in METRICS:
            value = row.get(key)
            counters[key] = dict(value=float(value), unit=units[key]) if value else None
        kernels.append(dict(name=row["Kernel Name"], counters=counters))
    return dict(artifact=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kernels=kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = dict(schema=1, traces=[summarize(path) for path in args.raw_csv],
                  limitations=["These are counter traces of individual operations, separate from timing samples.",
                               "Caches and clocks were uncontrolled; retain each profiler warning in raw evidence.",
                               "L2 request bytes are not measured DRAM bytes or physical memory allocations.",
                               "A nonzero tensor-pipe counter establishes activity, not high tensor-core utilization."])
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
