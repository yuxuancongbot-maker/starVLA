#!/usr/bin/env python3
"""Aggregate live CALVIN metrics from parallel shard logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"1/5\s*:\s*([\d.]+)%\s*\|\s*"
    r"2/5\s*:\s*([\d.]+)%\s*\|\s*"
    r"3/5\s*:\s*([\d.]+)%\s*\|\s*"
    r"4/5\s*:\s*([\d.]+)%\s*\|\s*"
    r"5/5\s*:\s*([\d.]+)%"
)
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def latest_metric_line(path: Path) -> str | None:
    if not path.exists():
        return None

    latest = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if METRIC_RE.search(line) and PROGRESS_RE.search(line):
                latest = line.strip()
    return latest


def parse_line(line: str) -> tuple[list[float], int, int] | None:
    metric_match = METRIC_RE.search(line)
    progress_match = PROGRESS_RE.search(line)
    if metric_match is None or progress_match is None:
        return None

    rates = [float(x) / 100.0 for x in metric_match.groups()]
    done = int(progress_match.group(1))
    total = int(progress_match.group(2))
    return rates, done, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/eval_results/eval_flower_vla_calvin_0520_0932/logs",
        help="Directory containing eval_shard*.log files.",
    )
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    weighted_success = [0.0] * 5
    total_done = 0
    total_expected = 0

    print(f"Reading logs from: {log_dir}")
    print("")

    for shard_id in range(args.num_shards):
        log_path = log_dir / f"eval_shard{shard_id}.log"
        line = latest_metric_line(log_path)
        if line is None:
            print(f"Shard {shard_id}: no metric yet ({log_path})")
            continue

        parsed = parse_line(line)
        if parsed is None:
            print(f"Shard {shard_id}: failed to parse latest metric")
            continue

        rates, done, expected = parsed
        total_done += done
        total_expected += expected
        for i, rate in enumerate(rates):
            weighted_success[i] += rate * done

        rate_text = " | ".join(f"{i + 1}/5: {rate * 100:.1f}%" for i, rate in enumerate(rates))
        print(f"Shard {shard_id}: {done}/{expected} | {rate_text}")

    print("")
    if total_done == 0:
        print("No completed sequences found yet.")
        return

    global_rates = [value / total_done for value in weighted_success]
    average_length = sum(global_rates)

    print("=== Aggregated Current Metrics ===")
    print(f"Completed sequences: {total_done}/{total_expected}")
    for i, rate in enumerate(global_rates):
        print(f"{i + 1}/5 success: {rate * 100:.2f}%")
    print(f"Average Length: {average_length:.4f}")


if __name__ == "__main__":
    main()
