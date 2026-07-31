from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "flash_benchmark.csv"
ASSETS = ROOT / "assets"


def rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["status"] == "success"]


def plot_latency(data: list[dict[str, str]]) -> None:
    grouped = defaultdict(list)
    for row in data:
        if row["phase"] == "forward" and row["head_dim"] == "64":
            grouped[row["implementation"]].append((int(row["sequence_length"]), float(row["latency_ms_p50"])))
    for implementation, values in grouped.items():
        values.sort()
        plt.plot([x for x, _ in values], [y for _, y in values], marker="o", label=implementation)
    plt.xscale("log", base=2)
    plt.xlabel("sequence length")
    plt.ylabel("forward p50 latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "flash_latency.png", dpi=160)
    plt.close()


def plot_memory(data: list[dict[str, str]]) -> None:
    grouped = defaultdict(list)
    for row in data:
        if row["phase"] == "forward" and row["head_dim"] == "64":
            grouped[row["implementation"]].append((int(row["sequence_length"]), float(row["peak_allocated_mib"])))
    for implementation, values in grouped.items():
        values.sort()
        plt.plot([x for x, _ in values], [y for _, y in values], marker="o", label=implementation)
    plt.xscale("log", base=2)
    plt.xlabel("sequence length")
    plt.ylabel("forward peak allocated (MiB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "flash_memory.png", dpi=160)
    plt.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = rows()
    if not data:
        raise RuntimeError(f"No successful rows found in {INPUT}")
    plot_latency(data)
    plot_memory(data)
    print(f"wrote {ASSETS / 'flash_latency.png'} and {ASSETS / 'flash_memory.png'}")


if __name__ == "__main__":
    main()
