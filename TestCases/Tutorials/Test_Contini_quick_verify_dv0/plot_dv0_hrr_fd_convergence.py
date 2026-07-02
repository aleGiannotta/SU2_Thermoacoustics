#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CSV = Path("dv0_hrr_fd_convergence.csv")
DEFAULT_PNG = Path("dv0_hrr_fd_convergence.png")
DEFAULT_ADJOINT = 337567.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized DV0 FD convergence from an existing CSV file.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input convergence CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PNG, help="Output plot image.")
    parser.add_argument(
        "--adjoint-dv0",
        type=float,
        default=DEFAULT_ADJOINT,
        help="Adjoint reference gradient used to normalize FD values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    steps: list[float] = []
    fd_gradients: list[float] = []

    with args.csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames or "fd_gradient" not in reader.fieldnames:
            raise RuntimeError(f"CSV must contain 'step' and 'fd_gradient' columns: {args.csv}")
        for row in reader:
            steps.append(float(row["step"]))
            fd_gradients.append(float(row["fd_gradient"]))

    if not steps:
        raise RuntimeError(f"No data rows found in {args.csv}")

    normalized = [value / args.adjoint_dv0 for value in fd_gradients]

    plt.figure(figsize=(7.0, 4.5))
    plt.semilogx(steps, normalized, marker="o", linewidth=2.0, label="FD / AD")
    plt.axhline(1.0, color="crimson", linestyle="--", linewidth=2.0, label="AD reference")
    plt.xlabel("FD step size")
    plt.ylabel("FD gradient / AD gradient")
    plt.title("DV0 steady HRR gradient convergence")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
