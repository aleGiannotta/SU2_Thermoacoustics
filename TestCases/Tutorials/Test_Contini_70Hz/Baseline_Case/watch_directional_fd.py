#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path("directional_fd_results.csv")
DEFAULT_STATUS = Path("directional_fd_status.txt")
DEFAULT_REFRESH = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live viewer for the directional finite-difference sweep."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="CSV file written by run_directional_fd_check.py",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
        help="Status file written by run_directional_fd_check.py",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=DEFAULT_REFRESH,
        help="Refresh interval in seconds.",
    )
    return parser.parse_args()


def read_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    status: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        status[key.strip()] = value.strip()
    return status


def read_results(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append(
                {
                    "alpha": float(row["alpha"]),
                    "objective_plus": float(row["objective_plus"]),
                    "objective_minus": float(row["objective_minus"]),
                    "centered_fd_directional": float(row["centered_fd_directional"]),
                    "adjoint_directional": float(row["adjoint_directional"]),
                    "rel_diff": float(row["rel_diff"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.canvas.manager.set_window_title("Directional FD Monitor")

    last_stamp: tuple[float | None, float | None] = (None, None)

    while plt.fignum_exists(fig.number):
        status_mtime = args.status.stat().st_mtime if args.status.exists() else None
        results_mtime = args.results.stat().st_mtime if args.results.exists() else None
        stamp = (status_mtime, results_mtime)
        if stamp != last_stamp:
            last_stamp = stamp
            status = read_status(args.status)
            results = read_results(args.results)

            for axis in axes:
                axis.clear()

            title = "Directional FD Monitor"
            if status:
                completed = status.get("completed_pairs", "?")
                total = status.get("total_pairs", "?")
                current_alpha = status.get("current_alpha", "-")
                title = f"Directional FD Monitor | completed {completed}/{total} | current alpha {current_alpha}"
            fig.suptitle(title)

            if results:
                alphas = [row["alpha"] for row in results]
                j_plus = [row["objective_plus"] for row in results]
                j_minus = [row["objective_minus"] for row in results]
                fd = [row["centered_fd_directional"] for row in results]
                adjoint = results[0]["adjoint_directional"]
                rel_diff = [row["rel_diff"] for row in results]

                axes[0].plot(alphas, j_plus, marker="o", lw=1.2, label="J(+alpha)")
                axes[0].plot(alphas, j_minus, marker="s", lw=1.2, label="J(-alpha)")
                axes[0].set_xscale("log")
                axes[0].set_ylabel("Objective")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                axes[1].axhline(adjoint, color="tab:red", ls="--", lw=1.2, label="Adjoint")
                axes[1].plot(alphas, fd, marker="o", lw=1.2, color="tab:green", label="Centered FD")
                axes[1].set_xscale("log")
                axes[1].set_ylabel("Dir. derivative")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

                axes[2].plot(alphas, rel_diff, marker="o", lw=1.2, color="tab:purple")
                axes[2].set_xscale("log")
                axes[2].set_yscale("log")
                axes[2].set_xlabel("alpha")
                axes[2].set_ylabel("Relative error")
                axes[2].grid(True, alpha=0.3)

            else:
                axes[0].text(0.5, 0.5, "Waiting for directional_fd_results.csv", ha="center", va="center")
                axes[1].text(0.5, 0.5, "Sweep not started yet", ha="center", va="center")
                axes[2].text(0.5, 0.5, "Status will update automatically", ha="center", va="center")
                for axis in axes:
                    axis.set_xticks([])
                    axis.set_yticks([])

            fig.tight_layout()
            fig.canvas.draw_idle()

        plt.pause(args.refresh)


if __name__ == "__main__":
    main()
