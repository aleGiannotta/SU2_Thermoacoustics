#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import TextIO

import matplotlib.pyplot as plt


CFG = Path("contini_flame.cfg")
TEMPLATE_CFG = Path("contini_flame_unsteady.cfg")
LOG_FILE = Path("dv0_hrr_fd_convergence.log")
RESULTS_FILE = Path("dv0_hrr_fd_convergence.csv")
PLOT_FILE = Path("dv0_hrr_fd_convergence.png")
BASELINE_HISTORY = Path("history_dv0_baseline.csv")
PERTURBED_HISTORY_FMT = "history_dv0_step_{step_index:02d}.csv"
FD_GRAD_FILE = Path("of_grad_dv0_hrr_fd.dat")
DOT_FILE = Path("of_grad")
BASE_MESH = "mesh_coarse_ffd.su2"
DEFORMED_MESH = "mesh_out.su2"
BASE_ADJOINT = "restart_adj_hrr"
DEFAULT_ITER = 10000
DEFAULT_STEPS = "1e-3,1e-4,1e-5"
DEFAULT_ADJOINT_DV0 = "-3.9610100000000002e+03"
DEFAULT_RESTART = "restart_flow.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FD step sizes for DV 0 of the steady Contini global heat-release objective."
    )
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS, help="Comma-separated FD step sizes.")
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER, help="Direct and adjoint iteration limit.")
    parser.add_argument(
        "--adjoint-dv0",
        type=float,
        default=float(DEFAULT_ADJOINT_DV0),
        help="Reference adjoint gradient for DV 0. No adjoint solve is run.",
    )
    parser.add_argument(
        "--restart-solution",
        type=str,
        default=DEFAULT_RESTART,
        help="Previously converged steady restart file to use as the primal starting point.",
    )
    return parser.parse_args()


def read_cfg_lines() -> list[str]:
    return CFG.read_text().splitlines()


def write_cfg_lines(lines: list[str]) -> None:
    CFG.write_text("\n".join(lines) + "\n")


def update_cfg(updates: dict[str, str]) -> None:
    lines = read_cfg_lines()
    remaining = {key.upper(): value for key, value in updates.items()}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip().upper()
        if key in remaining:
            lines[index] = f"{key}= {remaining.pop(key)}"
    for key, value in remaining.items():
        lines.append(f"{key}= {value}")
    write_cfg_lines(lines)


def set_dv_values(values: list[float]) -> None:
    lines = read_cfg_lines()
    for index, line in enumerate(lines):
        if line.strip().startswith("DV_VALUE"):
            lines[index] = "DV_VALUE= " + ", ".join(f"{value:.16e}" for value in values)
            write_cfg_lines(lines)
            return
    raise RuntimeError("DV_VALUE block not found in config.")


def read_dv_values() -> list[float]:
    for source in (CFG, TEMPLATE_CFG):
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("DV_VALUE"):
                _, rhs = line.split("=", 1)
                entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
                return [float(entry) for entry in entries]
    raise RuntimeError("DV_VALUE block not found in config or template.")


def run_and_log(cmd: list[str], log_handle: TextIO) -> None:
    log_handle.write(f"\n[RUN] {' '.join(cmd)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_su2(cmd_name: str, config_file: str, log_handle: TextIO) -> None:
    run_and_log([cmd_name, config_file], log_handle)


def run_def(config_file: str, log_handle: TextIO) -> None:
    run_and_log(["SU2_DEF", config_file], log_handle)


def snapshot_file(source: Path, target: Path) -> None:
    shutil.copy2(source, target)


def configure_direct(iterations: int, mesh_filename: str, restart_solution: str) -> None:
    update_cfg(
        {
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "YES",
            "SOLUTION_FILENAME": restart_solution,
            "MESH_FILENAME": mesh_filename,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
            "ITER": str(iterations),
            "OBJECTIVE_FUNCTION": "HEAT_RELEASE_GLOBAL",
        }
    )


def configure_adjoint(iterations: int) -> None:
    update_cfg(
        {
            "MATH_PROBLEM": "DISCRETE_ADJOINT",
            "RESTART_SOL": "NO",
            "SOLUTION_FILENAME": BASE_RESTART,
            "SOLUTION_ADJ_FILENAME": BASE_ADJOINT,
            "GRAD_OBJFUNC_FILENAME": DOT_FILE.name,
            "MESH_FILENAME": BASE_MESH,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
            "ITER": str(iterations),
            "OBJECTIVE_FUNCTION": "HEAT_RELEASE_GLOBAL",
        }
    )


def read_history_objective(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = None
        objective_index = None
        last_row = None
        for row in reader:
            if not row:
                continue
            if header is None:
                header = [entry.strip().strip('"') for entry in row]
                normalized = [entry.upper().replace("_", "") for entry in header]
                target = "HEATRELEASEGLOBAL"
                if target not in normalized:
                    raise RuntimeError(f"HEAT_RELEASE_GLOBAL column not found in {path}.")
                objective_index = normalized.index(target)
                continue
            last_row = row
        if last_row is None or objective_index is None:
            raise RuntimeError(f"No data rows found in {path}.")
        return float(last_row[objective_index])


def read_gradient_vector(path: Path) -> list[float]:
    gradients: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or "gradient" in stripped.lower():
            continue
        tokens = stripped.replace(",", " ").split()
        numeric: list[float] = []
        for token in tokens:
            try:
                numeric.append(float(token))
            except ValueError:
                continue
        if numeric:
            gradients.append(numeric[-1])
    if not gradients:
        raise RuntimeError(f"No gradients found in {path}.")
    return gradients


def main() -> None:
    args = parse_args()
    original_cfg = CFG.read_text()
    log_handle = LOG_FILE.open("w", encoding="utf-8")
    try:
        steps = [float(step.strip()) for step in args.steps.split(",") if step.strip()]
        dv_values = read_dv_values()
        if not dv_values:
            raise RuntimeError("No design variables found.")
        restart_solution = Path(args.restart_solution)
        if not restart_solution.exists():
            raise FileNotFoundError(f"Restart solution not found: {restart_solution}")

        baseline_values = [0.0] * len(dv_values)
        set_dv_values(baseline_values)

        log_handle.write("Steady Contini HRR DV0 FD convergence sweep\n")
        log_handle.write(f"Iterations: {args.iter}\n")
        log_handle.write(f"Steps: {steps}\n")
        log_handle.flush()

        configure_direct(args.iter, BASE_MESH, restart_solution.name)
        run_su2("SU2_CFD", CFG.name, log_handle)
        baseline_objective = read_history_objective(Path("history.csv"))
        snapshot_file(Path("history.csv"), BASELINE_HISTORY)

        with FD_GRAD_FILE.open("w", encoding="utf-8") as fd_handle, RESULTS_FILE.open(
            "w", encoding="utf-8", newline=""
        ) as results_handle:
            fd_handle.write("# DV0 FD gradients for steady global heat release\n")
            writer = csv.writer(results_handle)
            writer.writerow(
                [
                    "step",
                    "baseline_objective",
                    "perturbed_objective",
                    "fd_gradient",
                    "adjoint_gradient",
                    "abs_diff",
                    "rel_diff",
                ]
            )

            for step_index, step in enumerate(steps):
                perturbed = baseline_values[:]
                perturbed[0] = step
                set_dv_values(perturbed)

                configure_direct(args.iter, BASE_MESH, restart_solution.name)
                run_def(CFG.name, log_handle)
                configure_direct(args.iter, DEFORMED_MESH, restart_solution.name)
                run_su2("SU2_CFD", CFG.name, log_handle)

                perturbed_objective = read_history_objective(Path("history.csv"))
                snapshot_file(Path("history.csv"), Path(PERTURBED_HISTORY_FMT.format(step_index=step_index)))
                fd_gradient = (perturbed_objective - baseline_objective) / step
                abs_diff = abs(fd_gradient - args.adjoint_dv0)
                rel_diff = abs_diff / max(abs(fd_gradient), abs(args.adjoint_dv0), 1.0e-16)

                fd_handle.write(f"{step:.16e} {fd_gradient:.16e}\n")
                writer.writerow(
                    [
                        f"{step:.16e}",
                        f"{baseline_objective:.16e}",
                        f"{perturbed_objective:.16e}",
                        f"{fd_gradient:.16e}",
                        f"{args.adjoint_dv0:.16e}",
                        f"{abs_diff:.16e}",
                        f"{rel_diff:.16e}",
                    ]
                )
                print(
                    f"step={step:.1e} FD={fd_gradient:.6e} AD={args.adjoint_dv0:.6e} "
                    f"ABS={abs_diff:.3e} REL={rel_diff:.3e}"
                )
                set_dv_values(baseline_values)

        steps_for_plot = [float(step) for step in steps]
        fd_gradients_for_plot = []
        with RESULTS_FILE.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                fd_gradients_for_plot.append(float(row["fd_gradient"]))

        normalized_fd_gradients = [value / args.adjoint_dv0 for value in fd_gradients_for_plot]

        plt.figure(figsize=(7.0, 4.5))
        plt.semilogx(steps_for_plot, normalized_fd_gradients, marker="o", linewidth=2.0, label="FD / AD")
        plt.axhline(1.0, color="crimson", linestyle="--", linewidth=2.0, label="AD reference")
        plt.xlabel("FD step size")
        plt.ylabel("FD gradient / AD gradient")
        plt.title("DV0 steady HRR gradient convergence")
        plt.grid(True, which="both", linestyle=":", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=200)

        log_handle.write(f"\nBaseline objective: {baseline_objective:.16e}\n")
        log_handle.write(f"Adjoint DV0 gradient: {args.adjoint_dv0:.16e}\n")
        log_handle.write(f"FD gradients: {FD_GRAD_FILE}\n")
        log_handle.write(f"Results table: {RESULTS_FILE}\n")
        log_handle.write(f"Plot: {PLOT_FILE}\n")
        log_handle.flush()

    finally:
        CFG.write_text(original_cfg)
        log_handle.close()


if __name__ == "__main__":
    main()
