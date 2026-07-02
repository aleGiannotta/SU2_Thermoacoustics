#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import TextIO

import matplotlib.pyplot as plt


CFG = Path("contini_flame_unsteady.cfg")
TEMPLATE_CFG = Path("contini_flame_unsteady.cfg")
LOG_FILE = Path("dv0_dft_amplitude_fd_convergence.log")
RESULTS_FILE = Path("dv0_dft_amplitude_fd_convergence.csv")
PERTURBED_HISTORY_FMT = "dft_amplitude_step_{step_index:02d}.dat"
FD_GRAD_FILE = Path("of_grad_dv0_dft_amplitude_fd.dat")
PLOT_FILE = Path("dv0_dft_amplitude_fd_convergence.png")
BASELINE_OBJECTIVE_FILE = Path("dft_amplitude_baseline.dat")
BASE_RESTART = "restart_flow.dat"
BASE_MESH = "mesh_coarse_ffd.su2"
DEFORMED_MESH = "mesh_out.su2"
DOT_FILE = Path("of_grad")
BASE_ADJOINT = "restart_adj"
DEFAULT_ITER = 40
DEFAULT_INNER_ITER = 10000
DEFAULT_STEPS = "1e-4,3e-5,1e-5,1e-6"
OBJECTIVE_TOKEN = "DFT_AMPLITUDE"
OBJECTIVE_FILE = Path("dft_amplitude.dat")
DEFAULT_WINDOW_SIZE = 20
DEFAULT_WINDOW_START = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FD step sizes for DV 0 of the unsteady Contini DFT amplitude objective."
    )
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS, help="Comma-separated FD step sizes.")
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER, help="Unsteady time-iteration limit.")
    parser.add_argument("--inner-iter", type=int, default=DEFAULT_INNER_ITER, help="Inner iterations per time step.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of time samples used by the DFT objective window.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=DEFAULT_WINDOW_START,
        help="First time iteration included in the DFT objective window.",
    )
    parser.add_argument(
        "--restart-solution",
        type=str,
        default=BASE_RESTART,
        help="Restart solution used as the primal starting point.",
    )
    parser.add_argument(
        "--skip-baseline-direct",
        action="store_true",
        help="Reuse an existing baseline DFT file and start the workflow from the adjoint run.",
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


def configure_direct_run(
    iterations: int,
    inner_iter: int,
    mesh_filename: str,
    restart_solution: str,
    window_start: int,
    window_size: int,
) -> None:
    update_cfg(
        {
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "YES",
            "SOLUTION_FILENAME": restart_solution,
            "MESH_FILENAME": mesh_filename,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
            "OBJECTIVE_TEMPORAL_MODE": "DFT_AMPLITUDE",
            "OBJECTIVE_DFT_OUTPUT": OBJECTIVE_FILE.name,
            "TIME_ITER": str(iterations),
            "UNST_ADJOINT_ITER": str(iterations),
            "WINDOW_START_ITER": str(window_start),
            "ITER_AVERAGE_OBJ": str(window_size),
            "INNER_ITER": str(inner_iter),
        }
    )


def read_objective_value(path: Path, token: str) -> float:
    for line in path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) == 2 and tokens[0].upper() == token.upper():
            return float(tokens[1])
    raise RuntimeError(f"{token} not found in {path}.")


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


def configure_adjoint_run(
    iterations: int,
    inner_iter: int,
    restart_solution: str,
    window_start: int,
    window_size: int,
) -> None:
    update_cfg(
        {
            "MATH_PROBLEM": "DISCRETE_ADJOINT",
            "RESTART_SOL": "NO",
            "SOLUTION_FILENAME": restart_solution,
            "SOLUTION_ADJ_FILENAME": BASE_ADJOINT,
            "GRAD_OBJFUNC_FILENAME": DOT_FILE.name,
            "MESH_FILENAME": BASE_MESH,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
            "OBJECTIVE_TEMPORAL_MODE": "DFT_AMPLITUDE",
            "OBJECTIVE_DFT_OUTPUT": OBJECTIVE_FILE.name,
            "TIME_ITER": str(iterations),
            "UNST_ADJOINT_ITER": str(iterations),
            "WINDOW_START_ITER": str(window_start),
            "ITER_AVERAGE_OBJ": str(window_size),
            "INNER_ITER": str(inner_iter),
        }
    )


def pick_largest_gradient(gradients: list[float]) -> tuple[int, float]:
    best_index = max(range(len(gradients)), key=lambda idx: abs(gradients[idx]))
    return best_index, gradients[best_index]


def main() -> None:
    args = parse_args()
    original_cfg = CFG.read_text()
    log_handle = LOG_FILE.open("w", encoding="utf-8")
    try:
        restart_solution = Path(args.restart_solution)
        if not restart_solution.exists():
            raise FileNotFoundError(f"Restart solution not found: {restart_solution}")
        if args.window_size <= 0:
            raise ValueError("--window-size must be positive.")
        if args.window_start < 0:
            raise ValueError("--window-start must be non-negative.")
        if args.window_start + args.window_size > args.iter:
            raise ValueError("The DFT window must fit inside the simulated time range.")

        dv_values = read_dv_values()
        dv_count = len(dv_values)
        steps = [float(step.strip()) for step in args.steps.split(",") if step.strip()]

        log_handle.write("Unsteady Contini DFT amplitude DV0 FD convergence sweep\n")
        log_handle.write(f"Iterations: {args.iter}\n")
        log_handle.write(f"Inner iterations: {args.inner_iter}\n")
        log_handle.write(f"Window start: {args.window_start}\n")
        log_handle.write(f"Window size: {args.window_size}\n")
        log_handle.write(f"Steps: {steps}\n")
        log_handle.write(f"Restart solution: {restart_solution}\n")
        log_handle.write(f"Skip baseline direct: {args.skip_baseline_direct}\n")
        log_handle.flush()

        baseline_values = [0.0] * dv_count
        set_dv_values(baseline_values)

        if args.skip_baseline_direct:
            baseline_source = BASELINE_OBJECTIVE_FILE if BASELINE_OBJECTIVE_FILE.exists() else OBJECTIVE_FILE
            if not baseline_source.exists():
                raise FileNotFoundError(
                    f"Baseline DFT file not found. Expected {BASELINE_OBJECTIVE_FILE} or {OBJECTIVE_FILE}."
                )
            baseline_objective = read_objective_value(baseline_source, OBJECTIVE_TOKEN)
            if baseline_source != OBJECTIVE_FILE:
                snapshot_file(baseline_source, OBJECTIVE_FILE)
            log_handle.write("\n=== Reusing existing baseline direct data ===\n")
            log_handle.write(f"Baseline DFT source: {baseline_source}\n")
            log_handle.write(f"Baseline DFT amplitude: {baseline_objective:.16e}\n")
            log_handle.flush()
        else:
            log_handle.write("\n=== Baseline direct run ===\n")
            log_handle.flush()
            configure_direct_run(
                args.iter,
                args.inner_iter,
                BASE_MESH,
                restart_solution.name,
                args.window_start,
                args.window_size,
            )
            run_su2("SU2_CFD", CFG.name, log_handle)
            baseline_objective = read_objective_value(OBJECTIVE_FILE, OBJECTIVE_TOKEN)
            snapshot_file(OBJECTIVE_FILE, BASELINE_OBJECTIVE_FILE)
            log_handle.write(f"Baseline DFT amplitude: {baseline_objective:.16e}\n")
            log_handle.flush()

        log_handle.write("\n=== Baseline adjoint run ===\n")
        log_handle.flush()
        configure_adjoint_run(
            args.iter,
            args.inner_iter,
            restart_solution.name,
            args.window_start,
            args.window_size,
        )
        run_su2("SU2_CFD_AD", CFG.name, log_handle)
        run_su2("SU2_DOT_AD", CFG.name, log_handle)
        adjoint_gradients = read_gradient_vector(DOT_FILE)
        if not adjoint_gradients:
            raise RuntimeError(f"No adjoint gradients found in {DOT_FILE}.")
        target_dv, adjoint_target = pick_largest_gradient(adjoint_gradients)
        log_handle.write(f"Selected DV index: {target_dv}\n")
        log_handle.write(f"Adjoint selected gradient: {adjoint_target:.16e}\n")
        log_handle.flush()

        fd_gradients: list[float] = []
        perturbed_objectives: list[float] = []
        fd_grad_path = FD_GRAD_FILE.open("w", encoding="utf-8")
        fd_grad_path.write("# FD gradients for the unsteady Contini DFT amplitude objective\n")
        fd_grad_path.write(f"# Selected DV index: {target_dv}\n")
        fd_grad_path.write(f"# Adjoint reference: {adjoint_target:.16e}\n")
        fd_grad_path.flush()

        try:
            for step_index, step in enumerate(steps):
                log_handle.write(f"\n=== FD perturbation step {step:.6e} ===\n")
                log_handle.flush()

                perturbed = baseline_values[:]
                perturbed[target_dv] += step
                set_dv_values(perturbed)

                configure_direct_run(
                    args.iter,
                    args.inner_iter,
                    BASE_MESH,
                    restart_solution.name,
                    args.window_start,
                    args.window_size,
                )
                run_def(CFG.name, log_handle)

                configure_direct_run(
                    args.iter,
                    args.inner_iter,
                    DEFORMED_MESH,
                    restart_solution.name,
                    args.window_start,
                    args.window_size,
                )
                run_su2("SU2_CFD", CFG.name, log_handle)

                perturbed_objective = read_objective_value(OBJECTIVE_FILE, OBJECTIVE_TOKEN)
                snapshot_file(OBJECTIVE_FILE, Path(PERTURBED_HISTORY_FMT.format(step_index=step_index)))

                fd_gradient = (perturbed_objective - baseline_objective) / step
                fd_gradients.append(fd_gradient)
                perturbed_objectives.append(perturbed_objective)
                abs_diff = abs(fd_gradient - adjoint_target)
                rel_diff = abs_diff / max(abs(fd_gradient), abs(adjoint_target), 1.0e-16)

                fd_grad_path.write(f"{step:.16e} {fd_gradient:.16e}\n")
                fd_grad_path.flush()

                log_handle.write(
                    f"step={step:.6e} objective={perturbed_objective:.16e} "
                    f"FD={fd_gradient:.16e} AD={adjoint_target:.16e} "
                    f"ABS={abs_diff:.3e} REL={rel_diff:.3e}\n"
                )
                log_handle.flush()
                print(
                    f"DV={target_dv} step={step:.1e} FD={fd_gradient:.6e} AD={adjoint_target:.6e} "
                    f"ABS={abs_diff:.3e} REL={rel_diff:.3e}"
                )

                set_dv_values(baseline_values)

        finally:
            fd_grad_path.close()

        with RESULTS_FILE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "dv_index",
                    "baseline_objective",
                    "perturbed_objective",
                    "fd_gradient",
                    "adjoint_gradient",
                    "abs_diff",
                    "rel_diff",
                ]
            )
            for step, fd_gradient, perturbed_objective in zip(steps, fd_gradients, perturbed_objectives):
                abs_diff = abs(fd_gradient - adjoint_target)
                rel_diff = abs_diff / max(abs(fd_gradient), abs(adjoint_target), 1.0e-16)
                writer.writerow(
                    [
                        f"{step:.16e}",
                        str(target_dv),
                        f"{baseline_objective:.16e}",
                        f"{perturbed_objective:.16e}",
                        f"{fd_gradient:.16e}",
                        f"{adjoint_target:.16e}",
                        f"{abs_diff:.16e}",
                        f"{rel_diff:.16e}",
                    ]
                )

        normalized_fd_gradients = [value / adjoint_target for value in fd_gradients]
        plt.figure(figsize=(7.0, 4.5))
        plt.semilogx(steps, normalized_fd_gradients, marker="o", linewidth=2.0, label="FD / AD")
        plt.axhline(1.0, color="crimson", linestyle="--", linewidth=2.0, label="AD reference")
        plt.xlabel("FD step size")
        plt.ylabel("FD gradient / AD gradient")
        plt.title(f"DV{target_dv} unsteady DFT amplitude gradient convergence")
        plt.grid(True, which="both", linestyle=":", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=200)

        log_handle.write(f"\nBaseline objective: {baseline_objective:.16e}\n")
        log_handle.write(f"Selected DV index: {target_dv}\n")
        log_handle.write(f"Adjoint selected gradient: {adjoint_target:.16e}\n")
        log_handle.write(f"FD gradients: {FD_GRAD_FILE}\n")
        log_handle.write(f"Results table: {RESULTS_FILE}\n")
        log_handle.write(f"Plot: {PLOT_FILE}\n")
        log_handle.flush()

    finally:
        CFG.write_text(original_cfg)
        log_handle.close()


if __name__ == "__main__":
    main()
