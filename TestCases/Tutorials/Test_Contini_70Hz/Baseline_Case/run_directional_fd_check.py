#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CFG = Path("contini_flame_unsteady.cfg")
BASELINE_OBJECTIVE_FILE = Path("dft_amplitude.dat")
BASELINE_GRADIENT_FILE = Path("of_grad")
BASELINE_MESH = "mesh_coarse_ffd.su2"
BASELINE_RESTART = "restart_flow_00000.dat"
BASELINE_INLET = "inlet_00001.dat"
RESULTS_FILE = Path("directional_fd_results.csv")
LOG_FILE = Path("directional_fd_runs.log")
SUMMARY_FILE = Path("directional_fd_summary.txt")
PLOT_FILE = Path("directional_fd_convergence.png")
RUNS_DIR = Path("directional_fd_cases")
OBJECTIVE_TOKEN = "DFT_AMPLITUDE"
DEFAULT_ALPHAS = "3,2,1,5e-1,3e-1"
DEFAULT_MAX_DISP = 1.0e-4
STATUS_FILE = Path("directional_fd_status.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centered directional finite-difference check along the normalized adjoint gradient direction."
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=DEFAULT_ALPHAS,
        help="Comma-separated alpha values. Alpha=1 corresponds to the normalized direction itself.",
    )
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=DEFAULT_MAX_DISP,
        help="Maximum absolute DV perturbation at alpha=1 (meters). Default is 0.1 mm.",
    )
    parser.add_argument(
        "--baseline-objective-file",
        type=Path,
        default=BASELINE_OBJECTIVE_FILE,
        help="File containing the baseline DFT objective.",
    )
    parser.add_argument(
        "--gradient-file",
        type=Path,
        default=BASELINE_GRADIENT_FILE,
        help="Adjoint gradient file used to build the search direction.",
    )
    parser.add_argument(
        "--su2-def",
        type=str,
        default="SU2_DEF",
        help="Mesh deformation executable.",
    )
    parser.add_argument(
        "--su2-cfd",
        type=str,
        default="SU2_CFD",
        help="Direct solver executable.",
    )
    parser.add_argument(
        "--restart-solution",
        type=Path,
        default=Path(BASELINE_RESTART),
        help="Baseline restart solution copied into every perturbed case folder.",
    )
    parser.add_argument(
        "--inlet-file",
        type=Path,
        default=Path(BASELINE_INLET),
        help="Baseline inlet profile copied into every perturbed case folder.",
    )
    return parser.parse_args()


def read_cfg_lines() -> list[str]:
    return CFG.read_text().splitlines()


def write_cfg_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def update_cfg_lines(lines: list[str], updates: dict[str, str]) -> list[str]:
    remaining = {key.upper(): value for key, value in updates.items()}
    new_lines = list(lines)
    for index, line in enumerate(new_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip().upper()
        if key in remaining:
            new_lines[index] = f"{key}= {remaining.pop(key)}"
    for key, value in remaining.items():
        new_lines.append(f"{key}= {value}")
    return new_lines


def parse_cfg_value(lines: list[str], key: str) -> str | None:
    key_upper = key.upper()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or "=" not in stripped:
            continue
        line_key, rhs = stripped.split("=", 1)
        if line_key.strip().upper() == key_upper:
            return rhs.strip()
    return None


def parse_parenthesized_tokens(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return [entry.strip() for entry in text.split(",") if entry.strip()]


def set_dv_values(lines: list[str], values: list[float]) -> list[str]:
    new_lines = list(lines)
    for index, line in enumerate(new_lines):
        if line.strip().startswith("DV_VALUE"):
            new_lines[index] = "DV_VALUE= " + ",".join(f"{value:.16e}" for value in values)
            return new_lines
    raise RuntimeError("DV_VALUE block not found in config.")


def read_dv_values(lines: list[str]) -> list[float]:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("DV_VALUE"):
            _, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",") if entry.strip()]
            return [float(entry) for entry in entries]
    raise RuntimeError("DV_VALUE block not found in config.")


def read_objective_value(path: Path, token: str) -> float:
    for line in path.read_text().splitlines():
        pieces = line.strip().split()
        if len(pieces) == 2 and pieces[0].upper() == token.upper():
            return float(pieces[1])
    raise RuntimeError(f"{token} not found in {path}.")


def read_gradient(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or "gradient" in stripped.lower():
            continue
        values.append(float(stripped.split()[-1]))
    if not values:
        raise RuntimeError(f"No gradient entries found in {path}.")
    return values


def collect_local_runtime_assets(lines: list[str], cfg_dir: Path) -> list[Path]:
    assets: list[Path] = []
    interpolator_value = parse_cfg_value(lines, "FILENAMES_INTERPOLATOR")
    if interpolator_value is not None:
        for token in parse_parenthesized_tokens(interpolator_value):
            candidate = cfg_dir / token
            if candidate.exists() and candidate.is_file():
                assets.append(candidate)
    return assets


def copy_if_needed(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def run_and_log(cmd: list[str], log_handle: TextIO, cwd: Path) -> None:
    log_handle.write(f"\n[RUN] {' '.join(cmd)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
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


def write_results_csv(results: list[dict[str, float]], baseline_objective: float, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "run_dir_plus",
                "run_dir_minus",
                "max_dv_perturbation_m",
                "baseline_objective",
                "objective_plus",
                "objective_minus",
                "centered_fd_directional",
                "adjoint_directional",
                "abs_diff",
                "rel_diff",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    f"{row['alpha']:.16e}",
                    row["run_dir_plus"],
                    row["run_dir_minus"],
                    f"{row['max_dv_perturbation_m']:.16e}",
                    f"{baseline_objective:.16e}",
                    f"{row['objective_plus']:.16e}",
                    f"{row['objective_minus']:.16e}",
                    f"{row['centered_fd_directional']:.16e}",
                    f"{row['adjoint_directional']:.16e}",
                    f"{row['abs_diff']:.16e}",
                    f"{row['rel_diff']:.16e}",
                ]
            )


def write_status(
    current_alpha: float | None,
    completed_pairs: int,
    total_pairs: int,
    latest_fd: float | None,
    latest_rel_diff: float | None,
    output_path: Path,
) -> None:
    lines = [
        "# Directional FD live status",
        f"completed_pairs = {completed_pairs}",
        f"total_pairs = {total_pairs}",
    ]
    if current_alpha is not None:
        lines.append(f"current_alpha = {current_alpha:.16e}")
    if latest_fd is not None:
        lines.append(f"latest_centered_fd = {latest_fd:.16e}")
    if latest_rel_diff is not None:
        lines.append(f"latest_rel_diff = {latest_rel_diff:.16e}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_live_plot(results: list[dict[str, float]], adjoint_directional: float, plot_path: Path) -> None:
    if not results:
        return

    alphas = [row["alpha"] for row in results]
    objective_plus = [row["objective_plus"] for row in results]
    objective_minus = [row["objective_minus"] for row in results]
    centered = [row["centered_fd_directional"] for row in results]
    rel_diff = [row["rel_diff"] for row in results]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(alphas, objective_plus, marker="o", lw=1.2, color="tab:blue", label="J(+alpha)")
    axes[0].plot(alphas, objective_minus, marker="s", lw=1.2, color="tab:orange", label="J(-alpha)")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Objective")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(adjoint_directional, color="tab:red", ls="--", lw=1.2, label="Adjoint")
    axes[1].plot(alphas, centered, marker="o", lw=1.2, color="tab:green", label="Centered FD")
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

    fig.suptitle("Directional FD Convergence")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_case(
    run_dir: Path,
    tag: str,
    signed_alpha: float,
    base_lines: list[str],
    base_dv_values: list[float],
    direction: list[float],
    baseline_restart: Path,
    baseline_inlet: Path,
    restart_cfg_name: str,
    inlet_cfg_name: str,
    runtime_assets: list[Path],
    su2_def: str,
    su2_cfd: str,
    log_handle: TextIO,
) -> float:
    mesh_out = Path("mesh_dirfd.su2")
    def_cfg = Path("directional_def.cfg")
    direct_cfg = Path("directional_direct.cfg")
    conv_name = "history_dirfd"
    restart_name = "restart_flow_dirfd.dat"
    volume_name = "flow_dirfd"
    dft_name = Path("dft_amplitude_dirfd.dat")
    value_name = "of_func_dirfd"

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG, run_dir / CFG.name)
    shutil.copy2(CFG.parent / BASELINE_MESH, run_dir / BASELINE_MESH)
    shutil.copy2(baseline_restart, run_dir / BASELINE_RESTART)
    shutil.copy2(baseline_inlet, run_dir / BASELINE_INLET)
    copy_if_needed(baseline_restart, run_dir / restart_cfg_name)
    copy_if_needed(baseline_inlet, run_dir / inlet_cfg_name)
    for asset in runtime_assets:
        shutil.copy2(asset, run_dir / asset.name)

    perturbed_values = [base + signed_alpha * delta for base, delta in zip(base_dv_values, direction)]

    def_lines = set_dv_values(base_lines, perturbed_values)
    def_lines = update_cfg_lines(
        def_lines,
        {
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "MESH_FILENAME": BASELINE_MESH,
            "MESH_OUT_FILENAME": mesh_out.name,
        },
    )
    write_cfg_lines(run_dir / def_cfg, def_lines)
    print(f"[{tag}] deforming mesh in {run_dir}", flush=True)
    run_and_log([su2_def, def_cfg.name], log_handle, run_dir)

    direct_lines = set_dv_values(base_lines, perturbed_values)
    direct_lines = update_cfg_lines(
        direct_lines,
        {
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "YES",
            "MESH_FILENAME": mesh_out.name,
            "MESH_OUT_FILENAME": mesh_out.name,
            "SOLUTION_FILENAME": restart_cfg_name,
            "INLET_FILENAME": inlet_cfg_name,
            "CONV_FILENAME": conv_name,
            "RESTART_FILENAME": restart_name,
            "VOLUME_FILENAME": volume_name,
            "OBJECTIVE_DFT_OUTPUT": dft_name.name,
            "VALUE_OBJFUNC_FILENAME": value_name,
            "OUTPUT_FILES": "(RESTART)",
        },
    )
    write_cfg_lines(run_dir / direct_cfg, direct_lines)
    print(f"[{tag}] running direct solve in {run_dir}", flush=True)
    run_and_log([su2_cfd, direct_cfg.name], log_handle, run_dir)
    return read_objective_value(run_dir / dft_name, OBJECTIVE_TOKEN)


def main() -> None:
    args = parse_args()
    base_lines = read_cfg_lines()
    base_dv_values = read_dv_values(base_lines)
    gradient = read_gradient(args.gradient_file)
    baseline_objective = read_objective_value(args.baseline_objective_file, OBJECTIVE_TOKEN)
    baseline_restart = args.restart_solution
    baseline_inlet = args.inlet_file
    restart_cfg_name = parse_cfg_value(base_lines, "SOLUTION_FILENAME")
    inlet_cfg_name = parse_cfg_value(base_lines, "INLET_FILENAME")
    runtime_assets = collect_local_runtime_assets(base_lines, CFG.parent)

    if not baseline_restart.exists():
        raise FileNotFoundError(f"Baseline restart solution not found: {baseline_restart}")
    if not baseline_inlet.exists():
        raise FileNotFoundError(f"Baseline inlet file not found: {baseline_inlet}")
    if restart_cfg_name is None:
        raise RuntimeError("SOLUTION_FILENAME not found in config.")
    if inlet_cfg_name is None:
        raise RuntimeError("INLET_FILENAME not found in config.")

    if len(base_dv_values) != len(gradient):
        raise RuntimeError(
            f"DV count mismatch: config has {len(base_dv_values)} values but {args.gradient_file} has {len(gradient)}."
        )

    alphas = [float(entry.strip()) for entry in args.alphas.split(",") if entry.strip()]
    if not alphas:
        raise RuntimeError("No alpha values provided.")

    inf_norm = max(abs(value) for value in gradient)
    if inf_norm == 0.0:
        raise RuntimeError("Adjoint gradient is zero; cannot build a normalized direction.")

    direction = [args.max_displacement * value / inf_norm for value in gradient]
    adjoint_directional = sum(g_i * d_i for g_i, d_i in zip(gradient, direction))
    max_abs_direction = max(abs(value) for value in direction)

    results: list[dict[str, float]] = []
    RUNS_DIR.mkdir(exist_ok=True)
    write_status(None, 0, len(alphas), None, None, STATUS_FILE)
    with LOG_FILE.open("w", encoding="utf-8") as log_handle:
        log_handle.write("Centered directional FD check along normalized adjoint gradient\n")
        log_handle.write(f"Baseline objective: {baseline_objective:.16e}\n")
        log_handle.write(f"Gradient file: {args.gradient_file}\n")
        log_handle.write(f"Baseline restart: {baseline_restart}\n")
        log_handle.write(f"Baseline inlet: {baseline_inlet}\n")
        log_handle.write(f"Restart cfg name: {restart_cfg_name}\n")
        log_handle.write(f"Inlet cfg name: {inlet_cfg_name}\n")
        if runtime_assets:
            log_handle.write("Runtime assets:\n")
            for asset in runtime_assets:
                log_handle.write(f"  - {asset}\n")
        log_handle.write(f"Max DV displacement at alpha=1: {args.max_displacement:.16e} m\n")
        log_handle.write(f"Adjoint directional derivative: {adjoint_directional:.16e}\n")
        log_handle.flush()

        for step_index, alpha in enumerate(alphas):
            tag = f"a{step_index:02d}"
            run_dir_plus = RUNS_DIR / f"{tag}_plus"
            run_dir_minus = RUNS_DIR / f"{tag}_minus"
            print(
                f"\n========== alpha {alpha:.6e} ({step_index + 1}/{len(alphas)}) ==========",
                flush=True,
            )
            print(
                f"\n[alpha {alpha:.6e}] running +alpha case in {run_dir_plus}",
                flush=True,
            )
            objective_plus = run_case(
                run_dir_plus,
                f"{tag}_plus",
                alpha,
                base_lines,
                base_dv_values,
                direction,
                baseline_restart,
                baseline_inlet,
                restart_cfg_name,
                inlet_cfg_name,
                runtime_assets,
                args.su2_def,
                args.su2_cfd,
                log_handle,
            )
            print(
                f"[alpha {alpha:.6e}] J(+alpha) = {objective_plus:.16e}",
                flush=True,
            )
            print(
                f"[alpha {alpha:.6e}] running -alpha case in {run_dir_minus}",
                flush=True,
            )
            objective_minus = run_case(
                run_dir_minus,
                f"{tag}_minus",
                -alpha,
                base_lines,
                base_dv_values,
                direction,
                baseline_restart,
                baseline_inlet,
                restart_cfg_name,
                inlet_cfg_name,
                runtime_assets,
                args.su2_def,
                args.su2_cfd,
                log_handle,
            )
            print(
                f"[alpha {alpha:.6e}] J(-alpha) = {objective_minus:.16e}",
                flush=True,
            )
            fd_directional = (objective_plus - objective_minus) / (2.0 * alpha)
            abs_diff = abs(fd_directional - adjoint_directional)
            rel_diff = abs_diff / max(abs(adjoint_directional), 1.0e-30)
            max_dv_perturb = max(abs(alpha * value) for value in direction)

            results.append(
                {
                    "alpha": alpha,
                    "run_dir_plus": str(run_dir_plus),
                    "run_dir_minus": str(run_dir_minus),
                    "max_dv_perturbation_m": max_dv_perturb,
                    "objective_plus": objective_plus,
                    "objective_minus": objective_minus,
                    "centered_fd_directional": fd_directional,
                    "adjoint_directional": adjoint_directional,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
            write_results_csv(results, baseline_objective, RESULTS_FILE)
            update_live_plot(results, adjoint_directional, PLOT_FILE)
            write_status(alpha, len(results), len(alphas), fd_directional, rel_diff, STATUS_FILE)
            print(
                f"[alpha {alpha:.6e}] centered FD = {fd_directional:.16e}, "
                f"adjoint = {adjoint_directional:.16e}, rel diff = {rel_diff:.16e}",
                flush=True,
            )
            print(f"[alpha {alpha:.6e}] updated plot: {PLOT_FILE}", flush=True)
            print(f"[alpha {alpha:.6e}] updated csv: {RESULTS_FILE}", flush=True)

    write_results_csv(results, baseline_objective, RESULTS_FILE)

    with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        handle.write("# Directional FD setup\n")
        handle.write(f"gradient_file = {args.gradient_file}\n")
        handle.write(f"baseline_objective_file = {args.baseline_objective_file}\n")
        handle.write(f"baseline_restart = {baseline_restart}\n")
        handle.write(f"baseline_inlet = {baseline_inlet}\n")
        handle.write(f"baseline_objective = {baseline_objective:.16e}\n")
        handle.write(f"max_displacement_alpha_1_m = {args.max_displacement:.16e}\n")
        handle.write(f"max_abs_direction_component_m = {max_abs_direction:.16e}\n")
        handle.write(f"adjoint_directional = {adjoint_directional:.16e}\n")
        handle.write("direction_components_m = " + ", ".join(f"{value:.16e}" for value in direction) + "\n")

    print(f"Baseline objective: {baseline_objective:.16e}")
    print(f"Adjoint directional derivative: {adjoint_directional:.16e}")
    print(f"Max |direction component| at alpha=1: {max_abs_direction:.16e} m")
    print(f"Wrote: {RESULTS_FILE}")
    print(f"Wrote: {SUMMARY_FILE}")
    print(f"Wrote: {LOG_FILE}")
    print(f"Wrote: {PLOT_FILE}")


if __name__ == "__main__":
    main()
