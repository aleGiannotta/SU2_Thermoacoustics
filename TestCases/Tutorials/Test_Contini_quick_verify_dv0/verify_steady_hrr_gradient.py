#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import TextIO


CFG = Path("contini_flame.cfg")
DV_TEMPLATE_CFG = Path("contini_flame_unsteady.cfg")
LOG_FILE = Path("steady_hrr_gradient.log")
FD_GRAD_FILE = Path("of_grad_FD_steady.dat")
DOT_FILE = Path("of_grad_steady")
COMPARISON_FILE = Path("gradient_comparison_steady.csv")
SUMMARY_FILE = Path("steady_hrr_summary.txt")
PERTURBED_HISTORY_FMT = "history_steady_dv{index:04d}_plus.csv"
BASELINE_HISTORY_SNAPSHOT = Path("history_steady_baseline.csv")
BASE_MESH = "mesh_coarse_ffd.su2"
DEFORMED_MESH = "mesh_out.su2"
BASE_RESTART = "restart_flow"
BASE_ADJOINT = "restart_adj"
DEFAULT_STEP = 1.0e-5
DEFAULT_ITER = 1000
DEFAULT_DV_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steady FD/adjoint check for the Contini flame HRR objective.")
    parser.add_argument("--fd-step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER)
    parser.add_argument("--dv-indices", type=str, default=str(DEFAULT_DV_INDEX))
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
            _, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
            if len(entries) != len(values):
                raise RuntimeError(f"DV_VALUE has {len(entries)} entries, expected {len(values)}.")
            lines[index] = "DV_VALUE= " + ", ".join(f"{value:.16e}" for value in values)
            write_cfg_lines(lines)
            return
    lines.append("DV_VALUE= " + ", ".join(f"{value:.16e}" for value in values))
    write_cfg_lines(lines)


def read_dv_values() -> list[float]:
    for line in read_cfg_lines():
        stripped = line.strip()
        if stripped.startswith("DV_VALUE"):
            _, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
            return [float(entry) for entry in entries]
    for line in DV_TEMPLATE_CFG.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("DV_VALUE"):
            _, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
            return [float(entry) for entry in entries]
    raise RuntimeError("DV_VALUE block not found in config or template.")


def run_and_log(cmd: list[str], log_handle: TextIO) -> None:
    command_line = " ".join(cmd)
    log_handle.write(f"\n[RUN] {command_line}\n")
    log_handle.flush()
    print(f"[RUN] {command_line}", flush=True)
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


def configure_direct_run(iterations: int, mesh_filename: str) -> None:
    update_cfg(
        {
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "YES",
            "MESH_FILENAME": mesh_filename,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
            "ITER": str(iterations),
            "OBJECTIVE_FUNCTION": "HEAT_RELEASE_GLOBAL",
        }
    )


def configure_adjoint_run(iterations: int) -> None:
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
        last_row: list[str] | None = None
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
        dv_values = read_dv_values()
        dv_count = len(dv_values)
        dv_indices = [int(item.strip()) for item in args.dv_indices.split(",") if item.strip()]

        log_handle.write("Steady Contini HRR gradient check\n")
        log_handle.write(f"Config: {CFG.name}\n")
        log_handle.write(f"FD step: {args.fd_step:.6e}\n")
        log_handle.write(f"Iterations: {args.iter}\n")
        log_handle.write(f"DV indices: {dv_indices}\n")
        log_handle.flush()

        baseline_values = [0.0] * dv_count
        set_dv_values(baseline_values)

        log_handle.write("\n=== Baseline direct run ===\n")
        log_handle.flush()
        configure_direct_run(args.iter, BASE_MESH)
        run_su2("SU2_CFD", CFG.name, log_handle)
        baseline_objective = read_history_objective(Path("history.csv"))
        snapshot_file(Path("history.csv"), BASELINE_HISTORY_SNAPSHOT)
        log_handle.write(f"Baseline HEAT_RELEASE_GLOBAL: {baseline_objective:.16e}\n")
        log_handle.write(f"Baseline history snapshot: {BASELINE_HISTORY_SNAPSHOT}\n")
        log_handle.flush()

        log_handle.write("\n=== Baseline adjoint run ===\n")
        log_handle.flush()
        configure_adjoint_run(args.iter)
        run_su2("SU2_CFD_AD", CFG.name, log_handle)

        log_handle.write("\n=== DOT projection ===\n")
        log_handle.flush()
        run_su2("SU2_DOT_AD", CFG.name, log_handle)
        adjoint_gradients = read_gradient_vector(DOT_FILE)

        if max(dv_indices, default=-1) >= len(adjoint_gradients):
            raise RuntimeError(
                f"Adjoint gradient file {DOT_FILE} has {len(adjoint_gradients)} entries, "
                f"but dv_indices requests up to {max(dv_indices)}."
            )

        fd_gradients: list[float] = []
        perturbed_objectives: list[float] = []
        perturbed_history_snapshots: list[Path] = []
        fd_grad_path = FD_GRAD_FILE.open("w", encoding="utf-8")
        fd_grad_path.write("# Steady FD gradients for the Contini HRR objective\n")
        fd_grad_path.write(f"# FD step: {args.fd_step:.6e}\n")
        fd_grad_path.flush()

        try:
            for index in dv_indices:
                log_handle.write(f"\n=== FD perturbation DV {index:04d} ===\n")
                log_handle.flush()

                perturbed = baseline_values[:]
                perturbed[index] += args.fd_step
                set_dv_values(perturbed)

                configure_direct_run(args.iter, BASE_MESH)
                run_def(CFG.name, log_handle)

                configure_direct_run(args.iter, DEFORMED_MESH)
                run_su2("SU2_CFD", CFG.name, log_handle)

                perturbed_objective = read_history_objective(Path("history.csv"))
                perturbed_objectives.append(perturbed_objective)
                perturbed_snapshot = Path(PERTURBED_HISTORY_FMT.format(index=index))
                snapshot_file(Path("history.csv"), perturbed_snapshot)
                perturbed_history_snapshots.append(perturbed_snapshot)

                fd_gradient = (perturbed_objective - baseline_objective) / args.fd_step
                fd_gradients.append(fd_gradient)
                fd_grad_path.write(f"{index:04d} {fd_gradient:.12e}\n")
                fd_grad_path.flush()

                adjoint_gradient = adjoint_gradients[index]
                abs_diff = abs(fd_gradient - adjoint_gradient)
                rel_diff = abs_diff / max(abs(fd_gradient), abs(adjoint_gradient), 1.0e-16)

                log_handle.write(
                    f"DV {index:04d}: objective={perturbed_objective:.16e} "
                    f"FD={fd_gradient:.16e} AD={adjoint_gradient:.16e} "
                    f"ABS={abs_diff:.3e} REL={rel_diff:.3e}\n"
                )
                log_handle.write(f"DV {index:04d} history snapshot: {perturbed_snapshot}\n")
                log_handle.flush()
                print(
                    f"DV {index:04d}: objective={perturbed_objective:.16e} "
                    f"FD={fd_gradient:.16e} AD={adjoint_gradient:.16e} "
                    f"ABS={abs_diff:.3e} REL={rel_diff:.3e}"
                )

                set_dv_values(baseline_values)

        finally:
            fd_grad_path.close()

        with COMPARISON_FILE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "dv_index",
                    "baseline_history_snapshot",
                    "perturbed_history_snapshot",
                    "baseline_objective",
                    "perturbed_objective",
                    "fd_gradient",
                    "adjoint_gradient",
                    "abs_diff",
                    "rel_diff",
                ]
            )
            for index, fd_gradient, perturbed_objective, perturbed_snapshot in zip(
                dv_indices, fd_gradients, perturbed_objectives, perturbed_history_snapshots
            ):
                adjoint_gradient = adjoint_gradients[index]
                abs_diff = abs(fd_gradient - adjoint_gradient)
                rel_diff = abs_diff / max(abs(fd_gradient), abs(adjoint_gradient), 1.0e-16)
                writer.writerow(
                    [
                        index,
                        BASELINE_HISTORY_SNAPSHOT.name,
                        perturbed_snapshot.name,
                        f"{baseline_objective:.16e}",
                        f"{perturbed_objective:.16e}",
                        f"{fd_gradient:.16e}",
                        f"{adjoint_gradient:.16e}",
                        f"{abs_diff:.16e}",
                        f"{rel_diff:.16e}",
                    ]
                )

        SUMMARY_FILE.write_text(
            "\n".join(
                [
                    f"baseline_objective={baseline_objective:.16e}",
                    f"adjoint_grad0={adjoint_gradients[dv_indices[0]]:.16e}" if dv_indices else "adjoint_grad0=NA",
                    f"fd_grad0={fd_gradients[0]:.16e}" if fd_gradients else "fd_grad0=NA",
                    f"comparison_file={COMPARISON_FILE}",
                ]
            )
            + "\n"
        )

        log_handle.write(f"\nBaseline objective: {baseline_objective:.16e}\n")
        log_handle.write(f"Adjoint gradients read from: {DOT_FILE}\n")
        log_handle.write(f"FD gradients written to: {FD_GRAD_FILE}\n")
        log_handle.write(f"Comparison table written to: {COMPARISON_FILE}\n")
        log_handle.flush()

    finally:
        CFG.write_text(original_cfg)
        log_handle.close()


if __name__ == "__main__":
    main()
