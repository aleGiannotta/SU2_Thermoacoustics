#!/usr/bin/env python3
"""
Utility to compute finite-difference sensitivities of the DFT objective
by perturbing each DV_VALUE entry one at a time and re-running SU2_CFD.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def parse_dv_block(lines: List[str]) -> Tuple[int, int, List[float]]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("DV_VALUE"):
            start = idx
            break
    if start is None:
        raise RuntimeError("DV_VALUE block not found in config.")

    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped or stripped.startswith("%"):
            break
        end += 1

    block_text = "".join(lines[start:end])
    if "=" not in block_text:
        raise RuntimeError("Failed to parse DV_VALUE block.")
    block_text = block_text.split("=", 1)[1]
    block_text = block_text.replace("\\", " ")
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block_text)
    if not values:
        raise RuntimeError("No DV values detected in DV_VALUE block.")
    dv_values = [float(val) for val in values]
    return start, end, dv_values


def format_dv_block(values: List[float], per_line: int = 5) -> List[str]:
    formatted = []
    for val in values:
        if abs(val) < 1e-16:
            formatted.append("0.0")
        else:
            formatted.append(f"{val:.10f}".rstrip("0").rstrip("."))
    lines: List[str] = []
    for idx in range(0, len(formatted), per_line):
        chunk = ", ".join(formatted[idx:idx + per_line])
        prefix = "DV_VALUE= " if idx == 0 else "          "
        suffix = ", \\\n" if idx + per_line < len(formatted) else "\n"
        lines.append(prefix + chunk + suffix)
    lines.append("\n")
    return lines


def read_dft_value(path: Path, mode: str) -> float:
    if not path.exists():
        raise RuntimeError(f"DFT output file {path} not found.")
    token_name = mode.upper()
    value = None
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.replace(",", " ").split()
            if len(tokens) >= 2 and tokens[0].upper() == token_name:
                try:
                    value = float(tokens[1])
                except ValueError:
                    continue
    if value is None:
        raise RuntimeError(f"No numeric {token_name} found in {path}.")
    return value


def run_su2(executable: str, cfg_path: Path, workdir: Path, mpi_procs: int) -> None:
    log_path = workdir/"main_log.log"
    if mpi_procs > 1:
        cmd = ["mpirun", "-np", str(mpi_procs), executable, str(cfg_path)]
    else:
        cmd = [executable, str(cfg_path)]
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.run(cmd, cwd=workdir, check=True, stdout=log_file, stderr=log_file)


def replace_option(lines: List[str], option: str, value: str) -> List[str]:
    new_lines = []
    replaced = False
    target = option.strip()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(target + "=") or stripped.startswith(target + " ="):
            new_lines.append(f"{target}= {value}\n")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{target}= {value}\n")
    return new_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference check for DFT objective gradients.")
    parser.add_argument("config", type=Path, help="Path to the baseline SU2 config file.")
    parser.add_argument("--su2-cfd", default="SU2_CFD", help="Executable for SU2_CFD runs.")
    parser.add_argument("--su2-def", default="SU2_DEF", help="Executable for SU2_DEF runs.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Perturbation applied to each DV (default: 1e-5).")
    parser.add_argument("--mpi-procs", type=int, default=4, help="Number of MPI ranks to use when launching SU2.")
    parser.add_argument("--mesh-base", default="mesh.su2", help="Original mesh filename.")
    parser.add_argument("--mesh-def", default="mesh_def.su2", help="Deformed mesh filename to run SU2_CFD.")
    parser.add_argument("--objective-mode", choices=["auto", "DFT_AMPLITUDE", "DFT_PHASE"], default="auto",
                        help="DFT objective to compare. Default: read OBJECTIVE_TEMPORAL_MODE from config.")
    parser.add_argument("--log", type=Path, help="Path to log file for partial results.")
    parser.add_argument("--keep-results", action="store_true", help="Keep temporary configs and amplitude copies.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg_dir = config_path.parent
    lines = config_path.read_text().splitlines(True)
    start, end, base_values = parse_dv_block(lines)

    amplitude_file = None
    config_mode = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("OBJECTIVE_DFT_OUTPUT"):
            amplitude_file = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("OBJECTIVE_TEMPORAL_MODE"):
            config_mode = stripped.split("=", 1)[1].strip().upper()
    if amplitude_file is None:
        raise RuntimeError("OBJECTIVE_DFT_OUTPUT not defined in config.")
    if args.objective_mode == "auto":
        if config_mode in {"DFT_AMPLITUDE", "DFT_PHASE"}:
            objective_mode = config_mode
        else:
            raise RuntimeError("OBJECTIVE_TEMPORAL_MODE must be DFT_AMPLITUDE or DFT_PHASE, or pass --objective-mode.")
    else:
        objective_mode = args.objective_mode

    base_amp_path = (cfg_dir / amplitude_file).resolve()
    if args.log:
        log_candidate = args.log if args.log.is_absolute() else (cfg_dir / args.log)
    else:
        log_candidate = cfg_dir / "fd_dft.log"
    log_path = log_candidate.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write(f"# DV  {objective_mode}  FD_Gradient\n")
    log_handle.flush()

    tmp_root = tempfile.mkdtemp(prefix="su2_fd_", dir=cfg_dir)
    tmp_root_path = Path(tmp_root)
    def_mesh = (cfg_dir / args.mesh_def).resolve()

    try:
        base_block = format_dv_block(base_values)
        base_lines = lines[:start] + base_block + lines[end:]
        base_lines = replace_option(base_lines, "OBJECTIVE_DFT_OUTPUT", amplitude_file)

        base_cfg = tmp_root_path / "fd_base.cfg"
        base_cfg.write_text("".join(base_lines))
        run_su2(args.su2_cfd, base_cfg, cfg_dir, args.mpi_procs)
        base_value = read_dft_value(base_amp_path, objective_mode)
        print(f"Baseline {objective_mode}: {base_value:.10e}")

        results = []
        for idx, _ in enumerate(base_values):
            dv_values = base_values.copy()
            dv_values[idx] += args.delta
            new_block = format_dv_block(dv_values)
            dv_lines = lines[:start] + new_block + lines[end:]

            if def_mesh.exists():
                def_mesh.unlink()

            cfg_def = tmp_root_path / f"fd_dv{idx}_def.cfg"
            cfg_def_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_base)
            cfg_def_lines = replace_option(cfg_def_lines, "MESH_OUT_FILENAME", args.mesh_def)
            cfg_def_lines = replace_option(cfg_def_lines, "OBJECTIVE_DFT_OUTPUT", amplitude_file)
            cfg_def.write_text("".join(cfg_def_lines))
            run_su2(args.su2_def, cfg_def, cfg_dir, args.mpi_procs)

            cfg_run = tmp_root_path / f"fd_dv{idx}_run.cfg"
            cfg_run_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_def)
            perturbed_amp_file = f"dft_amplitude_{idx}.dat"
            cfg_run_lines = replace_option(cfg_run_lines, "OBJECTIVE_DFT_OUTPUT", perturbed_amp_file)
            cfg_run.write_text("".join(cfg_run_lines))
            run_su2(args.su2_cfd, cfg_run, cfg_dir, args.mpi_procs)
            amp_file = cfg_dir / perturbed_amp_file
            obj_plus = read_dft_value(amp_file, objective_mode)

            gradient = (obj_plus - base_value) / args.delta
            results.append((idx, obj_plus, gradient))

            log_handle.write(f"{idx:04d} {obj_plus:.10e} {gradient:.10e}\n")
            log_handle.flush()

            if args.keep_results:
                dest = cfg_dir / f"{amplitude_file}.{idx}_plus"
                shutil.copy2(amp_file, dest)
            if def_mesh.exists() and not args.keep_results:
                def_mesh.unlink()

        print(f"{'DV':>4} {objective_mode:>15} {'FD Gradient':>15}")
        for idx, obj_plus, grad in results:
            print(f"{idx:>4d} {obj_plus:>15.8e} {grad:>15.8e}")

    finally:
        log_handle.close()
        if def_mesh.exists():
            def_mesh.unlink()
        shutil.rmtree(tmp_root_path, ignore_errors=True)


if __name__ == "__main__":
    main()
