#!/usr/bin/env python3
"""
Utility to compute finite-difference sensitivities of the DFT amplitude objective
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
    formatted = [f"{val:.10e}".rstrip("0").rstrip(".") for val in values]
    lines: List[str] = []
    for idx in range(0, len(formatted), per_line):
        chunk = ", ".join(formatted[idx:idx + per_line])
        prefix = "DV_VALUE= " if idx == 0 else "          "
        suffix = ", \\\n" if idx + per_line < len(formatted) else "\n"
        lines.append(prefix + chunk + suffix)
    lines.append("\n")
    return lines


def read_amplitude_file(path: Path) -> float:
    if not path.exists():
        raise RuntimeError(f"Amplitude file {path} not found.")
    value = None
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.replace(",", " ").split()
            if len(tokens) >= 2 and tokens[0].upper() == "DFT_AMPLITUDE":
                try:
                    value = float(tokens[1])
                except ValueError:
                    continue
    if value is None:
        raise RuntimeError(f"No numeric amplitude found in {path}.")
    return value


def run_su2(executable: str, cfg_path: Path, workdir: Path, mpi_procs: int) -> None:
    if mpi_procs > 1:
        cmd = ["mpirun", "-np", str(mpi_procs), executable, str(cfg_path)]
    else:
        cmd = [executable, str(cfg_path)]
    subprocess.run(cmd, cwd=workdir, check=True)


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
    parser = argparse.ArgumentParser(description="Finite-difference check for DFT amplitude gradients.")
    parser.add_argument("config", type=Path, help="Path to the baseline SU2 config file.")
    parser.add_argument("--su2-cfd", default="SU2_CFD", help="Executable for SU2_CFD runs.")
    parser.add_argument("--su2-def", default="SU2_DEF", help="Executable for SU2_DEF runs.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Perturbation applied to each DV (default: 1e-5).")
    parser.add_argument("--mpi-procs", type=int, default=1, help="Number of MPI ranks to use when launching SU2.")
    parser.add_argument("--mesh-base", default="mesh.su2", help="Original mesh filename.")
    parser.add_argument("--mesh-scratch", default="mesh_tmp.su2", help="Scratch mesh filename for deformations.")
    parser.add_argument("--log", type=Path, help="Path to log file for partial results.")
    parser.add_argument("--keep-results", action="store_true", help="Keep temporary configs and amplitude copies.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg_dir = config_path.parent
    lines = config_path.read_text().splitlines(True)
    start, end, base_values = parse_dv_block(lines)
    zero_values = [0.0] * len(base_values)

    amplitude_file = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("OBJECTIVE_DFT_OUTPUT"):
            amplitude_file = stripped.split("=", 1)[1].strip()
            break
    if amplitude_file is None:
        raise RuntimeError("OBJECTIVE_DFT_OUTPUT not defined in config.")

    amp_path = (cfg_dir / amplitude_file).resolve()
    if args.log:
        log_candidate = args.log if args.log.is_absolute() else (cfg_dir / args.log)
    else:
        log_candidate = cfg_dir / "fd_dft.log"
    log_path = log_candidate.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write("# DV  Amplitude  FD_Gradient\n")
    log_handle.flush()

    tmp_root = tempfile.mkdtemp(prefix="su2_fd_", dir=cfg_dir)
    tmp_root_path = Path(tmp_root)
    scratch_mesh = (cfg_dir / args.mesh_scratch).resolve()

    try:
        zero_block = format_dv_block(zero_values)
        base_lines = lines[:start] + zero_block + lines[end:]

        base_cfg = tmp_root_path / "fd_base.cfg"
        base_cfg.write_text("".join(base_lines))
        run_su2(args.su2_cfd, base_cfg, cfg_dir, args.mpi_procs)
        base_amplitude = read_amplitude_file(amp_path)
        print(f"Baseline amplitude: {base_amplitude:.10e}")

        results = []
        for idx, _ in enumerate(base_values):
            dv_values = zero_values.copy()
            dv_values[idx] += args.delta
            new_block = format_dv_block(dv_values)
            dv_lines = lines[:start] + new_block + lines[end:]

            if scratch_mesh.exists():
                scratch_mesh.unlink()

            cfg_def = tmp_root_path / f"fd_dv{idx}_def.cfg"
            cfg_def_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_base)
            cfg_def_lines = replace_option(cfg_def_lines, "MESH_OUT_FILENAME", args.mesh_scratch)
            cfg_def.write_text("".join(cfg_def_lines))
            run_su2(args.su2_def, cfg_def, cfg_dir, args.mpi_procs)

            cfg_run = tmp_root_path / f"fd_dv{idx}_run.cfg"
            cfg_run_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_scratch)
            cfg_run.write_text("".join(cfg_run_lines))
            run_su2(args.su2_cfd, cfg_run, cfg_dir, args.mpi_procs)
            amp_plus = read_amplitude_file(amp_path)

            gradient = (amp_plus - base_amplitude) / args.delta
            results.append((idx, amp_plus, gradient))

            log_handle.write(f"{idx:04d} {amp_plus:.10e} {gradient:.10e}\n")
            log_handle.flush()

            if args.keep_results:
                dest = cfg_dir / f"{amplitude_file}.{idx}_plus"
                shutil.copy2(amp_path, dest)
            if scratch_mesh.exists() and not args.keep_results:
                scratch_mesh.unlink()

        print(f"{'DV':>4} {'Amp(+Δ)':>15} {'FD Gradient':>15}")
        for idx, amp_plus, grad in results:
            print(f"{idx:>4d} {amp_plus:>15.8e} {grad:>15.8e}")

    finally:
        log_handle.close()
        if scratch_mesh.exists():
            scratch_mesh.unlink()
        if args.keep_results:
            cfg_backup = cfg_dir / "fd_runs"
            cfg_backup.mkdir(exist_ok=True)
            for file in tmp_root_path.iterdir():
                shutil.move(str(file), cfg_backup / file.name)
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
