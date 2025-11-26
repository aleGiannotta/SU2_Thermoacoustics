#!/usr/bin/env python3
"""
Compute finite-difference gradients of the windowed-average cost
function by perturbing one DV at a time (using SU2_DEF+SU2_CFD).
"""

from __future__ import annotations

import argparse
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
        raise RuntimeError("DV_VALUE block not found.")

    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped or stripped.startswith("%"):
            break
        end += 1

    block = "".join(lines[start:end])
    if "=" not in block:
        raise RuntimeError("Malformed DV_VALUE block.")
    block = block.split("=", 1)[1]
    block = block.replace("\\", " ")
    values = [float(val) for val in block.replace(",", " ").split() if val]
    return start, end, values


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


def replace_option(lines: List[str], option: str, value: str) -> List[str]:
    target = option.strip()
    out = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(target + "=") or stripped.startswith(target + " ="):
            out.append(f"{target}= {value}\n")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{target}= {value}\n")
    return out


def read_history_value(hist_path: Path, column: str) -> float:
    if not hist_path.exists():
        raise RuntimeError(f"History file {hist_path} not found.")
    with hist_path.open("r") as handle:
        lines = handle.readlines()
    if len(lines) < 2:
        raise RuntimeError(f"No data in history file {hist_path}.")
    header = [name.strip().strip('"') for name in lines[0].split(",")]
    if column not in header:
        raise RuntimeError(f"Column {column} not found in history file.")
    col_idx = header.index(column)
    last_line = lines[-1].split(",")
    if len(last_line) <= col_idx:
        raise RuntimeError(f"Column index {col_idx} out of range in history row.")
    return float(last_line[col_idx])


def run_su2(executable: str, cfg: Path, workdir: Path, mpi_procs: int) -> None:
    log_path = workdir / "main_log.log"
    cmd = ["mpirun", "-np", str(mpi_procs), executable, str(cfg)] if mpi_procs > 1 else [executable, str(cfg)]
    with open(log_path, "a", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=workdir, check=True, stdout=log, stderr=log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference gradients for windowed average objective.")
    parser.add_argument("config", type=Path, help="Baseline SU2 config file.")
    parser.add_argument("--su2-cfd", default="SU2_CFD", help="SU2_CFD executable.")
    parser.add_argument("--su2-def", default="SU2_DEF", help="SU2_DEF executable.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Perturbation size.")
    parser.add_argument("--mpi-procs", type=int, default=4, help="MPI ranks.")
    parser.add_argument("--mesh-base", default="mesh.su2", help="Base mesh file.")
    parser.add_argument("--mesh-def", default="mesh_def.su2", help="Deformed mesh file.")
    parser.add_argument("--log", type=Path, help="FD result log file.")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg_dir = cfg_path.parent
    lines = cfg_path.read_text().splitlines(True)

    start, end, base_values = parse_dv_block(lines)
    history_file = cfg_dir / "history.csv"
    log_path = (cfg_dir / "fd_avg.log") if args.log is None else (args.log if args.log.is_absolute() else cfg_dir / args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write("# DV  ObjValue  FD_Gradient\n")
    log_handle.flush()

    tmp_root = Path(tempfile.mkdtemp(prefix="su2_fd_avg_", dir=cfg_dir))
    def_mesh = (cfg_dir / args.mesh_def).resolve()

    try:
        base_block = format_dv_block(base_values)
        base_cfg_lines = lines[:start] + base_block + lines[end:]
        base_cfg = tmp_root / "fd_base.cfg"
        base_cfg.write_text("".join(base_cfg_lines))
        run_su2(args.su2_cfd, base_cfg, cfg_dir, args.mpi_procs)
        base_obj = read_history_value(history_file, "tavg[CD]")
        print(f"Baseline objective: {base_obj:.10e}")

        results = []
        for idx in range(len(base_values)):
            dv_vals = base_values.copy()
            dv_vals[idx] += args.delta
            dv_block = format_dv_block(dv_vals)
            dv_lines = lines[:start] + dv_block + lines[end:]

            if def_mesh.exists():
                def_mesh.unlink()

            cfg_def = tmp_root / f"fd_dv{idx}_def.cfg"
            cfg_def_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_base)
            cfg_def_lines = replace_option(cfg_def_lines, "MESH_OUT_FILENAME", args.mesh_def)
            cfg_def.write_text("".join(cfg_def_lines))
            run_su2(args.su2_def, cfg_def, cfg_dir, args.mpi_procs)

            cfg_run = tmp_root / f"fd_dv{idx}_run.cfg"
            cfg_run_lines = replace_option(dv_lines, "MESH_FILENAME", args.mesh_def)
            cfg_run.write_text("".join(cfg_run_lines))
            run_su2(args.su2_cfd, cfg_run, cfg_dir, args.mpi_procs)

            obj_val = read_history_value(history_file, "tavg[CD]")
            gradient = (obj_val - base_obj) / args.delta
            results.append((idx, obj_val, gradient))
            log_handle.write(f"{idx:04d} {obj_val:.10e} {gradient:.10e}\n")
            log_handle.flush()

        print(f"{'DV':>4} {'Obj(+Δ)':>15} {'FD Gradient':>15}")
        for idx, obj_val, grad in results:
            print(f"{idx:>4d} {obj_val:>15.8e} {grad:>15.8e}")

    finally:
        log_handle.close()
        if def_mesh.exists():
            def_mesh.unlink()
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
