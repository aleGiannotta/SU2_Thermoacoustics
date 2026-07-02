#!/usr/bin/env python3
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


CFG = Path("contini_flame_unsteady.cfg")
PHASE_FILE = Path("dft_amplitude.dat")
DOT_FILE = Path("of_grad")
STEP = 1.0e-5
INNER_ITER = 300
TIME_ITER = 40


def read_cfg_lines() -> list[str]:
    return CFG.read_text().splitlines()


def write_cfg_lines(lines: list[str]) -> None:
    CFG.write_text("\n".join(lines) + "\n")


def set_cfg_options(updates: dict[str, str]) -> None:
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


def set_dv0(value: float) -> None:
    lines = read_cfg_lines()
    for index, line in enumerate(lines):
        if line.strip().startswith("DV_VALUE"):
            _, rhs = line.split("=", 1)
            values = [entry.strip() for entry in rhs.split(",")]
            values[0] = f"{value:.6f}"
            lines[index] = "DV_VALUE= " + ", ".join(values)
            write_cfg_lines(lines)
            return
    raise RuntimeError("DV_VALUE not found in config.")


def read_phase(path: Path) -> float:
    for line in path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) == 2 and tokens[0] == "DFT_PHASE":
            return float(tokens[1])
    raise RuntimeError(f"DFT_PHASE not found in {path}.")


def read_dot0(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or "gradient" in stripped.lower():
            continue
        values.append(float(stripped))
    if not values:
        raise RuntimeError(f"No gradient values found in {path}.")
    return values[0]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_direct() -> None:
    run(["SU2_CFD", CFG.name])


def run_deformation() -> None:
    run(["SU2_DEF", CFG.name])


def main() -> None:
    original_cfg = CFG.read_text()
    dot0 = read_dot0(DOT_FILE)
    try:
        set_cfg_options(
            {
                "RESTART_SOL": "NO",
                "INNER_ITER": str(INNER_ITER),
                "TIME_ITER": str(TIME_ITER),
                "OBJECTIVE_TEMPORAL_MODE": "DFT_PHASE",
                "MESH_FILENAME": "mesh_coarse_ffd.su2",
            }
        )

        set_dv0(0.0)
        run_direct()
        phi0 = read_phase(PHASE_FILE)
        shutil.copy2(PHASE_FILE, "dft_amplitude_baseline.dat")

        set_dv0(STEP)
        set_cfg_options({"MESH_FILENAME": "mesh_coarse_ffd.su2"})
        run_deformation()
        set_cfg_options({"MESH_FILENAME": "mesh_out.su2"})
        run_direct()
        phi1 = read_phase(PHASE_FILE)
        shutil.copy2(PHASE_FILE, "dft_amplitude_dv0_plus.dat")

        fd = (phi1 - phi0) / STEP
        abs_diff = abs(fd - dot0)
        rel_diff = abs_diff / max(abs(fd), abs(dot0), 1.0e-16)

        print()
        print(f"baseline_phase={phi0:.16e}")
        print(f"perturbed_phase={phi1:.16e}")
        print(f"fd_gradient_dv0={fd:.16e}")
        print(f"dot_gradient_dv0={dot0:.16e}")
        print(f"abs_diff={abs_diff:.16e}")
        print(f"rel_diff={rel_diff:.16e}")
        Path("dv0_check_summary.txt").write_text(
            "\n".join(
                [
                    f"baseline_phase={phi0:.16e}",
                    f"perturbed_phase={phi1:.16e}",
                    f"fd_gradient_dv0={fd:.16e}",
                    f"dot_gradient_dv0={dot0:.16e}",
                    f"abs_diff={abs_diff:.16e}",
                    f"rel_diff={rel_diff:.16e}",
                ]
            )
            + "\n"
        )
    finally:
        CFG.write_text(original_cfg)


if __name__ == "__main__":
    main()
