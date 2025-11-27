#!/usr/bin/env python3
import subprocess
from pathlib import Path

def get_cfg_option(cfg_path: Path, key: str) -> str | None:
    key = key.upper()
    for raw in cfg_path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if "=" not in stripped:
            continue
        lhs, rhs = stripped.split("=", 1)
        if lhs.strip().upper() == key:
            return rhs.strip()
    return None

def update_cfg(cfg_path: Path, updates: dict) -> None:
    """Replace cfg keys with new values. `updates` is {key: value_str}."""
    lines = cfg_path.read_text().splitlines()
    key_map = {k.upper(): str(v) for k, v in updates.items()}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):  # skip comments/blank
            continue
        if "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip().upper()
        if key in key_map:
            lines[idx] = f"{key}= {key_map[key]}"
    cfg_path.write_text("\n".join(lines) + "\n")

def read_dv_values(cfg_path: Path) -> list[float]:
    lines = cfg_path.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("DV_VALUE"):
            _, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
            return [float(entry) for entry in entries]
    raise RuntimeError("DV_VALUE block not found in config.")

def write_dv_values(cfg_path: Path, values: list[float]) -> None:
    lines = cfg_path.read_text().splitlines()
    formatted = ", ".join(f"{val:.6f}" for val in values)
    for idx, line in enumerate(lines):
        if line.strip().startswith("DV_VALUE"):
            prefix = line.split("=", 1)[0].strip()
            lines[idx] = f"{prefix}= {formatted}"
            cfg_path.write_text("\n".join(lines) + "\n")
            return
    raise RuntimeError("DV_VALUE block not found in config.")

def update_dv_value(cfg_path: Path, index: int, value: float) -> None:
    lines = cfg_path.read_text().splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DV_VALUE"):
            prefix, rhs = line.split("=", 1)
            entries = [entry.strip() for entry in rhs.replace("\\", " ").split(",")]
            if index < 0 or index >= len(entries):
                raise IndexError(f"DV_VALUE index {index} out of range (size {len(entries)})")
            entries[index] = f"{value:.6f}"
            new_rhs = ", ".join(entries)
            lines[idx] = f"{prefix.strip()}= {new_rhs}"
            cfg_path.write_text("\n".join(lines) + "\n")
            return
    raise RuntimeError("DV_VALUE block not found in config.")

def run_su2(config_file: str, mpi_ranks: int = 1) -> None:
    cmd = ["mpirun", "-n", str(mpi_ranks), "SU2_CFD", config_file] if mpi_ranks > 1 else ["SU2_CFD", config_file]
    subprocess.run(cmd, check=True)

def run_su2_ad(config_file: str, mpi_ranks: int = 1) -> None:
    cmd = ["mpirun", "-n", str(mpi_ranks), "SU2_CFD_AD", config_file] if mpi_ranks > 1 else ["SU2_CFD_AD", config_file]
    subprocess.run(cmd, check=True)

def run_def(config_file) -> None:
    cmd = ["SU2_DEF", config_file]
    subprocess.run(cmd, check=True)

def run_dot_ad(config_file) -> None:
    cmd = ["SU2_DOT_AD", config_file]
    subprocess.run(cmd, check=True)

def read_last_cd(history_path: Path) -> float:
    if not history_path.exists():
        raise RuntimeError(f"{history_path} not found.")
    lines = history_path.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError("History file has no data rows.")
    header = [col.strip().strip('"') for col in lines[0].split(",")]
    if "tavg[CD]" not in header:
        raise RuntimeError("Column tavg[CD] not found in history file.")
    idx = header.index("tavg[CD]")
    last_line = lines[-1].split(",")
    return float(last_line[idx])

def read_dft_amplitude(amplitude_path: Path) -> float:
    if not amplitude_path.exists():
        raise RuntimeError(f"{amplitude_path} not found.")
    value = None
    for line in amplitude_path.read_text().splitlines():
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
        raise RuntimeError(f"No numeric amplitude found in {amplitude_path}.")
    return value


def read_dft_phase(amplitude_path: Path) -> float:
    if not amplitude_path.exists():
        raise RuntimeError(f"{amplitude_path} not found.")
    value = None
    for line in amplitude_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.replace(",", " ").split()
        if len(tokens) >= 2 and tokens[0].upper() == "DFT_PHASE":
            try:
                value = float(tokens[1])
            except ValueError:
                continue
    if value is None:
        raise RuntimeError(f"No numeric phase found in {amplitude_path}.")
    return value

if __name__ == "__main__":
    cfg = Path("lam_flatplate_direct.cfg")
    cfg_dir = cfg.parent
    history_path = cfg_dir / "history.csv"
    objective_mode = (get_cfg_option(cfg, "OBJECTIVE_TEMPORAL_MODE") or "AVERAGE").upper()
    dft_output_name = get_cfg_option(cfg, "OBJECTIVE_DFT_OUTPUT")
    amp_file = cfg_dir / (dft_output_name if dft_output_name else "dft_amplitude.dat")
    if objective_mode == "DFT_AMPLITUDE":
        objective_label = "DFT amplitude"
        def read_objective() -> float:
            return read_dft_amplitude(amp_file)
    elif objective_mode == "DFT_PHASE":
        objective_label = "DFT phase"
        def read_objective() -> float:
            return read_dft_phase(amp_file)
    else:
        objective_label = "tavg[CD]"
        def read_objective() -> float:
            return read_last_cd(history_path)
    
    base_dv_values = read_dv_values(cfg)
    dv_count = len(base_dv_values)
    # Baseline simulation
    write_dv_values(cfg, [0.0]*dv_count)
    update_cfg(cfg, {"MESH_FILENAME": "mesh.su2"})
    update_cfg(cfg, {"CFL_NUMBER": "50"})
    run_su2(str(cfg), mpi_ranks=4)
    obj0 = read_objective()
    print(f"{objective_label} baseline = {obj0}")
    amp0 = None
    if objective_mode == "DFT_AMPLITUDE":
        amp0 = obj0
    elif amp_file.exists():
        try:
            amp0 = read_dft_amplitude(amp_file)
            print(f"DFT amplitude baseline = {amp0}")
        except RuntimeError as err:
            print(f"[WARN] {err}")
    # Adjoint simulation
    update_cfg(cfg, {"CFL_NUMBER": "10"})
    run_su2_ad(str(cfg), mpi_ranks=4)
    run_dot_ad(str(cfg))
    
    # Perturbation Tests for all DVs
    eps = 1e-5
    gradients = []
    grad_file = cfg_dir / "of_grad_FD.dat"
    grad_handle = grad_file.open("w", encoding="utf-8")
    grad_handle.write(f"# FD gradients for {objective_label}\n")
    grad_handle.write(f"# Perturbation size: {eps:.3e}\n")
    for index in range(dv_count):
        current_dv = [0.0]*dv_count
        current_dv[index] = eps
        write_dv_values(cfg, current_dv)
        update_cfg(cfg, {"MESH_FILENAME": "mesh.su2"})
        run_def(str(cfg))

        # Run CFD on deformed mesh
        update_cfg(cfg, {"MESH_FILENAME": "mesh_out.su2", "CFL_NUMBER":50})
        run_su2(str(cfg), mpi_ranks=4)
        obj1 = read_objective()
        grad = (obj1-obj0)/eps
        gradients.append(grad)
        grad_handle.write(f"{index:04d} {grad:.12e}\n")
        print(f"{objective_label} (perturbed, index {index}) = {obj1}")
        print(f"{objective_label} GRADIENT COMPONENT {grad}")
    grad_handle.close()
    print(f"\nFinite-difference gradients written to {grad_file}:")
    for idx, value in enumerate(gradients):
        print(f"DV {idx:02d}: {value:.12e}")
