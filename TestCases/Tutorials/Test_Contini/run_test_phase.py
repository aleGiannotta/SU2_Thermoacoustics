#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


CFG_NAME = "contini_flame_unsteady.cfg"
DEFAULT_MPI_RANKS = 4
DEFAULT_FD_STEP = 1e-5
DEFAULT_DIRECT_CFL = 50
DEFAULT_ADJOINT_CFL = 10
BASE_MESH = "mesh.su2"
FFD_SOURCE_MESH = "mesh_ffd.su2"
DEFORMED_MESH = "mesh_out.su2"
FD_GRAD_FILE = "of_grad_FD_phase.dat"
AD_GRAD_FILE = "of_grad_phase.csv"
COMPARISON_FILE = "gradient_comparison_phase.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DFT-phase gradients for Test_Contini.")
    parser.add_argument("--mpi-ranks", type=int, default=DEFAULT_MPI_RANKS)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--direct-cfl", type=int, default=DEFAULT_DIRECT_CFL)
    parser.add_argument("--adjoint-cfl", type=int, default=DEFAULT_ADJOINT_CFL)
    parser.add_argument("--inner-iter", type=int, default=None)
    parser.add_argument("--time-iter", type=int, default=None)
    parser.add_argument("--adjoint-time-iter", type=int, default=None)
    parser.add_argument("--dv-indices", type=str, default=None, help="Comma-separated DV indices to FD-check.")
    return parser.parse_args()


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
    lines = cfg_path.read_text().splitlines()
    key_map = {k.upper(): str(v) for k, v in updates.items()}
    updated_keys: set[str] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip().upper()
        if key in key_map:
            lines[idx] = f"{key}= {key_map[key]}"
            updated_keys.add(key)
    for key, value in key_map.items():
        if key not in updated_keys:
            lines.append(f"{key}= {value}")
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


def run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_su2(config_file: str, mpi_ranks: int = 1) -> None:
    cmd = ["mpirun", "-n", str(mpi_ranks), "SU2_CFD", config_file] if mpi_ranks > 1 else ["SU2_CFD", config_file]
    run_command(cmd)


def run_su2_ad(config_file: str, mpi_ranks: int = 1) -> None:
    cmd = ["mpirun", "-n", str(mpi_ranks), "SU2_CFD_AD", config_file] if mpi_ranks > 1 else ["SU2_CFD_AD", config_file]
    run_command(cmd)


def run_def(config_file: str) -> None:
    run_command(["SU2_DEF", config_file])


def run_dot_ad(config_file: str) -> None:
    run_command(["SU2_DOT_AD", config_file])


def read_dft_phase(path: Path) -> float:
    if not path.exists():
        raise RuntimeError(f"{path} not found.")
    value = None
    for line in path.read_text().splitlines():
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
        raise RuntimeError(f"No numeric phase found in {path}.")
    return value


def read_adjoint_gradient(path: Path, expected_size: int) -> list[float]:
    if not path.exists():
        raise RuntimeError(f"Adjoint gradient file {path} not found.")

    gradients: list[float] = []
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                cleaned = [entry.strip().strip('"') for entry in row if entry.strip()]
                if not cleaned:
                    continue
                numeric_values: list[float] = []
                for entry in cleaned:
                    try:
                        numeric_values.append(float(entry))
                    except ValueError:
                        continue
                if not numeric_values:
                    continue
                gradients.append(numeric_values[-1])
    else:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            numeric_values: list[float] = []
            for token in stripped.replace(",", " ").split():
                try:
                    numeric_values.append(float(token))
                except ValueError:
                    continue
            if numeric_values:
                gradients.append(numeric_values[-1])

    if len(gradients) < expected_size:
        raise RuntimeError(
            f"Adjoint gradient file {path} has {len(gradients)} values, expected at least {expected_size}."
        )
    return gradients[:expected_size]


def max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def build_design_mesh(cfg_path: Path, dv_values: list[float]) -> str:
    write_dv_values(cfg_path, dv_values)
    if max_abs(dv_values) <= 0.0:
        update_cfg(cfg_path, {"MESH_FILENAME": BASE_MESH})
        return BASE_MESH
    update_cfg(cfg_path, {"MESH_FILENAME": FFD_SOURCE_MESH})
    run_def(str(cfg_path))
    update_cfg(cfg_path, {"MESH_FILENAME": DEFORMED_MESH})
    return DEFORMED_MESH


def run_direct_objective(
    cfg_path: Path,
    dv_values: list[float],
    phase_file: Path,
    cfl_number: int,
    mpi_ranks: int,
) -> float:
    mesh_name = build_design_mesh(cfg_path, dv_values)
    update_cfg(
        cfg_path,
        {
            "MESH_FILENAME": mesh_name,
            "CFL_NUMBER": cfl_number,
            "OBJECTIVE_TEMPORAL_MODE": "DFT_PHASE",
        },
    )
    run_su2(str(cfg_path), mpi_ranks=mpi_ranks)
    return read_dft_phase(phase_file)


def write_comparison_table(
    path: Path,
    fd_gradients: list[float],
    ad_gradients: list[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dv_index", "fd_gradient", "adjoint_gradient", "abs_diff", "rel_diff"])
        for index, (fd_value, ad_value) in enumerate(zip(fd_gradients, ad_gradients)):
            abs_diff = abs(fd_value - ad_value)
            scale = max(abs(fd_value), abs(ad_value), 1e-16)
            rel_diff = abs_diff / scale
            writer.writerow([index, f"{fd_value:.16e}", f"{ad_value:.16e}", f"{abs_diff:.16e}", f"{rel_diff:.16e}"])


if __name__ == "__main__":
    args = parse_args()
    cfg = Path(CFG_NAME)
    cfg_dir = cfg.parent
    original_cfg_text = cfg.read_text()

    try:
        dft_output_name = get_cfg_option(cfg, "OBJECTIVE_DFT_OUTPUT")
        phase_file = cfg_dir / (dft_output_name if dft_output_name else "dft_amplitude.dat")

        base_dv_values = read_dv_values(cfg)
        dv_count = len(base_dv_values)
        if args.dv_indices:
            dv_indices = [int(item.strip()) for item in args.dv_indices.split(",") if item.strip()]
        else:
            dv_indices = list(range(dv_count))

        grad_output_name = get_cfg_option(cfg, "GRAD_OBJFUNC_FILENAME")
        adjoint_grad_path = cfg_dir / (
            grad_output_name if grad_output_name else AD_GRAD_FILE
        )

        cfg_updates = {
            "OBJECTIVE_TEMPORAL_MODE": "DFT_PHASE",
            "GRAD_OBJFUNC_FILENAME": adjoint_grad_path.name,
            "MESH_OUT_FILENAME": DEFORMED_MESH,
        }
        if args.inner_iter is not None:
            cfg_updates["INNER_ITER"] = args.inner_iter
        if args.time_iter is not None:
            cfg_updates["TIME_ITER"] = args.time_iter
        if args.adjoint_time_iter is not None:
            cfg_updates["UNST_ADJOINT_ITER"] = args.adjoint_time_iter
        elif args.time_iter is not None:
            cfg_updates["UNST_ADJOINT_ITER"] = args.time_iter
        update_cfg(cfg, cfg_updates)

        phi0 = run_direct_objective(cfg, base_dv_values, phase_file, args.direct_cfl, args.mpi_ranks)
        print(f"DFT phase baseline = {phi0:.16e}")

        baseline_mesh = DEFORMED_MESH if max_abs(base_dv_values) > 0.0 else BASE_MESH
        update_cfg(
            cfg,
            {
                "MESH_FILENAME": baseline_mesh,
                "CFL_NUMBER": args.adjoint_cfl,
                "OBJECTIVE_TEMPORAL_MODE": "DFT_PHASE",
            },
        )
        run_su2_ad(str(cfg), mpi_ranks=args.mpi_ranks)
        run_dot_ad(str(cfg))
        ad_gradients = read_adjoint_gradient(adjoint_grad_path, dv_count)

        fd_gradients: list[float] = []
        fd_grad_path = cfg_dir / FD_GRAD_FILE
        with fd_grad_path.open("w", encoding="utf-8") as grad_handle:
            grad_handle.write("# FD gradients for DFT phase\n")
            grad_handle.write(f"# Perturbation size: {args.fd_step:.3e}\n")
            for index in dv_indices:
                current_dv = base_dv_values[:]
                current_dv[index] += args.fd_step
                phi1 = run_direct_objective(cfg, current_dv, phase_file, args.direct_cfl, args.mpi_ranks)
                grad = (phi1 - phi0) / args.fd_step
                fd_gradients.append(grad)
                grad_handle.write(f"{index:04d} {grad:.12e}\n")
                print(f"DV {index:02d} perturbed phase = {phi1:.16e}")
                print(f"DV {index:02d} FD gradient = {grad:.16e}")

        comparison_path = cfg_dir / COMPARISON_FILE
        selected_ad_gradients = [ad_gradients[index] for index in dv_indices]
        write_comparison_table(comparison_path, fd_gradients, selected_ad_gradients)

        print("\nGradient comparison:")
        for index, fd_value, ad_value in zip(dv_indices, fd_gradients, selected_ad_gradients):
            abs_diff = abs(fd_value - ad_value)
            rel_diff = abs_diff / max(abs(fd_value), abs(ad_value), 1e-16)
            print(
                f"DV {index:02d}: FD={fd_value:.16e} "
                f"AD={ad_value:.16e} ABS={abs_diff:.3e} REL={rel_diff:.3e}"
            )

        print(f"\nFD gradients written to {fd_grad_path}")
        print(f"Adjoint gradients read from {adjoint_grad_path}")
        print(f"Comparison table written to {comparison_path}")
    finally:
        cfg.write_text(original_cfg_text)
