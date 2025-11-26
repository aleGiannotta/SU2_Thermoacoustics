import os
import subprocess
import shutil
import re
import csv

# ----------------------------------------------------------------------
# Low-level SU2 runners
# ----------------------------------------------------------------------

def run_cmd(cmd, workdir):
    print(f"Running: {' '.join(cmd)} in {os.path.abspath(workdir)}")
    subprocess.run(cmd, cwd=workdir, check=True)


def run_su2_cfd(cfg_file, workdir):
    run_cmd(["mpirun", "-n", "6", "SU2_CFD", cfg_file], workdir)


def run_su2_def(cfg_file, workdir):
    run_cmd(["SU2_DEF", cfg_file], workdir)


# ----------------------------------------------------------------------
# History file: extract objective
# ----------------------------------------------------------------------

def extract_functional(history_file, keyword="HeatReleaseGlobal"):
    """
    Parse SU2 history.csv file (CSV with header row) and return the LAST value
    for the column named `keyword`.
    """
    if not os.path.isfile(history_file):
        raise FileNotFoundError(f"History file not found: {history_file}")

    with open(history_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # First non-empty row is header
    header = None
    for r in rows:
        if len(r) > 0:
            header = r
            break

    if header is None:
        raise RuntimeError("No header found in history.csv")

    # Clean header fields (strip quotes)
    header = [h.strip().strip('"') for h in header]

    if keyword not in header:
        raise RuntimeError(f"Functional '{keyword}' not found. Available: {header}")

    idx = header.index(keyword)

    # Last non-empty row with numerical data
    for r in reversed(rows):
        if len(r) == len(header):
            try:
                val = float(r[idx])
                return val
            except ValueError:
                continue

    raise RuntimeError("No valid numeric data found in history.csv")


# ----------------------------------------------------------------------
# CFG parsing utilities
# ----------------------------------------------------------------------

def _first_noncomment_line_with(cfg_lines, key):
    """
    Return index of first line containing 'key' that is not commented out.
    """
    for i, line in enumerate(cfg_lines):
        stripped = line.strip()
        if stripped.startswith("%"):
            continue
        if key in stripped:
            return i
    return None


def get_mesh_filename_from_cfd_cfg(cfg_path):
    """
    Read MESH_FILENAME from SU2_CFD config.
    """
    with open(cfg_path, "r") as f:
        lines = f.readlines()

    idx = _first_noncomment_line_with(lines, "MESH_FILENAME")
    if idx is None:
        raise RuntimeError(f"MESH_FILENAME not found in {cfg_path}")

    line = lines[idx].strip()
    # Expect: MESH_FILENAME= mesh.su2
    if "=" not in line:
        raise RuntimeError(f"Malformed MESH_FILENAME line: {line}")

    rhs = line.split("=", 1)[1]
    # strip comments and spaces
    rhs = rhs.split("%")[0].strip()
    return rhs


def get_dv_values_and_block(cfg_lines):
    """
    From contini_flame_DEF.cfg content, extract:
      - DV values as a list[float]
      - start index of DV_VALUE block
      - end index (exclusive) of DV_VALUE block
    Handles multi-line DV_VALUE with continuations.
    """
    idx = _first_noncomment_line_with(cfg_lines, "DV_VALUE")
    if idx is None:
        raise RuntimeError("DV_VALUE not found in DEF cfg.")

    # Collect block lines: DV_VALUE line + continuation lines without '='
    block_lines = [cfg_lines[idx]]
    k = idx + 1
    while k < len(cfg_lines):
        stripped = cfg_lines[k].strip()
        if not stripped:
            break
        if stripped.startswith("%"):
            break
        if "=" in stripped:
            # new parameter starts
            break
        block_lines.append(cfg_lines[k])
        k += 1

    # Parse floats from the concatenated block
    text = " ".join(block_lines)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        raise RuntimeError("No numeric DV values found in DV_VALUE block.")

    dv_vals = [float(x) for x in nums]
    return dv_vals, idx, k


def build_def_cfg_with_dvs(base_lines, new_dv_values, deformed_mesh_name=None):
    """
    Return new list of lines for DEF cfg:
      - DV_VALUE replaced with new_dv_values on a single line
      - (optionally) deformed mesh filename updated if key found
    """
    lines = list(base_lines)  # shallow copy

    # Replace DV_VALUE block
    _, start, end = get_dv_values_and_block(lines)
    new_dv_line = "DV_VALUE= " + ", ".join(f"{v:.8g}" for v in new_dv_values) + "\n"
    lines[start:end] = [new_dv_line]

    # Optionally update deformed mesh filename
    if deformed_mesh_name is not None:
        updated = False
        keys = ["MESH_OUT_FILENAME", "DEFORMED_MESH_FILENAME"]
        for key in keys:
            idx = _first_noncomment_line_with(lines, key)
            if idx is not None:
                # replace rhs
                raw = lines[idx]
                # keep anything before '=' and any trailing comment
                prefix, rest = raw.split("=", 1)
                comment = ""
                if "%" in rest:
                    rest, comment = rest.split("%", 1)
                    comment = "%" + comment
                new_line = f"{prefix.strip()}= {deformed_mesh_name} {comment}".rstrip() + "\n"
                lines[idx] = new_line
                updated = True
                break

        # If no key exists, append one at end
        if not updated:
            lines.append(f"DEFORMED_MESH_FILENAME= {deformed_mesh_name}\n")

    return lines


def build_cfd_cfg_with_mesh(base_lines, mesh_name):
    """
    Return new CFD cfg lines where MESH_FILENAME is set to mesh_name.
    """
    lines = list(base_lines)
    idx = _first_noncomment_line_with(lines, "MESH_FILENAME")
    if idx is None:
        raise RuntimeError("MESH_FILENAME not found in CFD cfg.")

    raw = lines[idx]
    prefix, rest = raw.split("=", 1)
    comment = ""
    if "%" in rest:
        rest, comment = rest.split("%", 1)
        comment = "%" + comment
    new_line = f"{prefix.strip()}= {mesh_name} {comment}".rstrip() + "\n"
    lines[idx] = new_line
    return lines


# ----------------------------------------------------------------------
# Finite-difference gradient w.r.t. DVs
# ----------------------------------------------------------------------

def finite_difference_dv_gradient(
    cfd_cfg="contini_flame.cfg",
    def_cfg="contini_flame_DEF.cfg",
    base_dir=".",
    eps=1e-5,
    functional_keyword="HEAT_RELEASE_GLOBAL",
):
    """
    Compute FD gradient dJ/dDV_i by perturbing DVs in contini_flame_DEF.cfg
    and running SU2_DEF + SU2_CFD for each DV.
    """

    cfd_cfg_path = os.path.join(base_dir, cfd_cfg)
    def_cfg_path = os.path.join(base_dir, def_cfg)

    if not os.path.isfile(cfd_cfg_path):
        raise FileNotFoundError(f"CFD cfg not found: {cfd_cfg_path}")
    if not os.path.isfile(def_cfg_path):
        raise FileNotFoundError(f"DEF cfg not found: {def_cfg_path}")

    # Read base cfg contents
    with open(def_cfg_path, "r") as f:
        def_base_lines = f.readlines()
    with open(cfd_cfg_path, "r") as f:
        cfd_base_lines = f.readlines()

    # Get base DV values
    dv_values, _, _ = get_dv_values_and_block(def_base_lines)
    n_dv = len(dv_values)
    print(f"Found {n_dv} design variables in DV_VALUE.")

    # Get the mesh name used by the primal
    base_mesh_name = get_mesh_filename_from_cfd_cfg(cfd_cfg_path)
    print(f"Base CFD mesh: {base_mesh_name}")

    # ------------------------------------------------------------------
    # Baseline run: J0 with base mesh and base DVs
    # ------------------------------------------------------------------
    print("Running baseline SU2_CFD...")
    run_su2_cfd(cfd_cfg, base_dir)
    J0 = extract_functional(os.path.join(base_dir, "history.csv"),
                            keyword=functional_keyword)
    print(f"Baseline functional J0 = {J0}")

    # ------------------------------------------------------------------
    # Loop over DVs
    # ------------------------------------------------------------------
    gradients = []

    for i in range(n_dv):
        print(f"\n=== DV {i} / {n_dv-1} ===")

        # Perturb DV i
        dv_pert = dv_values.copy()
        dv_pert[i] += eps

        # Unique names for this DV
        deformed_mesh_name = f"mesh_def_dv_{i}.su2"
        def_cfg_fd = f"contini_flame_DEF_fd_{i}.cfg"
        cfd_cfg_fd = f"contini_flame_fd_{i}.cfg"

        # Build and write perturbed DEF cfg
        def_lines_fd = build_def_cfg_with_dvs(
            def_base_lines,
            dv_pert,
            deformed_mesh_name=deformed_mesh_name,
        )
        with open(os.path.join(base_dir, def_cfg_fd), "w") as f:
            f.writelines(def_lines_fd)

        # Run SU2_DEF to generate deformed mesh for this DV
        print(f"Generating deformed mesh for DV {i} with SU2_DEF...")
        run_su2_def(def_cfg_fd, base_dir)

        # Build and write CFD cfg that uses the deformed mesh
        cfd_lines_fd = build_cfd_cfg_with_mesh(cfd_base_lines, deformed_mesh_name)
        with open(os.path.join(base_dir, cfd_cfg_fd), "w") as f:
            f.writelines(cfd_lines_fd)

        # Run SU2_CFD with deformed mesh
        print(f"Running SU2_CFD on deformed mesh {deformed_mesh_name}...")
        run_su2_cfd(cfd_cfg_fd, base_dir)

        # Extract functional
        Ji = extract_functional(os.path.join(base_dir, "history.csv"),
                                keyword=functional_keyword)
        dJ = (Ji - J0) / eps
        gradients.append(dJ)

        print(f"DV {i}: J = {Ji}, FD gradient dJ/dDV_{i} = {dJ}")

    return dv_values, gradients


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = "."
    CFD_CFG = "contini_flame.cfg"
    DEF_CFG = "contini_flame_DEF.cfg"
    EPS = 1e-4  # adjust if needed

    dv_vals, grad_vals = finite_difference_dv_gradient(
        cfd_cfg=CFD_CFG,
        def_cfg=DEF_CFG,
        base_dir=BASE_DIR,
        eps=EPS,
        functional_keyword="HeatReleaseGlobal",
    )

    print("\n==== Summary ====")
    for i, (dv, g) in enumerate(zip(dv_vals, grad_vals)):
        print(f"DV {i}: value = {dv}, dJ/dDV = {g}")
