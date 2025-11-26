import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import re
import pyvista as pv
import os



def plot_FGM_table(file_path, output_dir="plots", plot2D = "yes", plot3D="yes", show ="no"):
    """
    Master function to parse an FGM table and generate plots for all variables.
    Parameters:
    - file_path (str): Path to the .drg FGM file.
    - output_dir (str): Directory to save output plots.
    - plot2D (str): "yes" to generate 2D contour plots.
    - plot3D (str): "yes" to generate 3D surface plots.
    - show (str): "yes" to display plots interactively (useful for debugging).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data
    x_scaled, y_scaled, x_raw, y_raw, triangles, variables = parse_file(file_path)

    for var_name, z in variables.items():
        print(f"Plotting: {var_name}")
        safe_var_name = var_name.replace("/", "_").replace(" ", "_")
        
        # 2D plot
        if plot2D == "yes":
            plot_variable(
                x_scaled, y_scaled, x_raw, y_raw, triangles, z, 
                title=var_name,
                show = show
            )
            plt.savefig(os.path.join(output_dir, f"{safe_var_name}_2D.png"), dpi=300)
            plt.close()

        # 3D plot
        if plot3D == "yes":
            plot_3d_surface(
                x_raw, y_raw, triangles, z, 
                title=var_name,
                show = show
            )
            plt.savefig(os.path.join(output_dir, f"{safe_var_name}_3D.png"), dpi=300)
            plt.close()

def parse_file(file_path):
    """
    Parses an FGM table .drg file and returns coordinates, mesh, and variable data.

    Returns:
    - x_scaled, y_scaled: Normalized coordinates (0-1 range)
    - x_raw, y_raw: Original ProgressVariable and EnthalpyTot
    - triangles: Triangle connectivity (zero-based index)
    - variables: Dictionary of variable arrays
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract variable names
    variable_section = re.search(r'\[Variable names\]\n(.*?)\n</Header>', content, re.DOTALL).group(1)
    variable_names = []
    for line in variable_section.strip().splitlines():
        variable_names.extend(line.strip().split())

    # Extract data
    data_block = re.findall(r'<Data>\n(.*?)</Data>', content, re.DOTALL)[0].strip().split('\n')
    data = np.array([[float(v) for v in line.split()] for line in data_block])

    # Create variable dictionary
    variables = {name: data[:, idx] for idx, name in enumerate(variable_names)}

    # Raw coordinates
    x_raw = variables["ProgressVariable"]
    y_raw = variables["EnthalpyTot"]

    # Scale x and y to similar range for plotting mesh
    x_scaled = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y_scaled = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # Extract mesh connectivity
    connectivity_block = re.findall(r'<Connectivity>\n(.*?)</Connectivity>', content, re.DOTALL)[0].strip().split('\n')
    triangles = np.array([[int(idx) -1 for idx in line.split()] for line in connectivity_block])

    return x_scaled, y_scaled, x_raw, y_raw, triangles, variables

def plot_variable(x_scaled, y_scaled, x_raw, y_raw, triangles, variable, title="Variable Plot", show = "yes"):
    """
    Plots a 2D triangulated contour plot of a given variable.

    Parameters:
    - x_scaled, y_scaled: Scaled coordinates for aspect-ratio balanced plotting
    - x_raw, y_raw: Raw values for accurate axis labeling
    - triangles: Triangle mesh connectivity
    - variable: Variable values to plot
    - title (str): Title of the plot
    - show (str): "yes" to display the plot interactively
    """
    triang = tri.Triangulation(x_scaled, y_scaled, triangles)
    plt.figure(figsize=(8, 6))

    # Plot with scaled coords
    tpc = plt.tricontourf(triang, variable, levels=100, cmap='viridis')
    plt.triplot(triang, color='white', linewidth=0.3, alpha=0.5)
    plt.colorbar(tpc, label=title)

    # Add custom ticks that reflect raw values
    xticks_locs = np.linspace(x_scaled.min(), x_scaled.max(), 5)
    yticks_locs = np.linspace(y_scaled.min(), y_scaled.max(), 5)
    xticks_labels = np.linspace(x_raw.min(), x_raw.max(), 5)
    yticks_labels = np.linspace(y_raw.min(), y_raw.max(), 5)

    plt.xticks(xticks_locs, [f"{v:.2e}" for v in xticks_labels])
    plt.yticks(yticks_locs, [f"{v:.1e}" for v in yticks_labels])

    plt.xlabel("ProgressVariable")
    plt.ylabel("EnthalpyTot")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.tight_layout()
    if show == "yes":
        plt.show()

def plot_3d_surface(x_raw, y_raw, triangles, z_raw, title="3D Variable Surface", show = "yes"):
    """
    Plots a 3D surface of a variable using triangulated mesh.

    Parameters:
    - x_raw, y_raw: Original ProgressVariable and EnthalpyTot values
    - triangles: Triangle mesh connectivity
    - z_raw: Variable values to plot on Z-axis
    - title (str): Plot title and Z-axis label
    - show (str): "yes" to display interactively
    """
    # Normalize x and y just for display purposes
    x_scaled = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y_scaled = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    triang = tri.Triangulation(x_scaled, y_scaled, triangles)
    surf = ax.plot_trisurf(triang, z_raw, cmap='viridis', linewidth=0.5, antialiased=True, shade=True)

    ax.set_xlabel("ProgressVariable")
    ax.set_ylabel("EnthalpyTot")
    ax.set_zlabel(title)
    ax.set_title(f"3D Surface: {title}")
    ax.view_init(elev=23, azim=-130)
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Add custom ticks that reflect raw values
    xticks_locs = np.linspace(x_scaled.min(), x_scaled.max(), 5)
    yticks_locs = np.linspace(y_scaled.min(), y_scaled.max(), 5)
    xticks_labels = np.linspace(x_raw.min(), x_raw.max(), 5)
    yticks_labels = np.linspace(y_raw.min(), y_raw.max(), 5)
    
    plt.xticks(xticks_locs, [f"{v:.2e}" for v in xticks_labels])
    plt.yticks(yticks_locs, [f"{v:.1e}" for v in yticks_labels])

    # plt.tight_layout()
    if show == "yes":
        plt.show()


def export_to_vtu(file_path, output_path="fgm_table.vtu"):
    # Parse the file
    x_scaled, y_scaled, x_raw, y_raw, triangles, variables = parse_file(file_path)

    # Combine ProgressVariable and EnthalpyTot into points
    points = np.column_stack((x_scaled, y_scaled, np.zeros_like(x_raw)))  # z=0 for 2D

    # Convert triangles to pyvista-compatible format
    cells = np.hstack([np.array([3]*len(triangles)).reshape(-1,1), triangles])
    cells = cells.flatten()

    # Create the mesh
    mesh = pv.UnstructuredGrid(cells, np.full(len(triangles), pv.CellType.TRIANGLE), points)

    # Add scalar variables
    for name, data in variables.items():
        mesh.point_data[name] = data

    # Save to VTU file
    mesh.save(output_path)
    print(f"FGM table exported to: {output_path}")


def read_cfd_vtu(file_path):
    mesh = pv.read(file_path)

    # Extract x, y coordinates (assumes 2D in XY plane)
    points = mesh.points[:, :2]  # shape: (N, 2)

    # Extract fields of interest
    fields = {}
    for name in mesh.point_data:
        fields[name] = mesh.point_data[name]

    return points, fields


from scipy.interpolate import LinearNDInterpolator

def plot_simulation_on_FGM(x_scaled, y_scaled, x_raw, y_raw, triangles, variable, sim_file, title="Overlay"):
    # Load the simulation .vtu file
    sim = pv.read(sim_file)

    # Extract ProgressVariable and EnthalpyTot from simulation
    sim_PV = sim["ProgressVariable"]
    sim_H = sim["EnthalpyTot"]

    # Normalize using FGM table range
    sim_x_scaled = (sim_PV - x_raw.min()) / (x_raw.max() - x_raw.min())
    sim_y_scaled = (sim_H - y_raw.min()) / (y_raw.max() - y_raw.min())

    # Plot the FGM background
    triang = tri.Triangulation(x_scaled, y_scaled, triangles)
    fig, ax = plt.subplots(figsize=(8, 6))
    tpc = ax.tricontourf(triang, variable, levels=100, cmap='viridis')
    ax.triplot(triang, color='white', linewidth=0.3, alpha=0.3)
    fig.colorbar(tpc, label=title)

    # Overlay simulation points
    ax.scatter(sim_x_scaled, sim_y_scaled, color='red', s=5, label="Simulation Points", alpha=0.6)

    # Set aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Tick labels for raw physical values
    xticks_locs = np.linspace(x_scaled.min(), x_scaled.max(), 5)
    yticks_locs = np.linspace(y_scaled.min(), y_scaled.max(), 5)
    xticks_labels = np.linspace(x_raw.min(), x_raw.max(), 5)
    yticks_labels = np.linspace(y_raw.min(), y_raw.max(), 5)

    ax.set_xticks(xticks_locs)
    ax.set_yticks(yticks_locs)
    ax.set_xticklabels([f"{v:.2e}" for v in xticks_labels])
    ax.set_yticklabels([f"{v:.1e}" for v in yticks_labels])

    ax.set_xlabel("ProgressVariable")
    ax.set_ylabel("EnthalpyTot")
    ax.set_title("Simulation Overlay on FGM Table")
    ax.legend()

    # Interpolator for Z (variable)
    interpolator = LinearNDInterpolator(list(zip(x_scaled, y_scaled)), variable)

    # Pointer shows raw values + interpolated Z
    def format_coord(x, y):
        raw_x = x * (x_raw.max() - x_raw.min()) + x_raw.min()
        raw_y = y * (y_raw.max() - y_raw.min()) + y_raw.min()
        z_val = interpolator(x, y)
        if z_val is np.nan:
            z_str = "N/A"
        else:
            z_str = f"{z_val:.2e}"
        return f"ProgressVariable = {raw_x:.3e}, Enthalpy = {raw_y:.3e}, {title} = {z_str}"

    ax.format_coord = format_coord

    plt.tight_layout()
    plt.show()





# === Example Usage for interactive plot manipulation ===
# file_path = 'fgm_ch4.drg'
# x_scaled, y_scaled, x_raw, y_raw, triangles, variables = parse_file(file_path)
# plot_variable(x_scaled, y_scaled, x_raw, y_raw, triangles, variables["HeatRelease"], title="Temperature Distribution")
# plot_3d_surface(x_raw, y_raw, triangles, variables["HeatRelease"], title="Temperature")

# === Example usage for plotting the full table, all the variables 
# plot_FGM_table("fgm_ch4.drg", output_dir="fgm_variable_plots_old",plot2D="yes",plot3D="no")
# export_to_vtu("fgm_ch4.drg", output_path="fgm_ch4.vtu")

# Load CFD data
# cfd_points, cfd_fields = read_cfd_vtu("flow.vtu")

# print(cfd_fields)

# Load FGM table
x_scaled, y_scaled, x_raw, y_raw, triangles, variables = parse_file("LUT.drg")


# Plot FGM with CFD points
plot_simulation_on_FGM(
    x_scaled, y_scaled, x_raw, y_raw, triangles,
    variables["Heat_Release"],           # Variable from FGM table
    sim_file="flow.vtu",                # Path to CFD .vtu file
    title="Heat release rate"
)
