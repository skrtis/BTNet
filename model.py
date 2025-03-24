import math
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from shapely.affinity import translate, scale as scale_geom

#############################################
# STEP 1: Load GeoJSON and Rescale Geometries
#############################################
geojson_path = "malpeque_tiles.geojson"  # local file
agents_gdf = gpd.read_file(geojson_path)
print(f"Total Polygons in GeoJSON: {len(agents_gdf)}")

# Print bounding box before rescaling.
minx, miny, maxx, maxy = agents_gdf.total_bounds
print("Bounding Box (before rescaling):")
print(f"  minx={minx:.4f}, miny={miny:.4f}, maxx={maxx:.4f}, maxy={maxy:.4f}")

# Map the domain to 0–62 in x and 0–65 in y.
scale_x = 62.0 / (maxx - minx)
scale_y = 65.0 / (maxy - miny)
print(f"Scale factors: scale_x = {scale_x:.3e}, scale_y = {scale_y:.3e}")

# Translate so that (minx, miny) becomes (0,0), then scale.
agents_gdf["geometry"] = agents_gdf["geometry"].apply(
    lambda geom: scale_geom(translate(geom, xoff=-minx, yoff=-miny),
                              xfact=scale_x, yfact=scale_y, origin=(0,0))
)
new_minx, new_miny, new_maxx, new_maxy = agents_gdf.total_bounds
print("Bounding Box (after rescaling):")
print(f"  minx={new_minx:.4f}, miny={new_miny:.4f}, maxx={new_maxx:.4f}, maxy={new_maxy:.4f}")

global_bounds = (0, 0, 62, 65)

#############################################
# STEP 2: Define the Agent Class
#############################################
class FlowPolygonAgent:
    def __init__(self, unique_id, geometry, row_index, col_index):
        self.unique_id = unique_id
        self.geometry = geometry
        self.row_index = row_index
        self.col_index = col_index
        # Each edge will be a reference into the global dictionary.
        # For vertical edges we store flux in "vx"; for horizontal edges, in "vy".
        self.velocity_n = None  # North edge (horizontal; uses vy)
        self.velocity_e = None  # East edge (vertical; uses vx)
        self.velocity_s = None  # South edge (horizontal; uses vy)
        self.velocity_w = None  # West edge (vertical; uses vx)
        self.source = False     # If True, this cell is a source and its injection edges remain locked

#############################################
# Build agents from the GeoDataFrame.
#############################################
agents = []
for idx, row in agents_gdf.iterrows():
    agent = FlowPolygonAgent(
        unique_id = row["id"],
        geometry = row["geometry"],
        row_index = row["row_index"],
        col_index = row["col_index"]
    )
    agents.append(agent)
print(f"Created {len(agents)} agent objects.")

#############################################
# STEP 3: Build a Global Dictionary of Unique Edges
#############################################
# We now determine whether an edge is a boundary by checking if a neighboring cell exists.
def assign_shared_edges(agents, global_bounds):
    # Build a dictionary mapping (row_index, col_index) to agent.
    cell_map = {}
    for agent in agents:
        cell_map[(agent.row_index, agent.col_index)] = agent

    unique_edges = {}
    for agent in agents:
        minx, miny, maxx, maxy = agent.geometry.bounds

        # WEST edge: key = ("V", row, col) where the edge is the west edge of the cell.
        key_w = ("V", agent.row_index, agent.col_index)
        # Check if a neighbor exists to the west (i.e. at (row, col-1)).
        locked_w = ( (agent.row_index, agent.col_index - 1) not in cell_map )
        if key_w not in unique_edges:
            unique_edges[key_w] = {
                "vx": 0.0,
                "locked": locked_w,
                "midpoint": (minx, (miny+maxy)/2)
            }
        agent.velocity_w = unique_edges[key_w]

        # EAST edge: key = ("V", row, col+1)
        key_e = ("V", agent.row_index, agent.col_index+1)
        locked_e = ( (agent.row_index, agent.col_index+1) not in cell_map )
        if key_e not in unique_edges:
            unique_edges[key_e] = {
                "vx": 0.0,
                "locked": locked_e,
                "midpoint": (maxx, (miny+maxy)/2)
            }
        agent.velocity_e = unique_edges[key_e]

        # NORTH edge: key = ("H", row, col)
        key_n = ("H", agent.row_index, agent.col_index)
        locked_n = ( (agent.row_index-1, agent.col_index) not in cell_map )
        if key_n not in unique_edges:
            unique_edges[key_n] = {
                "vy": 0.0,
                "locked": locked_n,
                "midpoint": ((minx+maxx)/2, maxy)
            }
        agent.velocity_n = unique_edges[key_n]

        # SOUTH edge: key = ("H", row+1, col)
        key_s = ("H", agent.row_index+1, agent.col_index)
        locked_s = ( (agent.row_index+1, agent.col_index) not in cell_map )
        if key_s not in unique_edges:
            unique_edges[key_s] = {
                "vy": 0.0,
                "locked": locked_s,
                "midpoint": ((minx+maxx)/2, miny)
            }
        agent.velocity_s = unique_edges[key_s]

    print("Assigned shared edges using canonical keys based on neighbor existence.")
    # Optionally, print a few keys:
    for k, v in list(unique_edges.items())[:10]:
        print(f"  {k}: {v}")
    return unique_edges

global_edges = assign_shared_edges(agents, global_bounds)

#############################################
# STEP 4: Initialize Source Vectors
#############################################
def initialize_source_vectors(agents, source_ids, magnitude, bearing_degrees):
    """
    For each agent whose unique_id is in source_ids, update its injection vector on valid (unlocked) edges.
    For vertical edges (west/east) update only the x component (vx);
    for horizontal edges (north/south) update only the y component (vy).
    After injection, lock those edges so they are not modified.
    """
    br = math.radians(360 - bearing_degrees)
    inj_vx = magnitude * math.cos(br)
    inj_vy = magnitude * math.sin(br)
    print(f"Injection vector: ({inj_vx:.2f}, {inj_vy:.2f})")
    eps = 1e-8
    for agent in agents:
        if agent.unique_id in source_ids:
            # For vertical edges: update vx if unlocked and currently zero.
            if agent.velocity_w and (not agent.velocity_w["locked"]) and abs(agent.velocity_w["vx"]) < eps:
                agent.velocity_w["vx"] = inj_vx
                agent.velocity_w["locked"] = True
            if agent.velocity_e and (not agent.velocity_e["locked"]) and abs(agent.velocity_e["vx"]) < eps:
                agent.velocity_e["vx"] = inj_vx
                agent.velocity_e["locked"] = True
            # For horizontal edges: update vy if unlocked and currently zero.
            if agent.velocity_n and (not agent.velocity_n["locked"]) and abs(agent.velocity_n["vy"]) < eps:
                agent.velocity_n["vy"] = inj_vy
                agent.velocity_n["locked"] = True
            if agent.velocity_s and (not agent.velocity_s["locked"]) and abs(agent.velocity_s["vy"]) < eps:
                agent.velocity_s["vy"] = inj_vy
                agent.velocity_s["locked"] = True
            agent.source = True
    print("Initialized source vectors for source agents (edges updated and locked).")

# Define source cell IDs (adjust as needed)
source_ids = {4570, 4648, 4726, 4492, 4414}
initialize_source_vectors(agents, source_ids, magnitude=2, bearing_degrees=140)

#############################################
# STEP 5: Red–Black Classification and Checkerboard Plot
#############################################
def classify_cells(agents):
    groups = {"red": [], "black": [], "source": []}
    for agent in agents:
        if agent.source:
            groups["source"].append(agent)
        else:
            if (agent.row_index + agent.col_index) % 2 == 0:
                groups["red"].append(agent)
            else:
                groups["black"].append(agent)
    return groups

def plot_cell_classification(agents):
    groups = classify_cells(agents)
    red_x, red_y = [], []
    black_x, black_y = [], []
    source_x, source_y = [], []
    for agent in groups["red"]:
        red_x.append(agent.geometry.centroid.x)
        red_y.append(agent.geometry.centroid.y)
    for agent in groups["black"]:
        black_x.append(agent.geometry.centroid.x)
        black_y.append(agent.geometry.centroid.y)
    for agent in groups["source"]:
        source_x.append(agent.geometry.centroid.x)
        source_y.append(agent.geometry.centroid.y)
    plt.figure(figsize=(10,10))
    plt.scatter(red_x, red_y, color="red", label="Red cells", s=20)
    plt.scatter(black_x, black_y, color="black", label="Black cells", s=20)
    plt.scatter(source_x, source_y, color="purple", label="Source cells", s=20)
    plt.title("Cell Classification: Checkerboard")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

plot_cell_classification(agents)

#############################################
# STEP 6: Gauss–Seidel Incompressibility (Red–Black Iteration)
#############################################
def do_incompressibility_iteration(cells):
    """
    For each non–source cell in 'cells', compute the net divergence as the sum of its four edge fluxes.
    For vertical edges, use the stored vx; for horizontal edges, the stored vy.
    Then, distribute the correction evenly among the unlocked edges.
    """
    for cell in cells:
        if cell.source:
            continue
        flux_n = cell.velocity_n["vy"] if (cell.velocity_n and not isinstance(cell.velocity_n["vy"], str)) else 0.0
        flux_e = cell.velocity_e["vx"] if (cell.velocity_e and not isinstance(cell.velocity_e["vx"], str)) else 0.0
        flux_w = cell.velocity_w["vx"] if (cell.velocity_w and not isinstance(cell.velocity_w["vx"], str)) else 0.0
        flux_s = cell.velocity_s["vy"] if (cell.velocity_s and not isinstance(cell.velocity_s["vy"], str)) else 0.0
        divergence = flux_n + flux_e + flux_w + flux_s

        valid_edges = []
        if cell.velocity_n and (not cell.velocity_n["locked"]):
            valid_edges.append(("N", cell.velocity_n))
        if cell.velocity_e and (not cell.velocity_e["locked"]):
            valid_edges.append(("E", cell.velocity_e))
        if cell.velocity_w and (not cell.velocity_w["locked"]):
            valid_edges.append(("W", cell.velocity_w))
        if cell.velocity_s and (not cell.velocity_s["locked"]):
            valid_edges.append(("S", cell.velocity_s))
        n_avail = len(valid_edges)
        if n_avail == 0:
            continue
        correction = divergence / n_avail
        for direction, edge in valid_edges:
            if direction in ("N", "S"):
                edge["vy"] -= correction
            else:
                edge["vx"] -= correction

def do_red_black_incompressibility(agents, num_steps=5):
    groups = classify_cells(agents)
    for step in range(num_steps):
        print(f"Time step {step+1}")
        do_incompressibility_iteration(groups["red"])
        do_incompressibility_iteration(groups["black"])

do_red_black_incompressibility(agents, num_steps=20)

#############################################
# STEP 7: Plot Global Edges (Tiles and Colored by Direction)
#############################################
def plot_global_edges(global_edges, agents):
    """
    Plots every edge from the global dictionary over the cell boundaries.
    For vertical edges (keys starting with "V"), we draw an arrow horizontally (using vx);
    for horizontal edges (keys starting with "H"), we draw an arrow vertically (using vy).
    Locked edges (boundaries or source-injected) are skipped.
    Edges with near–zero magnitude are not drawn.
    """
    plt.figure(figsize=(12,12))
    # Draw cell boundaries:
    for agent in agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color="black", linewidth=0.5)
    # Draw each edge arrow:
    for key, val in global_edges.items():
        if val["locked"]:
            continue
        if key[0] == "V":
            mag = abs(val["vx"])
            if mag < 1e-9:
                continue
            x, y = val["midpoint"]
            color = "green"  # vertical edge: flux in x
            plt.arrow(x, y, val["vx"], 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
        elif key[0] == "H":
            mag = abs(val["vy"])
            if mag < 1e-9:
                continue
            x, y = val["midpoint"]
            color = "red"  # horizontal edge: flux in y
            plt.arrow(x, y, 0, val["vy"], head_width=0.1, head_length=0.1, fc=color, ec=color)
    plt.title("Global Unique Edge Velocities (After Incompressibility)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

plot_global_edges(global_edges, agents)


def compute_cell_intensities(agents):
    """
    For each cell (agent), compute the sum of the absolute fluxes from its four edges.
    For vertical edges (east and west), we use the absolute value of the 'vx' component.
    For horizontal edges (north and south), we use the absolute value of the 'vy' component.
    Edges marked as boundaries (with a non‐numeric value) are skipped.
    Returns a list of intensities (one per agent).
    """
    intensities = []
    for agent in agents:
        intensity = 0.0
        if agent.velocity_n and not isinstance(agent.velocity_n.get("vy"), str):
            intensity += abs(agent.velocity_n["vy"])
        if agent.velocity_s and not isinstance(agent.velocity_s.get("vy"), str):
            intensity += abs(agent.velocity_s["vy"])
        if agent.velocity_e and not isinstance(agent.velocity_e.get("vx"), str):
            intensity += abs(agent.velocity_e["vx"])
        if agent.velocity_w and not isinstance(agent.velocity_w.get("vx"), str):
            intensity += abs(agent.velocity_w["vx"])
        intensities.append(intensity)
    return intensities

def plot_tile_heatmap(agents, cmap='hot'):
    """
    Plots the heatmap using the actual tile polygons.
    Each agent's geometry is colored by the sum of the absolute fluxes on its edges.
    """
    intensities = compute_cell_intensities(agents)
    # Create a GeoDataFrame using the agents' geometries.
    df = pd.DataFrame({"intensity": intensities})
    gdf = gpd.GeoDataFrame(df, geometry=[agent.geometry for agent in agents])
    
    # Plot with a legend. You can adjust edgecolor and linewidth as desired.
    ax = gdf.plot(column="intensity", cmap=cmap, legend=True, edgecolor="black", figsize=(12,12))
    ax.set_title("Heatmap: Sum of Edge Flux Magnitudes per Cell")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    plt.show()

# To run the heatmap plot:
plot_tile_heatmap(agents)
