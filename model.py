import math
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from shapely.affinity import translate, scale as scale_geom
from scipy.interpolate import griddata
import mesa

#############################################
# STEP 1: Load GeoJSON and Rescale Geometries
#############################################
geojson_path = "malpeque_tiles.geojson"  # Local file
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

# Translate so that (minx, miny) becomes (0,0) then scale.
agents_gdf["geometry"] = agents_gdf["geometry"].apply(
    lambda geom: scale_geom(translate(geom, xoff=-minx, yoff=-miny),
                              xfact=scale_x, yfact=scale_y, origin=(0, 0))
)
new_minx, new_miny, new_maxx, new_maxy = agents_gdf.total_bounds
print("Bounding Box (after rescaling):")
print(f"  minx={new_minx:.4f}, miny={new_miny:.4f}, maxx={new_maxx:.4f}, maxy={new_maxx:.4f}")

global_bounds = (0, 0, 62, 65)

#############################################
# STEP 2: Define the Agent Class (Mesa Agent)
#############################################
class FlowPolygonAgent(mesa.Agent):
    def __init__(self, unique_id, geometry, row_index, col_index, model):
        # Old-style explicit initialization without super()
        self.unique_id = unique_id
        self.model = model
        
        self.geometry = geometry
        self.row_index = row_index
        self.col_index = col_index
        # Each edge is a reference into the global dictionary.
        # Vertical edges store their primary flux in "vx"; horizontal edges store flux in "vy".
        self.velocity_n = None  # North edge: key = ("H", row_index, col_index)
        self.velocity_e = None  # East edge: key = ("V", row_index, col_index+1)
        self.velocity_s = None  # South edge: key = ("H", row_index+1, col_index)
        self.velocity_w = None  # West edge: key = ("V", row_index, col_index)
        self.source = False     # If True, this cell is a source (its injection edges remain locked)

    def step(self):
        # This agent does not update its own state per se;
        # all simulation updates occur in the global edge dictionary.
        pass

#############################################
# STEP 3: Build Agents from the GeoDataFrame
#############################################
agents = []
for idx, row in agents_gdf.iterrows():
    agent = FlowPolygonAgent(
        unique_id=row["id"],
        geometry=row["geometry"],
        row_index=row["row_index"],
        col_index=row["col_index"],
        model=None  # Will be assigned when added to the model.
    )
    agents.append(agent)
print(f"Created {len(agents)} agent objects.")

#############################################
# STEP 4: Build a Global Dictionary of Unique Edges
#############################################
def assign_shared_edges(agents, global_bounds):
    """
    For each agent (cell), assign canonical keys for its edges.
    For vertical edges:
         West edge: key = ("V", row_index, col_index)
         East edge: key = ("V", row_index, col_index+1)
    For horizontal edges:
         North edge: key = ("H", row_index, col_index)
         South edge: key = ("H", row_index+1, col_index)
    Each edge dictionary reserves both "vx" and "vy" (the secondary component is initially 0.0)
    and a "locked" flag. An edge is locked if no neighbor exists in that direction.
    """
    # Build a mapping from (row_index, col_index) to agent.
    cell_map = {}
    for agent in agents:
        cell_map[(agent.row_index, agent.col_index)] = agent

    unique_edges = {}
    for agent in agents:
        minx, miny, maxx, maxy = agent.geometry.bounds

        # WEST edge.
        key_w = ("V", agent.row_index, agent.col_index)
        locked_w = ((agent.row_index, agent.col_index - 1) not in cell_map)
        if key_w not in unique_edges:
            unique_edges[key_w] = {
                "vx": 0.0,
                "vy": 0.0,  # reserved for missing component interpolation
                "locked": locked_w,
                "midpoint": (minx, (miny+maxy)/2)
            }
        agent.velocity_w = unique_edges[key_w]

        # EAST edge.
        key_e = ("V", agent.row_index, agent.col_index+1)
        locked_e = ((agent.row_index, agent.col_index+1) not in cell_map)
        if key_e not in unique_edges:
            unique_edges[key_e] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_e,
                "midpoint": (maxx, (miny+maxy)/2)
            }
        agent.velocity_e = unique_edges[key_e]

        # NORTH edge.
        key_n = ("H", agent.row_index, agent.col_index)
        locked_n = ((agent.row_index-1, agent.col_index) not in cell_map)
        if key_n not in unique_edges:
            unique_edges[key_n] = {
                "vx": 0.0,  # reserved for missing component
                "vy": 0.0,
                "locked": locked_n,
                "midpoint": ((minx+maxx)/2, maxy)
            }
        agent.velocity_n = unique_edges[key_n]

        # SOUTH edge.
        key_s = ("H", agent.row_index+1, agent.col_index)
        locked_s = ((agent.row_index+1, agent.col_index) not in cell_map)
        if key_s not in unique_edges:
            unique_edges[key_s] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_s,
                "midpoint": ((minx+maxx)/2, miny)
            }
        agent.velocity_s = unique_edges[key_s]

    print("Assigned shared edges using canonical keys.")
    for k, v in list(unique_edges.items())[:10]:
        print(f"  {k}: {v}")
    return unique_edges

global_edges = assign_shared_edges(agents, global_bounds)

#############################################
# STEP 5: Initialize Source Vectors
#############################################
def initialize_source_vectors(agents, source_ids, magnitude, bearing_degrees):
    """
    For each agent whose unique_id is in source_ids, inject a prescribed flux on its edges.
    For vertical edges (west/east) update only "vx" and for horizontal edges (north/south) update only "vy".
    After injection, the edge is locked.
    """
    br = math.radians(360 - bearing_degrees)
    inj_vx = magnitude * math.cos(br)
    inj_vy = magnitude * math.sin(br)
    print(f"Injection vector: ({inj_vx:.2f}, {inj_vy:.2f})")
    eps = 1e-8
    for agent in agents:
        if agent.unique_id in source_ids:
            if agent.velocity_w and (not agent.velocity_w["locked"]) and abs(agent.velocity_w["vx"]) < eps:
                agent.velocity_w["vx"] = inj_vx
                agent.velocity_w["locked"] = True
            if agent.velocity_e and (not agent.velocity_e["locked"]) and abs(agent.velocity_e["vx"]) < eps:
                agent.velocity_e["vx"] = inj_vx
                agent.velocity_e["locked"] = True
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
# STEP 6: Define Projection (Incompressibility) Functions
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

def do_incompressibility_iteration(cells):
    for cell in cells:
        if cell.source:
            continue
        
        # Calculate fluxes
        flux_n = cell.velocity_n["vy"] if (cell.velocity_n and not isinstance(cell.velocity_n["vy"], str)) else 0.0
        flux_e = cell.velocity_e["vx"] if (cell.velocity_e and not isinstance(cell.velocity_e["vx"], str)) else 0.0
        flux_w = cell.velocity_w["vx"] if (cell.velocity_w and not isinstance(cell.velocity_w["vx"], str)) else 0.0
        flux_s = cell.velocity_s["vy"] if (cell.velocity_s and not isinstance(cell.velocity_s["vy"], str)) else 0.0
        divergence = flux_n + flux_e + flux_w + flux_s

        # Skip cells with negligible divergence
        if abs(divergence) < 1e-10:
            continue

        # Find valid edges to adjust
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
            
        # Calculate and apply correction
        correction = divergence / n_avail
        for direction, edge in valid_edges:
            if direction in ("N", "S"):
                edge["vy"] -= correction
            else:
                edge["vx"] -= correction

def do_red_black_incompressibility(agents, num_steps=5):
    groups = classify_cells(agents)
    for step in range(num_steps):
        do_incompressibility_iteration(groups["red"])
        do_incompressibility_iteration(groups["black"])

#############################################
# STEP 7: Define Semi-Lagrangian Advection Functions
#############################################
def advect_velocities(global_edges, agents, dt):
    """
    Semi-Lagrangian advection for the staggered grid veloscities.
    This function traces particles backwards in time and interpolates velocities.
    """
    # First identify source edges that should be preserved
    source_edges = set()
    for agent in agents:
        if agent.source:
            if agent.velocity_n: source_edges.add(id(agent.velocity_n))
            if agent.velocity_s: source_edges.add(id(agent.velocity_s))
            if agent.velocity_e: source_edges.add(id(agent.velocity_e))
            if agent.velocity_w: source_edges.add(id(agent.velocity_w))
    
    # Build velocity field data for interpolation
    h_positions = []  # Horizontal edge positions
    h_values = []     # Vertical velocity components (vy)
    v_positions = []  # Vertical edge positions
    v_values = []     # Horizontal velocity components (vx)
    
    # Collect velocity data from all valid edges
    for key, edge in global_edges.items():
        # Skip edges with string velocity components
        if isinstance(edge["vx"], str) or isinstance(edge["vy"], str):
            continue
        
        # Include all valid velocities for interpolation
        if key[0] == "H":  # Horizontal edge
            h_positions.append(edge["midpoint"])
            h_values.append(edge["vy"])
        elif key[0] == "V":  # Vertical edge
            v_positions.append(edge["midpoint"])
            v_values.append(edge["vx"])
    
    # Create new dictionary for updated edges
    new_edges = {}
    
    # Process each edge
    for key, edge in global_edges.items():
        # 1. Copy string-valued edges (boundaries) directly
        if isinstance(edge["vx"], str) or isinstance(edge["vy"], str):
            new_edges[key] = edge.copy()
            continue
        
        # 2. Copy source edges directly (preserve their velocities)
        if id(edge) in source_edges:
            new_edges[key] = edge.copy()
            continue
        
        # 3. For normal edges, perform semi-Lagrangian advection
        x, y = edge["midpoint"]
        
        if key[0] == "H":  # Horizontal edge (has primary vy component)
            # Get secondary velocity component at this point
            vx_here = griddata(v_positions, v_values, (x, y), 
                               method='linear', fill_value=0.0)
            
            # Trace backward to find the departure point
            x_departure = x - dt * vx_here
            y_departure = y - dt * edge["vy"]
            
            # Sample the velocity at the departure point
            vy_departure = griddata(h_positions, h_values, (x_departure, y_departure), 
                                   method='linear', fill_value=edge["vy"])
            
            # Create updated edge
            new_edge = edge.copy()
            new_edge["vy"] = vy_departure
            new_edges[key] = new_edge
            
        elif key[0] == "V":  # Vertical edge (has primary vx component)
            # Get secondary velocity component at this point
            vy_here = griddata(h_positions, h_values, (x, y), 
                               method='linear', fill_value=0.0)
            
            # Trace backward to find the departure point
            x_departure = x - dt * edge["vx"]
            y_departure = y - dt * vy_here
            
            # Sample the velocity at the departure point
            vx_departure = griddata(v_positions, v_values, (x_departure, y_departure), 
                                   method='linear', fill_value=edge["vx"])
            
            # Create updated edge
            new_edge = edge.copy()
            new_edge["vx"] = vx_departure
            new_edges[key] = new_edge
    
    return new_edges

#############################################
# STEP 8: Define Visualization Functions
#############################################
def plot_global_edges_diagnostics(global_dict, agents):
    plt.figure(figsize=(12,12))
    # Draw cell boundaries.
    for agent in agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color="black", linewidth=0.5)
    # Draw edge arrows.
    for key, val in global_dict.items():
        # Skip boundary edges.
        if isinstance(val["vx"], str) or isinstance(val["vy"], str):
            continue
        if key[0] in ("W", "E"):
            x, y = val["midpoint"]
            plt.arrow(x, y, val["vx"], 0, head_width=0.3, head_length=0.3, fc="green", ec="green")
        elif key[0] in ("N", "S"):
            x, y = val["midpoint"]
            plt.arrow(x, y, 0, val["vy"], head_width=0.3, head_length=0.3, fc="red", ec="red")
    plt.title("Global Edge Velocities After Advection")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

def compute_cell_intensities(agents):
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
    intensities = compute_cell_intensities(agents)
    df = pd.DataFrame({"intensity": intensities})
    gdf = gpd.GeoDataFrame(df, geometry=[agent.geometry for agent in agents])
    ax = gdf.plot(column="intensity", cmap=cmap, legend=True, edgecolor="black", figsize=(12,12))
    ax.set_title("Heatmap: Sum of Edge Flux Magnitudes per Cell")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    plt.show()

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
        if val.get("locked", False):
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

#############################################
# STEP 9: Define the Mesa Model and Simulation Loop
#############################################
class FlowModel(mesa.Model):
    def __init__(self):
        super().__init__()  # Call Mesa's Model.__init__
        self.dt = 0.1
        self.flow_agents = agents  # Renamed from self.agents to self.flow_agents
        self.global_edges = global_edges
        # No scheduler needed for now - we'll control steps manually
    
    def step(self):
        # 1. Advect the global edges.
        self.global_edges = advect_velocities(self.global_edges, self.flow_agents, self.dt)
        # 2. Perform one iteration of red–black projection.
        do_red_black_incompressibility(self.flow_agents, num_steps=1)
    
    def advect_only_step(self):
        # Only perform advection, no projection
        self.global_edges = advect_velocities(self.global_edges, self.flow_agents, self.dt)

#############################################
# STEP 10: Run the Simulation Loop and Produce Diagnostics
#############################################
model = FlowModel()
num_steps = 5 

# Skip initial projection phase entirely
# Start directly with advection-only simulation

print("\n=== Running pure advection-only simulation (no projection) ===")
for i in range(num_steps):
    print(f"Advection-only step {i+1}")
    model.advect_only_step()

# Add visualization with enhanced diagnostics
print("\n=== Visualizing results (pure advection only) ===")
plot_global_edges(model.global_edges, model.flow_agents)
plot_tile_heatmap(model.flow_agents)

# Print velocity magnitude statistics
h_vals = [edge["vy"] for key, edge in model.global_edges.items() 
          if key[0] in ("N", "S") and not isinstance(edge["vy"], str)]
v_vals = [edge["vx"] for key, edge in model.global_edges.items() 
          if key[0] in ("W", "E") and not isinstance(edge["vx"], str)]

if h_vals:
    print(f"Horizontal velocity stats: min={min(h_vals):.6f}, max={max(h_vals):.6f}, mean={sum(h_vals)/len(h_vals):.6f}")
if v_vals:
    print(f"Vertical velocity stats: min={min(v_vals):.6f}, max={max(v_vals)::.6f}, mean={sum(v_vals)/len(v_vals):.6f}")