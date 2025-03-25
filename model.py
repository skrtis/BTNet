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
    Semi-Lagrangian advection for staggered grid velocities.
    
    Args:
        global_edges: Dictionary of all velocity edges indexed by (type, i, j)
        agents: List of flow agents
        dt: Time step size
    
    Returns:
        Dictionary of updated edges after advection
    """
    # Step 1: Identify source edges that should be preserved
    # These are edges attached to source cells, which inject momentum
    source_edges = set()
    for agent in agents:
        if agent.source:
            if agent.velocity_n: source_edges.add(id(agent.velocity_n))
            if agent.velocity_s: source_edges.add(id(agent.velocity_s))
            if agent.velocity_e: source_edges.add(id(agent.velocity_e))
            if agent.velocity_w: source_edges.add(id(agent.velocity_w))
    
    # Step 2: Create new dictionary to store updated edges
    new_edges = {}
    
    # Step 3: Copy boundary edges and source edges directly (no advection)
    for key, edge in global_edges.items():
        if isinstance(edge["vx"], str) or isinstance(edge["vy"], str) or id(edge) in source_edges:
            new_edges[key] = edge.copy()
    
    # Step 4: Define interpolation functions for staggered grid
    # For horizontal edges (vy component), we need to account for the half-cell offset in x
    def interpolate_vy(x, y):
        """
        Interpolate vy from horizontal edges on a staggered grid.
        Horizontal edges are located at (j+0.5, i) where j is the column and i is the row.
        """
        # Adjust for staggering: horizontal edges are at (j+0.5, i)
        j0 = int(x - 0.5)    # Floor of (x - 0.5) to get the column index
        i0 = int(y)          # Floor of y to get the row index
        j1 = j0 + 1          # Next column
        i1 = i0 + 1          # Next row
        
        # Calculate fractional position within the cell
        s = (x - 0.5) - j0   # Fractional part in x, adjusted for staggering
        t = y - i0           # Fractional part in y
        
        # Safety check for domain boundaries
        if i0 < 0 or j0 < 0:
            return 0.0
            
        # Get velocities at the four surrounding horizontal edges
        v00 = global_edges.get(("H", i0, j0), {}).get("vy", 0.0)
        v01 = global_edges.get(("H", i0, j1), {}).get("vy", 0.0)
        v10 = global_edges.get(("H", i1, j0), {}).get("vy", 0.0)
        v11 = global_edges.get(("H", i1, j1), {}).get("vy", 0.0)
        
        # Handle string values (boundaries)
        if isinstance(v00, str): v00 = 0.0
        if isinstance(v01, str): v01 = 0.0
        if isinstance(v10, str): v10 = 0.0
        if isinstance(v11, str): v11 = 0.0
        
        # Perform bilinear interpolation
        # (1-s)(1-t)*v00 + s*(1-t)*v01 + (1-s)*t*v10 + s*t*v11
        return (1-s)*(1-t)*v00 + s*(1-t)*v01 + (1-s)*t*v10 + s*t*v11
    
    # For vertical edges (vx component), we need to account for the half-cell offset in y
    def interpolate_vx(x, y):
        """
        Interpolate vx from vertical edges on a staggered grid.
        Vertical edges are located at (j, i+0.5) where j is the column and i is the row.
        """
        # Adjust for staggering: vertical edges are at (j, i+0.5)
        j0 = int(x)          # Floor of x to get the column index
        i0 = int(y - 0.5)    # Floor of (y - 0.5) to get the row index
        j1 = j0 + 1          # Next column
        i1 = i0 + 1          # Next row
        
        # Calculate fractional position within the cell
        s = x - j0           # Fractional part in x
        t = (y - 0.5) - i0   # Fractional part in y, adjusted for staggering
        
        # Safety check for domain boundaries
        if i0 < 0 or j0 < 0:
            return 0.0
            
        # Get velocities at the four surrounding vertical edges
        v00 = global_edges.get(("V", i0, j0), {}).get("vx", 0.0)
        v01 = global_edges.get(("V", i0, j1), {}).get("vx", 0.0)
        v10 = global_edges.get(("V", i1, j0), {}).get("vx", 0.0)
        v11 = global_edges.get(("V", i1, j1), {}).get("vx", 0.0)
        
        # Handle string values (boundaries)
        if isinstance(v00, str): v00 = 0.0
        if isinstance(v01, str): v01 = 0.0
        if isinstance(v10, str): v10 = 0.0
        if isinstance(v11, str): v11 = 0.0
        
        # Perform bilinear interpolation
        # (1-s)(1-t)*v00 + s*(1-t)*v01 + (1-s)*t*v10 + s*t*v11
        return (1-s)*(1-t)*v00 + s*(1-t)*v01 + (1-s)*t*v10 + s*t*v11
    
    # Step 5: Process horizontal edges (vy component)
    # We need to trace backward to find where this velocity came from
    for key, edge in global_edges.items():
        # Skip edges that have already been processed
        if key in new_edges:
            continue
            
        # Process horizontal edges (where vy is stored)
        if key[0] == "H":
            # Get the midpoint of this edge
            x, y = edge["midpoint"]
            
            # 5a. Get the vx component at this horizontal edge by interpolation
            vx_here = interpolate_vx(x, y)
            
            # 5b. Trace backward to find the departure point
            # Semi-Lagrangian advection traces particles backward along flow
            x_departure = x - dt * vx_here       # Move backward in x direction
            y_departure = y - dt * edge["vy"]    # Move backward in y direction
            
            # 5c. Sample vy at the departure point using interpolation
            vy_departure = interpolate_vy(x_departure, y_departure)
            
            # 5d. Create the updated edge with new velocity
            new_edges[key] = {
                "vx": edge["vx"],             # Keep the secondary component unchanged
                "vy": vy_departure,           # Update the primary component
                "locked": edge.get("locked", False),
                "midpoint": edge["midpoint"]
            }
    
    # Step 6: Process vertical edges (vx component)
    # Similar to horizontal edges, but for vx component
    for key, edge in global_edges.items():
        # Skip edges that have already been processed
        if key in new_edges:
            continue
            
        # Process vertical edges (where vx is stored)
        if key[0] == "V":
            # Get the midpoint of this edge
            x, y = edge["midpoint"]
            
            # 6a. Get the vy component at this vertical edge by interpolation
            vy_here = interpolate_vy(x, y)
            
            # 6b. Trace backward to find the departure point
            x_departure = x - dt * edge["vx"]    # Move backward in x direction
            y_departure = y - dt * vy_here       # Move backward in y direction
            
            # 6c. Sample vx at the departure point using interpolation
            vx_departure = interpolate_vx(x_departure, y_departure)
            
            # 6d. Create the updated edge with new velocity
            new_edges[key] = {
                "vx": vx_departure,           # Update the primary component
                "vy": edge["vy"],             # Keep the secondary component unchanged
                "locked": edge.get("locked", False),
                "midpoint": edge["midpoint"]
            }
    
    # Step 7: Provide some statistics for debugging
    count_nonzero_h = sum(1 for k, e in new_edges.items() 
                       if k[0] == "H" and abs(e.get("vy", 0.0)) > 1e-10)
    count_nonzero_v = sum(1 for k, e in new_edges.items() 
                       if k[0] == "V" and abs(e.get("vx", 0.0)) > 1e-10)
    print(f"After advection: {count_nonzero_h} horizontal edges and {count_nonzero_v} vertical edges have non-zero velocity")
    
    return new_edges

#############################################
# STEP 8: Define Visualization Functions
#############################################

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

def plot_global_edges_enhanced(global_edges, agents, threshold=1e-12, scale=10.0):
    """
    Enhanced visualization of edge velocities with adjustable threshold and scaling.
    
    Args:
        global_edges: Dictionary of edge velocities
        agents: List of agent objects
        threshold: Minimum velocity magnitude to display
        scale: Scale factor for arrow lengths
    """
    plt.figure(figsize=(15, 15))
    
    # Draw cell boundaries
    for agent in agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color="black", linewidth=0.2)
    
    # Find source agents for highlighting
    source_agents = [agent for agent in agents if agent.source]
    
    # Count displayed edges
    displayed_h = 0
    displayed_v = 0
    
    # Draw each edge arrow with scaled velocity
    for key, val in global_edges.items():
        if val.get("locked", False):
            continue
            
        x, y = val["midpoint"]
        
        if key[0] == "V":
            mag = abs(val["vx"])
            if mag > threshold:
                displayed_v += 1
                color = "green"  # vertical edge: flux in x
                plt.arrow(x, y, val["vx"] * scale, 0, 
                         head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.7)
                
        elif key[0] == "H":
            mag = abs(val["vy"])
            if mag > threshold:
                displayed_h += 1
                color = "red"  # horizontal edge: flux in y
                plt.arrow(x, y, 0, val["vy"] * scale, 
                         head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.7)
    
    # Highlight source cells
    for agent in source_agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color="orange", linewidth=1.0)
    
    # Print statistics on the plot
    plt.text(5, 5, f"Displaying {displayed_h} horizontal and {displayed_v} vertical edges",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f"Edge Velocities (Scale={scale:.1f}, Threshold={threshold:.1e})")
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
def analyze_grid_indices(agents):
    """Analyze the distribution of row_index and col_index values"""
    row_indices = [agent.row_index for agent in agents]
    col_indices = [agent.col_index for agent in agents]
    
    # Get the range of indices
    min_row = min(row_indices)
    max_row = max(row_indices)
    min_col = min(col_indices)
    max_col = max(col_indices)
    
    # Calculate statistics
    unique_rows = len(set(row_indices))
    unique_cols = len(set(col_indices))
    
    # Create a sparse matrix representation to check grid density
    grid_points = set((agent.row_index, agent.col_index) for agent in agents)
    
    print("\n=== Grid Index Analysis ===")
    print(f"Row indices range: {min_row} to {max_row} (total span: {max_row-min_row+1})")
    print(f"Column indices range: {min_col} to {max_col} (total span: {max_col-min_col+1})")
    print(f"Unique row values: {unique_rows}")
    print(f"Unique column values: {unique_cols}")
    print(f"Total cells: {len(agents)}")
    print(f"Grid density: {len(agents)/(unique_rows*unique_cols):.2%}")
    
    # Visualize the grid index distribution
    plt.figure(figsize=(12, 12))
    plt.scatter([agent.col_index for agent in agents], 
                [agent.row_index for agent in agents],
                marker='s', s=10, color='blue', alpha=0.5)
    plt.title("Grid Index Distribution")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Return the grid bounds for reference
    return (min_row, min_col, max_row, max_col)

# Add this to your code before simulation
grid_bounds = analyze_grid_indices(agents)

model = FlowModel()
num_steps = 1

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
          if key[0] == "H" and not isinstance(edge["vy"], str)]
v_vals = [edge["vx"] for key, edge in model.global_edges.items() 
          if key[0] == "V" and not isinstance(edge["vx"], str)]

if h_vals:
    print(f"Horizontal velocity stats: min={min(h_vals):.6f}, max={max(h_vals):.6f}, mean={sum(h_vals)/len(h_vals):.6f}")
if v_vals:
    print(f"Vertical velocity stats: min={min(v_vals):.6f}, max={max(v_vals):.6f}, mean={sum(v_vals)/len(v_vals):.6f}")

print("\n=== Enhanced visualization with lower threshold ===")
plot_global_edges_enhanced(model.global_edges, model.flow_agents, threshold=1e-12, scale=50.0)