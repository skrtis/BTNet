import json
import math
import mesa
import mesa_geo as mg
from shapely.geometry import shape, Point
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import numpy as np
from shapely.affinity import translate, scale as scale_geom
from scipy.interpolate import griddata

# ------------------------
# Load GeoJSON Data and Rescale Geometries
# ------------------------

geojson_path = "malpeque_tiles.geojson"  # Local file
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)
print(f"Total Polygons in GeoJSON: {len(geojson_data['features'])}")

# Load the data as a GeoDataFrame.
agents_gdf = gpd.read_file(geojson_path)

# Compute total bounds.
minx, miny, maxx, maxy = agents_gdf.total_bounds
x_range = maxx - minx
y_range = maxy - miny
# We want to rescale to a 62 x 65 grid.
scale_x = 62.0 / x_range
scale_y = 65.0 / y_range
print(f"Rescaling: x_range={x_range:.3e}, y_range={y_range:.3e}")
print(f"Scale factors: scale_x={scale_x:.3e}, scale_y={scale_y:.3e}")

# Translate so that minimum is (0,0), then scale.
agents_gdf["geometry"] = agents_gdf["geometry"].apply(
    lambda geom: scale_geom(translate(geom, xoff=-minx, yoff=-miny), 
                              xfact=scale_x, yfact=scale_y, origin=(0, 0))
)

# ------------------------
# Utility Function: Assign Shared Edge Velocities
# ------------------------
def assign_edge_velocities(agents):
    """
    For each agent (cell), assign four velocity vectors (north, east, south, west).
    If two adjacent cells share an edge, they reference the same velocity object.
    We assume cells are square and axis-aligned.
    
    Each edge is keyed by a tuple that uniquely identifies its location.
    """
    edge_dict = {}
    tol = 1e-6

    for agent in agents:
        minx, miny, maxx, maxy = agent.geometry.bounds
        # Round coordinates for robustness.
        minx_r = round(minx, 6)
        miny_r = round(miny, 6)
        maxx_r = round(maxx, 6)
        maxy_r = round(maxy, 6)
        
        # Define keys for each edge.
        west_key  = ("W", minx_r, miny_r, maxy_r)
        east_key  = ("E", maxx_r, miny_r, maxy_r)
        south_key = ("S", miny_r, minx_r, maxx_r)
        north_key = ("N", maxy_r, minx_r, maxx_r)
        
        if west_key in edge_dict:
            agent.velocity_w = edge_dict[west_key]
        else:
            edge_dict[west_key] = {"vx": 0.0, "vy": 0.0}
            agent.velocity_w = edge_dict[west_key]
            
        if east_key in edge_dict:
            agent.velocity_e = edge_dict[east_key]
        else:
            edge_dict[east_key] = {"vx": 0.0, "vy": 0.0}
            agent.velocity_e = edge_dict[east_key]
            
        if south_key in edge_dict:
            agent.velocity_s = edge_dict[south_key]
        else:
            edge_dict[south_key] = {"vx": 0.0, "vy": 0.0}
            agent.velocity_s = edge_dict[south_key]
            
        if north_key in edge_dict:
            agent.velocity_n = edge_dict[north_key]
        else:
            edge_dict[north_key] = {"vx": 0.0, "vy": 0.0}
            agent.velocity_n = edge_dict[north_key]
    print("Assigned shared edge velocities to all agents.")
    return edge_dict

# ------------------------
# Function: Initialize Source Cells
# ------------------------
def initialize_currents(model, polygon_ids, magnitude, bearing_degrees):
    """
    Sets the initial flow for specified source cells by injecting a constant velocity.
    The injection is applied to specific edges of the cell (for example, the west and south edges)
    so that adjacent cells share the same edge velocity.
    """
    # Convert bearing to Cartesian components.
    bearing_radians = math.radians(360 - bearing_degrees)
    init_vx = magnitude * math.cos(bearing_radians)
    init_vy = magnitude * math.sin(bearing_radians)
    
    print(f"\nInitializing currents for {len(polygon_ids)} source cells...\n")
    found_tiles = 0
    for agent in model.polygons:
        if agent.unique_id in polygon_ids:
            # Inject on the west and south edges.
            agent.velocity_w["vx"] = init_vx
            agent.velocity_w["vy"] = init_vy
            agent.velocity_s["vx"] = init_vx
            agent.velocity_s["vy"] = init_vy
            agent.source = True
            found_tiles += 1
            print(f"Agent {agent.unique_id} → Source injection (west & south): ({init_vx:.2f}, {init_vy:.2f})")
    if found_tiles == 0:
        print("ERROR: No matching source cells found!")
    else:
        print(f"\nSuccessfully initialized {found_tiles} source cells.\n")

# ------------------------
# Semi-Lagrangian Advection for Edge Velocities with Bilinear Interpolation
# ------------------------
def advect_edge_velocities(model, dt):
    """
    Performs a semi-Lagrangian advection step on the edge velocities using bilinear interpolation.
    
    For each cell and for each edge, the function:
      1. Computes the midpoint of the edge.
      2. Uses the current edge velocity to backtrace a departure point.
      3. Interpolates the new edge velocity at that departure point using griddata.
      4. Updates the cell's edge velocity with the interpolated value.
      
    This version uses griddata with a query point wrapped in a list so that we can
    reliably extract a 2-element vector.
    """
    # Precompute donor data for each edge orientation.
    west_positions = []
    west_velocities = []
    east_positions = []
    east_velocities = []
    south_positions = []
    south_velocities = []
    north_positions = []
    north_velocities = []
    
    for cell in model.polygons:
        minx, miny, maxx, maxy = cell.geometry.bounds
        # West edge midpoint.
        west_positions.append((minx, (miny + maxy) / 2))
        west_velocities.append((cell.velocity_w["vx"], cell.velocity_w["vy"]))
        # East edge midpoint.
        east_positions.append((maxx, (miny + maxy) / 2))
        east_velocities.append((cell.velocity_e["vx"], cell.velocity_e["vy"]))
        # South edge midpoint.
        south_positions.append(((minx + maxx) / 2, miny))
        south_velocities.append((cell.velocity_s["vx"], cell.velocity_s["vy"]))
        # North edge midpoint.
        north_positions.append(((minx + maxx) / 2, maxy))
        north_velocities.append((cell.velocity_n["vx"], cell.velocity_n["vy"]))
    
    west_positions = np.array(west_positions)
    west_velocities = np.array(west_velocities)
    east_positions = np.array(east_positions)
    east_velocities = np.array(east_velocities)
    south_positions = np.array(south_positions)
    south_velocities = np.array(south_velocities)
    north_positions = np.array(north_positions)
    north_velocities = np.array(north_velocities)
    
    # Update each cell's edge velocities.
    for cell in model.polygons:
        minx, miny, maxx, maxy = cell.geometry.bounds
        
        # West edge:
        west_mid = np.array([minx, (miny + maxy) / 2])
        current_w = np.array([cell.velocity_w["vx"], cell.velocity_w["vy"]])
        departure_w = west_mid - dt * current_w
        new_w = griddata(west_positions, west_velocities, [departure_w], method='linear', fill_value=0.0)[0]
        cell.velocity_w["vx"], cell.velocity_w["vy"] = new_w
        
        # East edge:
        east_mid = np.array([maxx, (miny + maxy) / 2])
        current_e = np.array([cell.velocity_e["vx"], cell.velocity_e["vy"]])
        departure_e = east_mid - dt * current_e
        new_e = griddata(east_positions, east_velocities, [departure_e], method='linear', fill_value=0.0)[0]
        cell.velocity_e["vx"], cell.velocity_e["vy"] = new_e
        
        # South edge:
        south_mid = np.array([(minx + maxx) / 2, miny])
        current_s = np.array([cell.velocity_s["vx"], cell.velocity_s["vy"]])
        departure_s = south_mid - dt * current_s
        new_s = griddata(south_positions, south_velocities, [departure_s], method='linear', fill_value=0.0)[0]
        cell.velocity_s["vx"], cell.velocity_s["vy"] = new_s
        
        # North edge:
        north_mid = np.array([(minx + maxx) / 2, maxy])
        current_n = np.array([cell.velocity_n["vx"], cell.velocity_n["vy"]])
        departure_n = north_mid - dt * current_n
        new_n = griddata(north_positions, north_velocities, [departure_n], method='linear', fill_value=0.0)[0]
        cell.velocity_n["vx"], cell.velocity_n["vy"] = new_n

# ------------------------
# Incompressibility Projection (Projection Step)
# ------------------------
def incompressible(model, domain_bounds=(0, 0, 62, 65), num_iterations=10, o=0.8):
    """
    Iteratively adjusts the edge velocities for each cell so that the net divergence is forced toward zero.
    
    For each cell, the divergence is computed as:
        divergence = v_e + v_n - v_w - v_s
    where:
      - v_e is the east edge x–velocity,
      - v_n is the north edge y–velocity,
      - v_w is the west edge x–velocity,
      - v_s is the south edge y–velocity.
    
    Available edges are those that are not on the global boundary (given by domain_bounds).
    The correction for each available edge is computed as:
        correction = (divergence / n_avail) * o
    where o is the relaxation factor (0 < o <= 1).
    
    The function iterates this correction for num_iterations iterations and prints the average divergence
    after each iteration.
    """
    min_dom, min_doy, max_dom, max_doy = domain_bounds
    for it in range(num_iterations):
        for cell in model.polygons:
            v_e = cell.velocity_e["vx"]
            v_n = cell.velocity_n["vy"]
            v_w = cell.velocity_w["vx"]
            v_s = cell.velocity_s["vy"]
            divergence = v_e + v_n - v_w - v_s

            bx = cell.geometry.bounds  # [minx, miny, maxx, maxy]
            available_edges = []
            if bx[2] < max_dom - 1e-6:
                available_edges.append("east")
            if bx[3] < max_doy - 1e-6:
                available_edges.append("north")
            if bx[0] > min_dom + 1e-6:
                available_edges.append("west")
            if bx[1] > min_doy + 1e-6:
                available_edges.append("south")
            
            n_avail = len(available_edges)
            if n_avail == 0:
                continue

            correction = (divergence / n_avail) * o

            if "east" in available_edges:
                cell.velocity_e["vx"] -= correction
            if "north" in available_edges:
                cell.velocity_n["vy"] -= correction
            if "west" in available_edges:
                cell.velocity_w["vx"] += correction
            if "south" in available_edges:
                cell.velocity_s["vy"] += correction

        total_div = 0.0
        count = 0
        for cell in model.polygons:
            v_e = cell.velocity_e["vx"]
            v_n = cell.velocity_n["vy"]
            v_w = cell.velocity_w["vx"]
            v_s = cell.velocity_s["vy"]
            total_div += (v_e + v_n - v_w - v_s)
            count += 1
        avg_div = total_div / count if count > 0 else 0.0
        print(f"Iteration {it+1}: Average divergence = {avg_div:.3e}")

# ------------------------
# Compute Cell-Centered Velocity (for Heatmap)
# ------------------------
def compute_cell_center_velocity(model):
    centers = []
    magnitudes = []
    for cell in model.polygons:
        center = cell.geometry.centroid.coords[0]
        centers.append(center)
        avg_vx = (cell.velocity_w["vx"] + cell.velocity_e["vx"]) / 2.0
        avg_vy = (cell.velocity_s["vy"] + cell.velocity_n["vy"]) / 2.0
        mag = math.sqrt(avg_vx**2 + avg_vy**2)
        magnitudes.append(mag)
    return centers, magnitudes

# ------------------------
# Visualization Function: Plot Edge Velocities (Vector Field)
# ------------------------
def plot_edge_velocities(model, arrow_spacing=1, scale=1):
    x_n, y_n, vx_n, vy_n = [], [], [], []
    x_e, y_e, vx_e, vy_e = [], [], [], []
    x_s, y_s, vx_s, vy_s = [], [], [], []
    x_w, y_w, vx_w, vy_w = [], [], [], []
    
    for cell in model.polygons:
        minx, miny, maxx, maxy = cell.geometry.bounds
        x_n.append((minx + maxx) / 2)
        y_n.append(maxy)
        vx_n.append(cell.velocity_n["vx"] * scale)
        vy_n.append(cell.velocity_n["vy"] * scale)
        x_s.append((minx + maxx) / 2)
        y_s.append(miny)
        vx_s.append(cell.velocity_s["vx"] * scale)
        vy_s.append(cell.velocity_s["vy"] * scale)
        x_e.append(maxx)
        y_e.append((miny + maxy) / 2)
        vx_e.append(cell.velocity_e["vx"] * scale)
        vy_e.append(cell.velocity_e["vy"] * scale)
        x_w.append(minx)
        y_w.append((miny + maxy) / 2)
        vx_w.append(cell.velocity_w["vx"] * scale)
        vy_w.append(cell.velocity_w["vy"] * scale)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    polygons = [cell.geometry for cell in model.polygons]
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=model.space.crs)
    gdf.plot(edgecolor="black", facecolor="none", ax=ax)
    
    ax.quiver(x_n[::arrow_spacing], y_n[::arrow_spacing], vx_n[::arrow_spacing], vy_n[::arrow_spacing],
              angles="xy", scale_units="xy", scale=1, color="red", label="North")
    ax.quiver(x_s[::arrow_spacing], y_s[::arrow_spacing], vx_s[::arrow_spacing], vy_s[::arrow_spacing],
              angles="xy", scale_units="xy", scale=1, color="blue", label="South")
    ax.quiver(x_e[::arrow_spacing], y_e[::arrow_spacing], vx_e[::arrow_spacing], vy_e[::arrow_spacing],
              angles="xy", scale_units="xy", scale=1, color="green", label="East")
    ax.quiver(x_w[::arrow_spacing], y_w[::arrow_spacing], vx_w[::arrow_spacing], vy_w[::arrow_spacing],
              angles="xy", scale_units="xy", scale=1, color="purple", label="West")
    
    plt.title("Edge Velocities (Vector Field)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# ------------------------
# Visualization Function: Plot Cell-Centered Velocity Magnitude Heatmap
# ------------------------
def plot_heatmap(model):
    centers, magnitudes = compute_cell_center_velocity(model)
    xs = [pt[0] for pt in centers]
    ys = [pt[1] for pt in centers]
    
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(xs, ys, c=magnitudes, cmap='viridis', s=20)
    plt.colorbar(sc, label="Velocity Magnitude")
    plt.title("Cell-Centered Velocity Magnitude Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# ------------------------
# Agent Class: FlowPolygonAgent with Edge Velocities
# ------------------------
class FlowPolygonAgent(mg.GeoAgent):
    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(model, geometry, crs)
        self.unique_id = unique_id
        self.velocity_n = None
        self.velocity_e = None
        self.velocity_s = None
        self.velocity_w = None
        self.source = False  # Flag indicating if this cell is a source cell.
    def step(self):
        # No dynamic updates for now.
        pass

# ------------------------
# Model Class: FlowModel with Caching and Edge Sharing
# ------------------------
class FlowModel(mesa.Model):
    def __init__(self, geojson_input):
        super().__init__()
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        
        # Load and rescale GeoJSON data.
        if isinstance(geojson_input, str):
            agents_gdf = gpd.read_file(geojson_input)
        else:
            agents_gdf = gpd.GeoDataFrame.from_features(geojson_input["features"])
        
        if "id" in agents_gdf.columns:
            agents_gdf["unique_id"] = agents_gdf["id"] - 511
        else:
            raise ValueError("ERROR: No 'id' field found in GeoJSON properties.")
        
        # Rescale geometries to a 62 x 65 grid.
        minx, miny, maxx, maxy = agents_gdf.total_bounds
        scale_x = 62.0 / (maxx - minx)
        scale_y = 65.0 / (maxy - miny)
        agents_gdf["geometry"] = agents_gdf["geometry"].apply(
            lambda geom: scale_geom(translate(geom, xoff=-minx, yoff=-miny),
                                      xfact=scale_x, yfact=scale_y, origin=(0,0))
        )
        
        self.polygons = []
        for _, row in agents_gdf.iterrows():
            agent = FlowPolygonAgent(
                unique_id=row["unique_id"],
                model=self,
                geometry=row.geometry,
                crs=agents_gdf.crs
            )
            self.polygons.append(agent)
        
        self.space.add_agents(self.polygons)
        
        # Assign shared edge velocities.
        self.edge_dict = assign_edge_velocities(self.polygons)
        
        # Load or compute initial source velocities.
        init_velocities_filename = "init_velocities.pkl"
        if os.path.exists(init_velocities_filename):
            with open(init_velocities_filename, "rb") as f:
                init_velocities = pickle.load(f)
            for agent in self.polygons:
                if agent.unique_id in init_velocities:
                    vx, vy = init_velocities[agent.unique_id]
                    agent.velocity_w["vx"] = vx
                    agent.velocity_w["vy"] = vy
                    agent.velocity_s["vx"] = vx
                    agent.velocity_s["vy"] = vy
                    agent.source = True
            print("Loaded initial source velocities from cache.")
        else:
            polygon_ids_to_initialize = {
                4260, 4261, 4262, 4263, 4264, 4265,
                4337, 4338, 4339, 4340, 4341,
                4414, 4415, 4416, 4417, 4418,
                4492, 4493, 4494, 4495, 4496,
                4570, 4571, 4572, 4573,
                4648, 4649, 4650,
                4725, 4726, 4727,
                4802, 4803, 4804
            }
            polygon_ids_to_initialize = {pid - 511 for pid in polygon_ids_to_initialize}
            initialize_currents(self, polygon_ids_to_initialize, magnitude=0.5, bearing_degrees=140)
            init_velocities = {agent.unique_id: (agent.velocity_w["vx"], agent.velocity_w["vy"])
                               for agent in self.polygons if agent.source}
            with open(init_velocities_filename, "wb") as f:
                pickle.dump(init_velocities, f)
            print("Computed and cached initial source velocities.")
        
        print("\nModel initialized with the following polygon IDs (first 20 shown):")
        for agent in self.polygons[:20]:
            print(f"  - Agent ID: {agent.unique_id}")

    def step(self):
        # No dynamic update yet.
        pass

# ------------------------
# Run the Model and Visualize with Advection and Projection
# ------------------------

model = FlowModel(geojson_path)

# Run several time steps: advect then enforce incompressibility.
num_steps = 20
dt = 0.1  # time step for advection
for i in tqdm(range(num_steps)):
    advect_edge_velocities(model, dt)
    incompressible(model, domain_bounds=(0, 0, 62, 65), num_iterations=5, o=0.8)

# Visualize the results.
print("Edge velocities (Vector Field):")
plot_edge_velocities(model, arrow_spacing=1, scale=1)

print("Cell-Centered Velocity Magnitude (Heatmap):")
plot_heatmap(model)