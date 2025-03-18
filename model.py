import json
import math
import mesa
import mesa_geo as mg
from shapely.geometry import shape
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
import os

# ------------------------
# Load GeoJSON Data
# ------------------------

geojson_path = "malpeque_tiles.geojson"  # Local file
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)
print(f"Total Polygons in GeoJSON: {len(geojson_data['features'])}")

# ------------------------
# Utility Function: Compute Shared Edges using Shapely Intersection
# ------------------------

def compute_shared_edges(agents):
    """
    For each agent (tile), identify direct neighbors (up, down, left, right) by checking
    if the intersection between two geometries yields a line (or multi-line). This indicates a
    shared edge. Tiles that represent islands or edges will simply have fewer neighbors.
    """
    for polygon in agents:
        polygon.neighbors = []  # Reset neighbor list
        for neighbor in agents:
            if polygon is not neighbor:
                inter = polygon.geometry.intersection(neighbor.geometry)
                # Consider only intersections that yield a line (shared edge), not a point.
                if inter and inter.geom_type in ['LineString', 'MultiLineString']:
                    polygon.neighbors.append(neighbor)

# ------------------------
# Function: Initialize Flow in Specific Tiles (Source Cells)
# ------------------------

def initialize_currents(model, polygon_ids, magnitude, bearing_degrees):
    """
    Sets initial flow (velocity) vectors in the selected cells (tiles) by converting a given
    magnitude and bearing into Cartesian components. These cells are then marked as sources.
    The velocities are stored in each agent's "fields" dictionary.
    """
    # Convert bearing to Cartesian components
    bearing_radians = math.radians(360 - bearing_degrees)
    init_vx = magnitude * math.cos(bearing_radians)
    init_vy = magnitude * math.sin(bearing_radians)

    print(f"\nInitializing currents for {len(polygon_ids)} polygons as source cells...\n")
    found_tiles = 0

    for agent in model.polygons:
        if agent.unique_id in polygon_ids:
            agent.fields["velocity_x"] = init_vx
            agent.fields["velocity_y"] = init_vy
            agent.source = True  # Mark this cell as a source.
            found_tiles += 1
            print(f"Agent {agent.unique_id} → Source flow vector: ({init_vx:.2f}, {init_vy:.2f})")

    if found_tiles == 0:
        print("ERROR: No matching polygons found. Check your polygon IDs!")
    else:
        print(f"\nSuccessfully initialized {found_tiles} source tiles with flow vectors!\n")

# ------------------------
# Function: Propagate Eulerian Flow (Simplified Update)
# ------------------------

def propagate_eulerian_flow(model, beta=0.95, alpha=0.3, nu=0.1, delta_x=1.0, delta_t=0.1):
    """
    A simplified Eulerian update for the flow field with explicit bounce (reflection)
    conditions at any cell edge that does not have a neighbor.
    
    For each non-source cell:
      - Compute advective and diffusive terms as before.
      - Then, for each edge (left, right, bottom, top), if no neighbor exists there
        and the cell's updated velocity is directed outwards, reflect that component
        (reducing it to 50% of its value).
    """
    new_fields = {}  # Dictionary to store updated field values for each agent
    tol = 1e-6  # Tolerance for comparing floating-point bounds

    for agent in model.polygons:
        # Skip source cells so they maintain a constant flow.
        if agent.source:
            continue
        if not agent.neighbors:
            continue  # Skip isolated cells

        # ------------------------
        # Compute Advective Term (Simplified)
        # ------------------------
        sum_vx_adv, sum_vy_adv = 0.0, 0.0
        count_adv = 0
        for neighbor in agent.neighbors:
            # Consider only neighbors with nonzero flow.
            if neighbor.fields["velocity_x"] != 0.0 or neighbor.fields["velocity_y"] != 0.0:
                sum_vx_adv += neighbor.fields["velocity_x"]
                sum_vy_adv += neighbor.fields["velocity_y"]
                count_adv += 1
        advective_vx = beta * (sum_vx_adv / count_adv) if count_adv > 0 else 0.0
        advective_vy = beta * (sum_vy_adv / count_adv) if count_adv > 0 else 0.0

        # ------------------------
        # Compute Diffusive Term (Finite-Difference Laplacian)
        # ------------------------
        sum_vx_diff, sum_vy_diff = 0.0, 0.0
        count_diff = len(agent.neighbors)
        for neighbor in agent.neighbors:
            sum_vx_diff += neighbor.fields["velocity_x"]
            sum_vy_diff += neighbor.fields["velocity_y"]
        if count_diff > 0:
            avg_vx_diff = sum_vx_diff / count_diff
            avg_vy_diff = sum_vy_diff / count_diff
        else:
            avg_vx_diff = agent.fields["velocity_x"]
            avg_vy_diff = agent.fields["velocity_y"]
        diffusive_vx = nu * ((avg_vx_diff - agent.fields["velocity_x"]) / (delta_x**2))
        diffusive_vy = nu * ((avg_vy_diff - agent.fields["velocity_y"]) / (delta_x**2))

        # ------------------------
        # Combine Terms to Update Velocity (Euler Step)
        # ------------------------
        new_vx = agent.fields["velocity_x"] + delta_t * (-advective_vx + diffusive_vx)
        new_vy = agent.fields["velocity_y"] + delta_t * (-advective_vy + diffusive_vy)

        # ------------------------
        # Bounce (Reflection) Conditions Based on Missing Neighbors
        # ------------------------
        # Get the cell's bounding box.
        cell_minx, cell_miny, cell_maxx, cell_maxy = agent.geometry.bounds

        # Determine for each edge if a neighbor exists.
        has_left  = any(abs(neighbor.geometry.bounds[2] - cell_minx) < tol for neighbor in agent.neighbors)
        has_right = any(abs(neighbor.geometry.bounds[0] - cell_maxx) < tol for neighbor in agent.neighbors)
        has_bottom = any(abs(neighbor.geometry.bounds[3] - cell_miny) < tol for neighbor in agent.neighbors)
        has_top   = any(abs(neighbor.geometry.bounds[1] - cell_maxy) < tol for neighbor in agent.neighbors)

        # Bounce left edge if missing neighbor and velocity is outward.
        if (not has_left) and new_vx < 0:
            new_vx = -0.5 * new_vx

        # Bounce right edge if missing neighbor and velocity is outward.
        if (not has_right) and new_vx > 0:
            new_vx = -0.5 * new_vx

        # Bounce bottom edge if missing neighbor and velocity is outward.
        if (not has_bottom) and new_vy < 0:
            new_vy = -0.5 * new_vy

        # Bounce top edge if missing neighbor and velocity is outward.
        if (not has_top) and new_vy > 0:
            new_vy = -0.5 * new_vy

        new_fields[agent] = {"velocity_x": new_vx, "velocity_y": new_vy}

    # ------------------------
    # Update all agents simultaneously with the new velocities.
    # ------------------------
    for agent, new_vals in new_fields.items():
        agent.fields["velocity_x"] = new_vals["velocity_x"]
        agent.fields["velocity_y"] = new_vals["velocity_y"]

# ------------------------
# Agent Class: FlowPolygonAgent (Each Tile)
# ------------------------

class FlowPolygonAgent(mg.GeoAgent):
    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(model, geometry, crs)
        self.unique_id = unique_id
        self.neighbors = []  # List to store adjacent tiles
        
        # Eulerian flow fields stored in a dictionary.
        self.fields = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
        }
        self.source = False  # Indicator whether this cell is a constant flow source.

    def step(self):
        # This agent does not update its own flow; the global Eulerian update is applied at the model level.
        pass

# ------------------------
# Model Class: FlowModel with Caching
# ------------------------

class FlowModel(mesa.Model):
    def __init__(self, geojson_path):
        super().__init__()
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        
        # Load GeoJSON data as a GeoDataFrame.
        agents_gdf = gpd.read_file(geojson_path)
        
        # Adjust the IDs by subtracting 511 to match the expected range.
        if "id" in agents_gdf.columns:
            agents_gdf["unique_id"] = agents_gdf["id"] - 511
        else:
            raise ValueError("ERROR: No 'id' field found in GeoJSON properties.")
        
        # Create a list of agents.
        self.polygons = []
        for _, row in agents_gdf.iterrows():
            agent = FlowPolygonAgent(
                unique_id=row["unique_id"],
                model=self,
                geometry=row.geometry,
                crs=agents_gdf.crs
            )
            self.polygons.append(agent)
        
        # Add agents to the GeoSpace.
        self.space.add_agents(self.polygons)
        
        # --------------
        # Load or Compute Neighbors Cache
        # --------------
        neighbors_filename = "neighbors.pkl"
        if os.path.exists(neighbors_filename):
            with open(neighbors_filename, "rb") as f:
                neighbors_dict = pickle.load(f)
            id_to_agent = {agent.unique_id: agent for agent in self.polygons}
            for agent in self.polygons:
                if agent.unique_id in neighbors_dict:
                    agent.neighbors = [id_to_agent[n_id] for n_id in neighbors_dict[agent.unique_id]]
            print("Loaded neighbor data from cache.")
        else:
            compute_shared_edges(self.polygons)
            neighbors_dict = {agent.unique_id: [n.unique_id for n in agent.neighbors] for agent in self.polygons}
            with open(neighbors_filename, "wb") as f:
                pickle.dump(neighbors_dict, f)
            print("Computed and cached neighbor data.")
        
        # --------------
        # Load or Compute Initial Source Velocities Cache
        # --------------
        init_velocities_filename = "init_velocities.pkl"
        if os.path.exists(init_velocities_filename):
            with open(init_velocities_filename, "rb") as f:
                init_velocities = pickle.load(f)
            for agent in self.polygons:
                if agent.unique_id in init_velocities:
                    vx, vy = init_velocities[agent.unique_id]
                    agent.fields["velocity_x"] = vx
                    agent.fields["velocity_y"] = vy
                    agent.source = True
            print("Loaded initial source velocities from cache.")
        else:
            # Define the polygon IDs for current initialization.
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
            initialize_currents(self, polygon_ids_to_initialize, magnitude=1.6, bearing_degrees=140)
            init_velocities = {agent.unique_id: (agent.fields["velocity_x"], agent.fields["velocity_y"]) 
                               for agent in self.polygons if agent.source}
            with open(init_velocities_filename, "wb") as f:
                pickle.dump(init_velocities, f)
            print("Computed and cached initial source velocities.")
        
        print("\nModel initialized with the following polygon IDs (first 20 shown):")
        for agent in self.polygons[:20]:
            print(f"  - Agent ID: {agent.unique_id}")

    def step(self):
        # Apply the Eulerian update to the entire grid.
        propagate_eulerian_flow(self)

# ------------------------
# Visualization Function: Plot Flow Vectors
# ------------------------

def plot_flow_vectors(model, scale_factor=1, arrow_spacing=1,scaler=1):
    """
    Visualizes the flow field by plotting arrows (using a quiver plot) over the tile polygons.
    The arrow lengths are scaled for visualization.
    """
    # Extract polygon geometries and Eulerian field values.
    polygons = [agent.geometry for agent in model.polygons]
    flow_x = [agent.fields["velocity_x"] for agent in model.polygons]
    flow_y = [agent.fields["velocity_y"] for agent in model.polygons]

    # Create a GeoDataFrame for plotting.
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=model.space.crs)
    centroids = gdf.geometry.centroid
    x_coords, y_coords = centroids.x, centroids.y

    # Subsample arrows to reduce clutter.
    x_coords = x_coords[::arrow_spacing]
    y_coords = y_coords[::arrow_spacing]
    flow_x = [vx * scale_factor for vx in flow_x[::arrow_spacing]]
    flow_y = [vy * scale_factor for vy in flow_y[::arrow_spacing]]

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(edgecolor="black", facecolor="none", ax=ax)
    
    ax.quiver(
        x_coords, y_coords, flow_x, flow_y,
        angles="xy", scale_units="inches", scale=scaler,
        color="blue", width=0.005, headwidth=5
    )
    plt.title("Eulerian Flow Field in Malpeque Bay (Scaled)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# ------------------------
# Run the Model and Visualize
# ------------------------

geojson_test_path = "malpeque_tiles.geojson"
model = FlowModel(geojson_test_path)

# Advance the model several steps to propagate the flow.
for _ in tqdm(range(500)):
    model.step()

plot_flow_vectors(model,scaler=5e12)
