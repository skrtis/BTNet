import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import random
class FlowPolygonAgent:
    def __init__(self, unique_id, geometry, row, col):
        self.unique_id = unique_id
        self.geometry = geometry
        self.row = row
        self.col = col
        
        # Add properties matching FlowPolygonAgent in mechanisms.py
        self.source = False
        self.clam_presence = False
        self.water = True
        self.concentration = 0.0
        self.btn_concentration = 0.0
        self.healthy_clams = 0
        self.infected_clams = 0
        self.dead_clams = 0
        
        # Velocities on edges (similar to mechanisms.py)
        self.velocity_n = None  # North edge
        self.velocity_e = None  # East edge
        self.velocity_s = None  # South edge
        self.velocity_w = None  # West edge

def extract_grid_indices(geojson_file_path):
    """Extract (col_index, row_index, clam_presence) from a GeoJSON file"""
    with open(geojson_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract indices and clam presence
    grid_data = []
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        for feature in features:
            props = feature.get("properties", {})
            if "col_index" in props and "row_index" in props:
                col = props["col_index"]
                row = props["row_index"]
                # Extract clam presence (default to False if not present)
                clam_presence = props.get("clam_presence", False)
                grid_data.append((col, row, clam_presence))
    
    print(f"Extracted {len(grid_data)} grid cells from GeoJSON")
    assert len(grid_data) == 1722, f"Expected 1722 entries, but found {len(grid_data)}"
    
    return grid_data

def transform_indices(grid_data):
    """Apply the transformation to the grid data"""
    transformed = []
    for col, row, clam_presence in grid_data:
        col_new = col - 6
        row_new = abs(row - 69)
        transformed.append((col_new, row_new, clam_presence))
    return transformed

def create_grid_agents(transformed_data, complete_grid=True, grid_width=62, grid_height=65):
    """
    Create a grid of agents with options for complete or partial grid
    
    Parameters:
    -----------
    transformed_data : list
        List of (col, row, clam_presence) tuples from transformed GeoJSON data
    complete_grid : bool
        If True, create a complete grid including non-water cells
        If False, create only the water cells defined in transformed_data
    grid_width : int
        Width of the grid
    grid_height : int
        Height of the grid
        
    Returns:
    --------
    list of FlowPolygonAgent
    """
    # Extract existing coordinates
    existing_coords = set((col, row) for col, row, _ in transformed_data)
    
    # Create a list for all agents
    all_agents = []
    agent_id = 0

    
    # First, add all the existing agents from the GeoJSON data (water cells)
    for col, row, clam_presence in transformed_data:
        if 0 <= col < grid_width and 0 <= row < grid_height:
            square = Polygon([
                (col, row),        # Bottom-left corner
                (col+1, row),      # Bottom-right corner
                (col+1, row+1),    # Top-right corner
                (col, row+1)       # Top-left corner
            ])
            
            agent = FlowPolygonAgent(agent_id, square, row, col)
            agent.clam_presence = clam_presence  # Set clam presence from GeoJSON
            if agent.clam_presence == True:
                agent.btn_concentration = 0.0
                agent.healthy_clams = 2500
                agent.infected_clams = 0
                agent.dead_clams = 0

            agent.water = True  # All these cells are water
            
            all_agents.append(agent)
            agent_id += 1
    
    # If complete_grid is True, add missing cells as non-water cells
    if complete_grid:
        for col in range(grid_width):
            for row in range(grid_height):
                if (col, row) not in existing_coords:
                    square = Polygon([
                        (col, row),
                        (col+1, row),
                        (col+1, row+1),
                        (col, row+1)
                    ])
                    
                    agent = FlowPolygonAgent(agent_id, square, row, col)
                    agent.water = False  # These cells are not water
                    agent.clam_presence = False  # No clams in non-water cells
                    
                    all_agents.append(agent)
                    agent_id += 1
    
    # Print summary
    if complete_grid:
        print(f"Created complete grid with {len(all_agents)} cells:")
        print(f" - {sum(1 for a in all_agents if a.water)} water cells")
        print(f" - {sum(1 for a in all_agents if not a.water)} non-water cells")
        print(f" - {sum(1 for a in all_agents if a.clam_presence)} cells with clam presence")
    else:
        print(f"Created Malpeque Bay grid with {len(all_agents)} water cells")
        print(f" - {sum(1 for a in all_agents if a.clam_presence)} cells with clam presence")
    
    return all_agents

def assign_edge_velocities(agents):
    """Assign velocity references to each agent's edges"""
    import math
    
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    print(f"Grid dimensions: {n_rows} rows, {n_cols} columns")
    
    # Create dictionaries to store velocities
    h_velocities = {}  # Horizontal edges (stores vertical velocity)
    v_velocities = {}  # Vertical edges (stores horizontal velocity)
    
    # Initialize all velocities to zero
    for row in range(n_rows+1):  # +1 for bottom edge of bottom row
        for col in range(n_cols):
            h_velocities[(row, col)] = {"vy": 0.0, "locked": False}
            
    for row in range(n_rows):
        for col in range(n_cols+1):  # +1 for right edge of rightmost column
            v_velocities[(row, col)] = {"vx": 0.0, "locked": False}
   
    # Lock grid boundary edges
    boundary_edges_count = 0
    
    # Lock top and bottom horizontal edges
    for col in range(n_cols):
        # Top boundary
        h_velocities[(0, col)]["locked"] = True
        boundary_edges_count += 1
        
        # Bottom boundary
        h_velocities[(n_rows, col)]["locked"] = True
        boundary_edges_count += 1
    
    # Lock left and right vertical edges
    for row in range(n_rows):
        # Left boundary
        v_velocities[(row, 0)]["locked"] = True
        boundary_edges_count += 1
        
        # Right boundary
        v_velocities[(row, n_cols)]["locked"] = True
        boundary_edges_count += 1
            
    # Assign velocity references to agents
    for agent in agents:
        # North edge
        agent.velocity_n = h_velocities[(agent.row, agent.col)]
        
        # South edge
        agent.velocity_s = h_velocities[(agent.row+1, agent.col)]
        
        # East edge
        agent.velocity_e = v_velocities[(agent.row, agent.col+1)]
        
        # West edge
        agent.velocity_w = v_velocities[(agent.row, agent.col)]
    
    # Define source cells by coordinates (col, row)
    source_coords = [(51, 45),(52, 44), (53, 43)]
    vert_source_coords =[(45,48),(46,47),(47,46)] 
    
    # Set and lock velocities for land cells
    land_cells_count = 0
    water_cells_count = 0
    source_cells_count = 0
    
    for agent in agents:
        if (agent.col, agent.row) in source_coords or (agent.col, agent.row) in vert_source_coords:
            # Handle source cells
            agent.source = True
            source_cells_count += 1
            continue  # Skip to next agent, we'll handle source cells separately
        
        if hasattr(agent, 'water'):
            if not agent.water:
                # Land cells: Set edges to zero and lock them
                land_cells_count += 1
                
                if agent.velocity_n:
                    agent.velocity_n["vy"] = 0.0
                    agent.velocity_n["locked"] = True
                    
                if agent.velocity_s:
                    agent.velocity_s["vy"] = 0.0
                    agent.velocity_s["locked"] = True
                    
                if agent.velocity_e:
                    agent.velocity_e["vx"] = 0.0
                    agent.velocity_e["locked"] = True
                    
                if agent.velocity_w:
                    agent.velocity_w["vx"] = 0.0
                    agent.velocity_w["locked"] = True
            else:
                # Regular water cells: Initialize to zero, keep unlocked
                water_cells_count += 1
                
                # Edges already initialized to zero and unlocked above,
                # no additional action needed here
    
    # Set source velocities with a bearing of 140 degrees and magnitude of 1.0
    bearing_degrees = 130
    magnitude = 0.0032
    
    # Calculate injection vector components
    br = math.radians(360 - bearing_degrees)
    inj_vx = magnitude * math.cos(br)
    inj_vy = magnitude * math.sin(br)
    print(f"Source injection vector: ({inj_vx:.2f}, {inj_vy:.2f}) with bearing {bearing_degrees}°")

    bearing_degrees_two = 110 
    magnitude_two =0.0032
    br2 = math.radians(360 - bearing_degrees_two)
    inj_vx2 = magnitude_two * math.cos(br2)
    inj_vy2 = magnitude_two * math.sin(br2)
    print(f"Source injection vector: ({inj_vx2:.2f}, {inj_vy2:.2f}) with bearing {bearing_degrees_two}°")
    
    # Set source velocities
    for agent in agents:
        if (agent.col, agent.row) in source_coords:
            # Set velocities on all edges and lock them
            if agent.velocity_n:
                agent.velocity_n["vy"] = inj_vy
                agent.velocity_n["locked"] = True
                
            if agent.velocity_s:
                agent.velocity_s["vy"] = inj_vy
                agent.velocity_s["locked"] = True
                
            if agent.velocity_e:
                agent.velocity_e["vx"] = inj_vx
                agent.velocity_e["locked"] = True
                
            if agent.velocity_w:
                agent.velocity_w["vx"] = inj_vx
                agent.velocity_w["locked"] = True
        if (agent.col, agent.row) in vert_source_coords:
            # Set velocities on all edges and lock them
            if agent.velocity_n:
                agent.velocity_n["vy"] = inj_vy2
                agent.velocity_n["locked"] = True
                
            if agent.velocity_s:
                agent.velocity_s["vy"] = inj_vy2
                agent.velocity_s["locked"] = True
                
            if agent.velocity_e:
                agent.velocity_e["vx"] = inj_vx2
                agent.velocity_e["locked"] = True
                
            if agent.velocity_w:
                agent.velocity_w["vx"] = inj_vx2
                agent.velocity_w["locked"] = True
    
    # Print summary
    print(f"Velocity assignment summary:")
    print(f" - {water_cells_count} water cells (initialized to zero, unlocked)")
    print(f" - {land_cells_count} land cells (set to zero, locked)")
    print(f" - {source_cells_count} source cells (set to ({inj_vx:.2f}, {inj_vy:.2f}), locked)")
    print(f" - {boundary_edges_count} grid boundary edges (set to zero, locked)")
    
    return h_velocities, v_velocities

