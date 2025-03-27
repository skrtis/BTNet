import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np

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

def create_malpeque_grid(grid_data, grid_width=62, grid_height=65):
    """Create a grid with squares at the specified tile indices"""
    grid_agents = []
    agent_id = 0
    
    # Create squares at the specified locations
    for col, row, clam_presence in grid_data:
        # Check if coordinates are within grid bounds
        if 0 <= col < grid_width and 0 <= row < grid_height:
            # Create a 1x1 square with bottom-left corner at (col, row)
            square = Polygon([
                (col, row),        # Bottom-left corner
                (col+1, row),      # Bottom-right corner
                (col+1, row+1),    # Top-right corner
                (col, row+1)       # Top-left corner
            ])
            
            agent = FlowPolygonAgent(agent_id, square, row, col)
            agent.clam_presence = clam_presence  # Set clam presence from GeoJSON
            agent.water = True  # All cells are water
            
            grid_agents.append(agent)
            agent_id += 1
    
    print(f"Created {len(grid_agents)} FlowPolygonAgent instances in a {grid_width}x{grid_height} grid")
    return grid_agents

def visualize_malpeque_grid(grid_agents, grid_width=62, grid_height=65):
    """Visualize the Malpeque Bay grid"""
    plt.figure(figsize=(15, 15))
    
    # Plot each agent
    for agent in grid_agents:
        x, y = agent.geometry.exterior.xy
        plt.plot(x, y, 'k-', linewidth=0.5)
        
        # Color based on cell properties
        if agent.clam_presence:
            color = 'green'
            alpha = 0.5
        else:
            color = 'skyblue'
            alpha = 0.3
            
        plt.fill(x, y, alpha=alpha, color=color)
    
    # Create a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', alpha=0.3, edgecolor='k', label='Water'),
        Patch(facecolor='green', alpha=0.5, edgecolor='k', label='Clam Presence')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set plot bounds and labels
    plt.xlim(-1, grid_width+1)
    plt.ylim(-1, grid_height+1)
    plt.title("Malpeque Bay Grid")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.savefig('malpeque_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate and visualize the grid
if __name__ == "__main__":
    # Extract and transform indices
    grid_data = extract_grid_indices("malpeque_tiles.geojson")
    transformed_data = transform_indices(grid_data)
    
    # Create grid agents
    grid_agents = create_malpeque_grid(transformed_data)
    
    # Visualize grid
    visualize_malpeque_grid(grid_agents)