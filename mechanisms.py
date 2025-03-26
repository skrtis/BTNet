import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class FlowPolygonAgent:
    def __init__(self, unique_id, geometry, row, col):
        self.unique_id = unique_id
        self.geometry = geometry
        self.row = row
        self.col = col
        self.source = False
        
        # Velocities on edges
        self.velocity_n = None  # North edge
        self.velocity_e = None  # East edge
        self.velocity_s = None  # South edge
        self.velocity_w = None  # West edge

def create_grid(n_rows=10, n_cols=10):
    """Create a grid of n_rows x n_cols square cells"""
    agents = []
    
    # Create grid cells as square polygons
    for row in range(n_rows):
        for col in range(n_cols):
            # Create a 1x1 square polygon
            square = Polygon([
                (col, row),
                (col+1, row),
                (col+1, row+1),
                (col, row+1)
            ])
            
            agent = FlowPolygonAgent(
                unique_id=row*n_cols + col,
                geometry=square,
                row=row,
                col=col
            )
            agents.append(agent)
    
    print(f"Created {len(agents)} agent objects in a {n_rows}x{n_cols} grid")
    return agents

def assign_edge_velocities(agents):
    """Assign velocity references to each agent's edges"""
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
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
    
    return h_velocities, v_velocities

def visualize_grid(agents, h_velocities, v_velocities, title="Grid Visualization", threshold=1e-6, scale=0.5):
    plt.figure(figsize=(10, 10))
    
    # Plot cell boundaries
    for agent in agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color='black', linewidth=0.5)
        
        # Plot source cells with red fill
        if agent.source:
            plt.fill(xs, ys, color='red', alpha=0.2)
    
    # Plot horizontal velocities (at horizontal edges)
    for (row, col), vel in h_velocities.items():
        vy = vel["vy"]
        if abs(vy) > threshold:
            x = col + 0.5  # Middle of horizontal edge
            y = row      # At the edge
            plt.arrow(x, y, 0, vy*scale, head_width=0.1, head_length=0.1*abs(vy), 
                     fc='green', ec='green', width=0.02)
    
    # Plot vertical velocities (at vertical edges)
    for (row, col), vel in v_velocities.items():
        vx = vel["vx"]
        if abs(vx) > threshold:
            x = col      # At the edge
            y = row + 0.5  # Middle of vertical edge
            plt.arrow(x, y, vx*scale, 0, head_width=0.1, head_length=0.1*abs(vx), 
                     fc='blue', ec='blue', width=0.02)
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def initialize_source_velocities(agents, v_velocities, magnitude=2.0):
    """Set source velocities at the right edge of the grid"""
    # Find rightmost column
    max_col = max(agent.col for agent in agents)
    
    # Set velocities on right edge of rightmost column
    for agent in agents:
        if agent.col == max_col:
            agent.source = True
            right_edge_key = (agent.row, agent.col+1)
            v_velocities[right_edge_key]["vx"] = -magnitude  # Negative = leftward flow
            v_velocities[right_edge_key]["locked"] = True
    
    print(f"Set source velocities on right edge with magnitude {magnitude}")

# Create the 10x10 grid
agents = create_grid(10, 10)
h_velocities, v_velocities = assign_edge_velocities(agents)
initialize_source_velocities(agents, v_velocities, 2.0)

# After initializing the grid:

def create_global_edges_dict(h_velocities, v_velocities):
    """Combine horizontal and vertical velocities into a single dictionary"""
    global_edges = {}
    
    # Add horizontal edges (with vertical velocities)
    for (row, col), vel in h_velocities.items():
        key = ("H", row, col)  # H = horizontal edge
        global_edges[key] = {
            "vy": vel["vy"],
            "vx": 0.0,  # Horizontal edges don't have horizontal velocity
            "locked": vel["locked"],
            "midpoint": (col + 0.5, row)  # Middle of horizontal edge
        }
    
    # Add vertical edges (with horizontal velocities)
    for (row, col), vel in v_velocities.items():
        key = ("V", row, col)  # V = vertical edge
        global_edges[key] = {
            "vx": vel["vx"],
            "vy": 0.0,  # Vertical edges don't have vertical velocity
            "locked": vel["locked"],
            "midpoint": (col, row + 0.5)  # Middle of vertical edge
        }
    
    return global_edges

# Create the global edges dictionary
global_edges = create_global_edges_dict(h_velocities, v_velocities)

visualize_grid(agents, h_velocities, v_velocities, "Initial 10x10 Grid with Source Velocities")

def set_boundary_conditions(h_velocities, v_velocities):
    """Set boundary conditions - lock the domain boundary edges with zero velocity"""
    # Find grid dimensions
    max_row = max(row for row, _ in h_velocities.keys())
    max_col = max(col for _, col in v_velocities.keys())
    
    # Left boundary (x = 0)
    for row in range(max_row):
        left_edge = (row, 0)
        if left_edge in v_velocities:
            # Don't overwrite velocity if it's already non-zero (e.g., for source edges)
            if abs(v_velocities[left_edge]["vx"]) < 1e-10:
                v_velocities[left_edge]["vx"] = 0.0
            v_velocities[left_edge]["locked"] = True
    
    # Right boundary (x = max_col)
    # Note: already set as sources in initialize_source_velocities
    
    # Top boundary (y = 0)
    for col in range(max_col):
        top_edge = (0, col)
        if top_edge in h_velocities:
            h_velocities[top_edge]["vy"] = 0.0
            h_velocities[top_edge]["locked"] = True
    
    # Bottom boundary (y = max_row)
    for col in range(max_col):
        bottom_edge = (max_row, col)
        if bottom_edge in h_velocities:
            h_velocities[bottom_edge]["vy"] = 0.0
            h_velocities[bottom_edge]["locked"] = True
    
    print("Set boundary conditions: all domain boundary edges are now locked")

# Apply boundary conditions after initializing source velocities
set_boundary_conditions(h_velocities, v_velocities)

# Recreate global edges dictionary with the updated edge data
global_edges = create_global_edges_dict(h_velocities, v_velocities)

# Print updated information about locked edges
locked_count = sum(1 for edge in global_edges.values() if edge["locked"])
print(f"\nTotal locked edges: {locked_count}")

print("\nLocked edges:")
for key, edge in global_edges.items():
    if edge["locked"]:
        edge_type = "Horizontal" if key[0] == "H" else "Vertical"
        row, col = key[1], key[2]
        print(f"  {edge_type} edge at ({row},{col}): vx={edge['vx']:.2f}, vy={edge['vy']:.2f}")

# Visualize grid with boundary conditions
visualize_grid(agents, h_velocities, v_velocities, "10x10 Grid with Source and Boundary Conditions")

def advect_velocities_step1(h_velocities, v_velocities):
    """
    Step 1 of advection: Create 2D arrays for horizontal and vertical velocities
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1  # Number of cell rows (10)
    n_cols = max(agent.col for agent in agents) + 1  # Number of cell columns (10)
    
    # HORIZONTAL VELOCITIES (vx at vertical edges)
    # For a 10×10 cell grid, shape should be 10×11
    h_vel_grid = np.zeros((n_rows, n_cols+1))
    
    # Fill the grid with values from v_velocities
    for row in range(n_rows):
        for col in range(n_cols+1):
            edge_key = (row, col)
            if edge_key in v_velocities:
                h_vel_grid[row, col] = v_velocities[edge_key]["vx"]
    
    # Print info about horizontal velocity grid
    print(f"\nHorizontal velocity grid shape: {h_vel_grid.shape}")
    print(f"Number of non-zero horizontal velocities: {np.count_nonzero(h_vel_grid)}")
    print("\nHorizontal velocity grid (vx at vertical edges):")
    print(h_vel_grid)
    
    # VERTICAL VELOCITIES (vy at horizontal edges)
    # For a 10×10 cell grid, shape should be 11×10
    v_vel_grid = np.zeros((n_rows+1, n_cols))
    
    # Fill the grid with values from h_velocities
    for row in range(n_rows+1):
        for col in range(n_cols):
            edge_key = (row, col)
            if edge_key in h_velocities:
                v_vel_grid[row, col] = h_velocities[edge_key]["vy"]
    
    # Print info about vertical velocity grid
    print(f"\nVertical velocity grid shape: {v_vel_grid.shape}")
    print(f"Number of non-zero vertical velocities: {np.count_nonzero(v_vel_grid)}")
    print("\nVertical velocity grid (vy at horizontal edges):")
    print(v_vel_grid)
    
    # Identify source and boundary locations
    source_locs = []
    boundary_locs = []
    for (row, col), vel in v_velocities.items():
        if abs(vel["vx"]) > 1e-10 and vel["locked"]:
            source_locs.append((row, col))
        elif abs(vel["vx"]) < 1e-10 and vel["locked"]:
            boundary_locs.append((row, col))
    
    print("\nSource locations (row, col):")
    for loc in source_locs:
        print(f"  {loc}")
        
    print("\nBoundary locations with locked zero velocity (first 5):")
    for loc in boundary_locs[:5]:
        print(f"  {loc}")
    if len(boundary_locs) > 5:
        print(f"  ... and {len(boundary_locs) - 5} more")
    
    # Create a dictionary mapping grid positions to whether they're locked
    locked_positions = {}
    for (row, col), vel in v_velocities.items():
        locked_positions[(row, col)] = vel["locked"]
    
    h_locked_positions = {}
    for (row, col), vel in h_velocities.items():
        h_locked_positions[(row, col)] = vel["locked"]
    
    # Return both velocity grids and locked positions
    return h_vel_grid, v_vel_grid, locked_positions, h_locked_positions

# Test the first step of advection
h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions = advect_velocities_step1(h_velocities, v_velocities)

def advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions):
    """
    Step 2 of advection: Calculate secondary velocity components for each point
    by averaging neighboring points
    
    For horizontal edges (which primarily store vy): Calculate secondary vx
    For vertical edges (which primarily store vx): Calculate secondary vy
    """
    n_rows, n_cols_h = h_vel_grid.shape  # 10x11
    n_rows_v, n_cols = v_vel_grid.shape  # 11x10
    
    # Create arrays to store the secondary velocity components
    h_secondary_vx = np.zeros((n_rows_v, n_cols))  # Secondary horizontal velocity for horizontal edges
    v_secondary_vy = np.zeros((n_rows, n_cols_h))  # Secondary vertical velocity for vertical edges
    
    # For each HORIZONTAL edge, calculate secondary vx by averaging from nearby VERTICAL edges
    for row in range(n_rows_v):  # 0 to 10
        for col in range(n_cols):  # 0 to 9
            # For a horizontal edge at (row,col), get vx from vertical edges at:
            # (row-1,col), (row,col), (row-1,col+1), (row,col+1)
            count = 0
            vx_sum = 0.0
            
            # Check each potential neighbor and only include if in bounds
            if row > 0 and col < n_cols_h:
                vx_sum += h_vel_grid[row-1, col]  # Top-left
                count += 1
            
            if row < n_rows and col < n_cols_h:
                vx_sum += h_vel_grid[row, col]  # Bottom-left
                count += 1
                
            if row > 0 and col+1 < n_cols_h:
                vx_sum += h_vel_grid[row-1, col+1]  # Top-right
                count += 1
                
            if row < n_rows and col+1 < n_cols_h:
                vx_sum += h_vel_grid[row, col+1]  # Bottom-right
                count += 1
            
            # Calculate average
            if count > 0:
                h_secondary_vx[row, col] = vx_sum / count
    
    # For each VERTICAL edge, calculate secondary vy by averaging from nearby HORIZONTAL edges
    for row in range(n_rows):  # 0 to 9
        for col in range(n_cols_h):  # 0 to 10
            # For a vertical edge at (row,col), get vy from horizontal edges at:
            # (row,col-1), (row+1,col-1), (row,col), (row+1,col)
            count = 0
            vy_sum = 0.0
            
            # Check each potential neighbor and only include if in bounds
            if col > 0 and row < n_rows_v:
                vy_sum += v_vel_grid[row, col-1]  # Top-left
                count += 1
                
            if col > 0 and row+1 < n_rows_v:
                vy_sum += v_vel_grid[row+1, col-1]  # Bottom-left
                count += 1
                
            if col < n_cols and row < n_rows_v:
                vy_sum += v_vel_grid[row, col]  # Top-right
                count += 1
                
            if col < n_cols and row+1 < n_rows_v:
                vy_sum += v_vel_grid[row+1, col]  # Bottom-right
                count += 1
            
            # Calculate average
            if count > 0:
                v_secondary_vy[row, col] = vy_sum / count
    
    # Print some statistics about the secondary velocities
    print(f"\nSecondary velocity components:")
    print(f"  Horizontal edges with non-zero secondary vx: {np.count_nonzero(h_secondary_vx)}")
    print(f"  Vertical edges with non-zero secondary vy: {np.count_nonzero(v_secondary_vy)}")
    
    return h_secondary_vx, v_secondary_vy

# Test the second step of advection
h_secondary_vx, v_secondary_vy = advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions)

def advect_velocities_step3(h_vel_grid, v_vel_grid, h_secondary_vx, v_secondary_vy, 
                           v_locked_positions, h_locked_positions, dt=0.5):
    """
    Step 3 of advection: Backtrack along velocity field and interpolate
    
    This uses semi-Lagrangian advection:
    1. For each edge, backtrack along the velocity field
    2. Find where the fluid "came from"
    3. Interpolate the velocity at that location
    """
    n_rows, n_cols_h = h_vel_grid.shape  # 10x11
    n_rows_v, n_cols = v_vel_grid.shape  # 11x10
    
    # Create new grids to store the updated velocities
    h_vel_grid_new = np.copy(h_vel_grid)
    v_vel_grid_new = np.copy(v_vel_grid)
    
    # Process horizontal velocities (at vertical edges)
    for row in range(n_rows):
        for col in range(n_cols_h):
            # Skip locked edges
            key = (row, col)
            if key in v_locked_positions and v_locked_positions[key]:
                continue
            
            # Current position and velocity
            x = col  # Horizontal coordinate
            y = row + 0.5  # Vertical coordinate (staggered)
            vx = h_vel_grid[row, col]  # Primary velocity component
            vy = v_secondary_vy[row, col]  # Secondary velocity component
            
            # Calculate departure point by backtracking
            x_dep = x - dt * vx
            y_dep = y - dt * vy
            
            # Find grid cell containing departure point
            x_floor = int(x_dep)
            y_floor = int(y_dep - 0.5)  # Adjust for staggering
            
            # Ensure valid bounds
            if x_floor < 0: x_floor = 0
            if x_floor > n_cols_h - 2: x_floor = n_cols_h - 2
            if y_floor < 0: y_floor = 0
            if y_floor > n_rows - 2: y_floor = n_rows - 2
            
            # Compute fractional position within the cell
            x_frac = x_dep - x_floor
            y_frac = y_dep - (y_floor + 0.5)  # Adjust for staggering
            
            # Ensure fractions are between 0 and 1
            x_frac = max(0.0, min(1.0, x_frac))
            y_frac = max(0.0, min(1.0, y_frac))
            
            # Get velocities at the four corners of the cell
            v00 = h_vel_grid[y_floor, x_floor]         # Bottom-left
            v10 = h_vel_grid[y_floor+1, x_floor]       # Top-left
            v01 = h_vel_grid[y_floor, x_floor+1]       # Bottom-right
            v11 = h_vel_grid[y_floor+1, x_floor+1]     # Top-right
            
            # Perform bilinear interpolation
            # First interpolate along x
            v0 = v00 * (1 - x_frac) + v01 * x_frac  # Bottom edge
            v1 = v10 * (1 - x_frac) + v11 * x_frac  # Top edge
            
            # Then interpolate along y
            v_interp = v0 * (1 - y_frac) + v1 * y_frac
            
            # Update velocity
            h_vel_grid_new[row, col] = v_interp
    
    # Process vertical velocities (at horizontal edges)
    for row in range(n_rows_v):
        for col in range(n_cols):
            # Skip locked edges
            key = (row, col)
            if key in h_locked_positions and h_locked_positions[key]:
                continue
            
            # Current position and velocity
            x = col + 0.5  # Horizontal coordinate (staggered)
            y = row  # Vertical coordinate
            vy = v_vel_grid[row, col]  # Primary velocity component
            vx = h_secondary_vx[row, col]  # Secondary velocity component
            
            # Calculate departure point by backtracking
            x_dep = x - dt * vx
            y_dep = y - dt * vy
            
            # Find grid cell containing departure point
            x_floor = int(x_dep - 0.5)  # Adjust for staggering
            y_floor = int(y_dep)
            
            # Ensure valid bounds
            if x_floor < 0: x_floor = 0
            if x_floor > n_cols - 2: x_floor = n_cols - 2
            if y_floor < 0: y_floor = 0
            if y_floor > n_rows_v - 2: y_floor = n_rows_v - 2
            
            # Compute fractional position within the cell
            x_frac = x_dep - (x_floor + 0.5)  # Adjust for staggering
            y_frac = y_dep - y_floor
            
            # Ensure fractions are between 0 and 1
            x_frac = max(0.0, min(1.0, x_frac))
            y_frac = max(0.0, min(1.0, y_frac))
            
            # Get velocities at the four corners of the cell
            v00 = v_vel_grid[y_floor, x_floor]         # Bottom-left
            v10 = v_vel_grid[y_floor+1, x_floor]       # Top-left
            v01 = v_vel_grid[y_floor, x_floor+1]       # Bottom-right
            v11 = v_vel_grid[y_floor+1, x_floor+1]     # Top-right
            
            # Perform bilinear interpolation
            # First interpolate along x
            v0 = v00 * (1 - x_frac) + v01 * x_frac  # Bottom edge
            v1 = v10 * (1 - x_frac) + v11 * x_frac  # Top edge
            
            # Then interpolate along y
            v_interp = v0 * (1 - y_frac) + v1 * y_frac
            
            # Update velocity
            v_vel_grid_new[row, col] = v_interp
    
    # Print statistics about the updated velocities
    print("\nAfter advection step:")
    print(f"  Non-zero horizontal velocities: {np.count_nonzero(h_vel_grid_new)}")
    print(f"  Non-zero vertical velocities: {np.count_nonzero(v_vel_grid_new)}")
    
    return h_vel_grid_new, v_vel_grid_new

# Test the third step of advection
h_vel_grid_new, v_vel_grid_new = advect_velocities_step3(h_vel_grid, v_vel_grid, 
                                                      h_secondary_vx, v_secondary_vy,
                                                      v_locked_positions, h_locked_positions, dt=1.0)

# Create a function to transfer back from grid representation to the original dictionaries
def update_velocities_from_grid(h_velocities, v_velocities, h_vel_grid_new, v_vel_grid_new):
    """Update the velocity dictionaries from the grid representation"""
    
    # Update horizontal velocities in v_velocities dictionary
    for row in range(h_vel_grid_new.shape[0]):
        for col in range(h_vel_grid_new.shape[1]):
            key = (row, col)
            if key in v_velocities:
                v_velocities[key]["vx"] = h_vel_grid_new[row, col]
    
    # Update vertical velocities in h_velocities dictionary
    for row in range(v_vel_grid_new.shape[0]):
        for col in range(v_vel_grid_new.shape[1]):
            key = (row, col)
            if key in h_velocities:
                h_velocities[key]["vy"] = v_vel_grid_new[row, col]
    
    return h_velocities, v_velocities

# Update velocities and visualize
h_velocities, v_velocities = update_velocities_from_grid(h_velocities, v_velocities, h_vel_grid_new, v_vel_grid_new)
visualize_grid(agents, h_velocities, v_velocities, "Grid after one advection step", threshold=1e-8)

# Add this just before running advection:
# Add this just before running advection:

def bootstrap_velocities(h_velocities, v_velocities):
    """Create an initial velocity gradient to help advection start properly"""
    # Find the source locations (right edge)
    max_col = max(agent.col for agent in agents)
    
    # Directly set initial velocities on edges next to the source
    # This creates a gradient that helps advection begin
    for agent in agents:
        if agent.col == max_col - 1:  # Column just to the left of sources
            # Set the right edge to have a small initial velocity
            edge_key = (agent.row, agent.col+1)
            if edge_key in v_velocities and not v_velocities[edge_key]["locked"]:
                v_velocities[edge_key]["vx"] = -1.0  # Half the source magnitude
    
    print("Bootstrapped initial velocities to help advection start")
    return h_velocities, v_velocities

# Bootstrap velocities before running advection
h_velocities, v_velocities = bootstrap_velocities(h_velocities, v_velocities)
visualize_grid(agents, h_velocities, v_velocities, "Grid with bootstrap velocities", threshold=1e-8)

# Run multiple steps of advection
def run_advection(h_velocities, v_velocities, steps=5, dt=1.0):
    """Run multiple steps of advection"""
    print(f"\n=== Running {steps} advection steps with dt={dt} ===")
    
    for step in range(steps):
        print(f"\nAdvection step {step+1}:")
        
        # Step 1: Create grid representation
        h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions = advect_velocities_step1(h_velocities, v_velocities)
        
        # Step 2: Calculate secondary velocity components
        h_secondary_vx, v_secondary_vy = advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions)
        
        # Step 3: Perform advection
        h_vel_grid_new, v_vel_grid_new = advect_velocities_step3(h_vel_grid, v_vel_grid, 
                                                              h_secondary_vx, v_secondary_vy,
                                                              v_locked_positions, h_locked_positions, dt=dt)
        
        # Update velocities
        h_velocities, v_velocities = update_velocities_from_grid(h_velocities, v_velocities, h_vel_grid_new, v_vel_grid_new)
        
        # Visualize every few steps
        if (step+1) % 2 == 0 or step == steps-1:
            visualize_grid(agents, h_velocities, v_velocities, 
                          f"Grid after {step+1} advection steps", 
                          threshold=1e-8)

# Now run advection with larger time steps (AFTER defining the function)
run_advection(h_velocities, v_velocities, steps=8, dt=2.0)

def initialize_all_velocities(h_velocities, v_velocities, magnitude=0.1):
    """Initialize all non-source, non-boundary edges with a small velocity"""
    # Find the source and boundary positions
    max_col = max(agent.col for agent in agents)
    
    # Initialize all vertical edges (horizontal velocities) to a small value
    for (row, col), vel in v_velocities.items():
        if not vel["locked"]:  # Skip already locked edges (source and boundary)
            # Small leftward flow everywhere
            vel["vx"] = -magnitude
    
    print(f"Initialized all non-locked edges with small magnitude {magnitude}")
    return h_velocities, v_velocities

# After setting boundary conditions and before running advection
h_velocities, v_velocities = initialize_all_velocities(h_velocities, v_velocities, magnitude=0.1)
visualize_grid(agents, h_velocities, v_velocities, 
              "Grid with non-zero initialization", threshold=1e-8)

# Now run advection with this initialization
run_advection(h_velocities, v_velocities, steps=8, dt=2.0)


