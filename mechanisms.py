import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd

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

def initialize_source_velocities(agents, h_velocities, v_velocities, magnitude=0.5):
    """Set source velocities at the top edge of all cells in the top row"""
    # Target the top row (row = 0)
    top_row = 8
    right_edge = 39
    other_row = 4

    # Count for reporting
    source_cells = 0
    
    for agent in agents:
        if agent.row == top_row:
            # Mark as source cell
            agent.source = True
            source_cells += 1
            
            # Set south-flowing velocity at the top edge (which is also the north edge of this cell)
            top_edge_key = (agent.row, agent.col)  # Top/north edge of source cell
            h_velocities[top_edge_key]["vy"] = -magnitude  # Positive = downward flow
            h_velocities[top_edge_key]["locked"] = True
        elif agent.col == right_edge and agent.row == other_row:
            agent.source = True
            source_cells += 1
            left_edge_key = (agent.row, agent.col)
            v_velocities[left_edge_key]["vx"] = -magnitude*10
            v_velocities[left_edge_key]["locked"] = True
    
    print(f"Set source velocities on top edges of {source_cells} cells in row {top_row} with magnitude {magnitude}")

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
    for row in range(max_row):
        right_edge = (row, max_col)
        if right_edge in v_velocities:
            # Don't overwrite velocity if it's already non-zero
            if abs(v_velocities[right_edge]["vx"]) < 1e-10:
                v_velocities[right_edge]["vx"] = 0.0
            v_velocities[right_edge]["locked"] = True
    
    # Top boundary (y = 0)
    for col in range(max_col):
        top_edge = (0, col)
        if top_edge in h_velocities:
            # Don't overwrite velocity if it's already non-zero (for source cells)
            if abs(h_velocities[top_edge]["vy"]) < 1e-10:
                h_velocities[top_edge]["vy"] = 0.0
            h_velocities[top_edge]["locked"] = True
    
    # Bottom boundary (y = max_row)
    for col in range(max_col):
        bottom_edge = (max_row, col)
        if bottom_edge in h_velocities:
            h_velocities[bottom_edge]["vy"] = 0.0
            h_velocities[bottom_edge]["locked"] = True
    
    print("Set boundary conditions: all domain boundary edges are now locked")

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

def advect_velocities_step1(h_velocities, v_velocities):
    """Step 1 of advection: Create 2D arrays for horizontal and vertical velocities"""
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # HORIZONTAL VELOCITIES (vx at vertical edges)
    h_vel_grid = np.zeros((n_rows, n_cols+1))
    
    # Fill the grid with values from v_velocities
    for row in range(n_rows):
        for col in range(n_cols+1):
            edge_key = (row, col)
            if edge_key in v_velocities:
                h_vel_grid[row, col] = v_velocities[edge_key]["vx"]
    
    # VERTICAL VELOCITIES (vy at horizontal edges)
    v_vel_grid = np.zeros((n_rows+1, n_cols))
    
    # Fill the grid with values from h_velocities
    for row in range(n_rows+1):
        for col in range(n_cols):
            edge_key = (row, col)
            if edge_key in h_velocities:
                v_vel_grid[row, col] = h_velocities[edge_key]["vy"]
    
    # Create locked position dictionaries
    v_locked_positions = {(row, col): vel["locked"] for (row, col), vel in v_velocities.items()}
    h_locked_positions = {(row, col): vel["locked"] for (row, col), vel in h_velocities.items()}
    
    # Return both velocity grids and locked positions
    return h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions

def advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions):
    """Step 2 of advection: Calculate secondary velocity components by averaging"""
    n_rows, n_cols_h = h_vel_grid.shape  # 10x11
    n_rows_v, n_cols = v_vel_grid.shape  # 11x10
    
    # Create arrays to store the secondary velocity components
    h_secondary_vx = np.zeros((n_rows_v, n_cols))  # Secondary horizontal velocity for horizontal edges
    v_secondary_vy = np.zeros((n_rows, n_cols_h))  # Secondary vertical velocity for vertical edges
    
    # For each HORIZONTAL edge, calculate secondary vx by averaging from nearby VERTICAL edges
    for row in range(n_rows_v):
        for col in range(n_cols):
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
    for row in range(n_rows):
        for col in range(n_cols_h):
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
    
    return h_secondary_vx, v_secondary_vy

def advect_velocities_step3(h_vel_grid, v_vel_grid, h_secondary_vx, v_secondary_vy, 
                           v_locked_positions, h_locked_positions, dt=0.5):
    """Step 3 of advection: Backtrack along velocity field and interpolate"""
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
            v0 = v00 * (1 - x_frac) + v01 * x_frac  # Bottom edge
            v1 = v10 * (1 - x_frac) + v11 * x_frac  # Top edge
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
            v0 = v00 * (1 - x_frac) + v01 * x_frac  # Bottom edge
            v1 = v10 * (1 - x_frac) + v11 * x_frac  # Top edge
            v_interp = v0 * (1 - y_frac) + v1 * y_frac
            
            # Update velocity
            v_vel_grid_new[row, col] = v_interp
    
    return h_vel_grid_new, v_vel_grid_new

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

def advect_velocities(h_velocities, v_velocities, dt=0.5):
    """Complete advection process"""
    # Step 1: Create grid representation
    h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions = advect_velocities_step1(h_velocities, v_velocities)
    
    # Step 2: Calculate secondary velocity components
    h_secondary_vx, v_secondary_vy = advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions)
    
    # Step 3: Perform advection
    h_vel_grid_new, v_vel_grid_new = advect_velocities_step3(h_vel_grid, v_vel_grid, 
                                                          h_secondary_vx, v_secondary_vy,
                                                          v_locked_positions, h_locked_positions, dt=dt)
    
    # Update velocities
    return update_velocities_from_grid(h_velocities, v_velocities, h_vel_grid_new, v_vel_grid_new)

def project_velocities(h_velocities, v_velocities, iterations=20):
    """
    Make the velocity field divergence-free using red-black Gauss-Seidel
    with alternating update order to reduce directional bias
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Step 1: Group agents into red and black cells (checkerboard pattern)
    red_agents = []
    black_agents = []
    for agent in agents:
        if agent.source:
            continue  # Skip source cells
        elif (agent.row + agent.col) % 2 == 0:
            red_agents.append(agent)
        else:
            black_agents.append(agent)
    
    print(f"Red cells: {len(red_agents)}, Black cells: {len(black_agents)}")
    
    # Step 2: Perform red-black Gauss-Seidel iterations with alternating order
    for iteration in range(iterations):
        # Alternate the update order based on iteration number
        if iteration % 2 == 0:
            # Even iterations: red first, then black
            update_order = [(red_agents, "red"), (black_agents, "black")]
        else:
            # Odd iterations: black first, then red
            update_order = [(black_agents, "black"), (red_agents, "red")]
        
        # Process cells in the determined order
        for agents_to_update, color in update_order:
            for agent in agents_to_update:
                project_single_cell(agent)
    
    return h_velocities, v_velocities

def project_single_cell(agent):
    """Apply projection to a single cell to make it divergence-free"""
    # Calculate the net flux (divergence)
    # Outflow is positive, inflow is negative
    
    # Horizontal edges
    flux_n = agent.velocity_n["vy"] if agent.velocity_n else 0.0  # North edge: vy > 0 means outflow
    flux_s = -agent.velocity_s["vy"] if agent.velocity_s else 0.0  # South edge: vy > 0 means inflow, so negate
    
    # Vertical edges
    flux_e = agent.velocity_e["vx"] if agent.velocity_e else 0.0  # East edge: vx > 0 means outflow
    flux_w = -agent.velocity_w["vx"] if agent.velocity_w else 0.0  # West edge: vx > 0 means inflow, so negate
    
    # Calculate total divergence - should be zero for incompressible flow
    divergence = flux_n + flux_e + flux_w + flux_s
 
    # Find valid edges that we can adjust (not locked)
    valid_edges = []
    if agent.velocity_n and not agent.velocity_n["locked"]:
        valid_edges.append(("N", agent.velocity_n))
    if agent.velocity_s and not agent.velocity_s["locked"]:
        valid_edges.append(("S", agent.velocity_s))
    if agent.velocity_e and not agent.velocity_e["locked"]:
        valid_edges.append(("E", agent.velocity_e))
    if agent.velocity_w and not agent.velocity_w["locked"]:
        valid_edges.append(("W", agent.velocity_w))
    
    # If no adjustable edges, we can't fix divergence
    if not valid_edges:
        return
    
    # Calculate correction factor
    correction = divergence / len(valid_edges)
    
    # Apply correction to drive divergence to zero
    for direction, edge in valid_edges:
        if direction in ("N", "S"):
            # For north edge: decrease outflow if divergence > 0
            # For south edge: increase inflow if divergence > 0
            multiplier = 1.0 if direction == "N" else -1.0
            edge["vy"] -= multiplier * correction
        else:  # E or W
            # For east edge: decrease outflow if divergence > 0
            # For west edge: increase inflow if divergence > 0
            multiplier = 1.0 if direction == "E" else -1.0
            edge["vx"] -= multiplier * correction

def run_simulation(h_velocities, v_velocities, steps=5, dt=0.5, projection_iters=20):
    """Run multiple steps of advection and projection"""
    print(f"Running {steps} simulation steps (dt={dt}, projection_iters={projection_iters})")
    
    # Visualize initial state
    visualize_grid(agents, h_velocities, v_velocities, "Initial Grid with Source Velocities")
    
    for step in range(steps):
        print(f"Step {step+1}/{steps}")
        
        # Step 1: Advection
        h_velocities, v_velocities = advect_velocities(h_velocities, v_velocities, dt)
        
        # Step 2: Projection
        h_velocities, v_velocities = project_velocities(h_velocities, v_velocities, projection_iters)
        
        # Visualize every few steps
        if (step+1) % 5 == 0 or step == steps-1:
            visualize_grid(agents, h_velocities, v_velocities, 
                         f"Grid after {step+1} steps (dt={dt})", threshold=1e-8)
    
    return h_velocities, v_velocities

def compute_cell_intensities(agents):
    """Calculate cell intensity values based on edge velocities"""
    intensities = []
    for agent in agents:
        # Sum the absolute value of all edge velocities
        intensity = 0.0
        if agent.velocity_n:
            intensity += abs(agent.velocity_n["vy"])
        if agent.velocity_s:
            intensity += abs(agent.velocity_s["vy"])
        if agent.velocity_e:
            intensity += abs(agent.velocity_e["vx"])
        if agent.velocity_w:
            intensity += abs(agent.velocity_w["vx"])
        intensities.append(intensity)
    return intensities

def plot_heatmap(agents, title="Flow Heatmap", cmap='viridis'):
    """Create a heatmap visualization of flow intensities"""
    # Calculate intensity values
    intensities = compute_cell_intensities(agents)
    
    # Create a DataFrame with intensities
    df = pd.DataFrame({"intensity": intensities})
    
    # Create a GeoDataFrame using agent geometries
    gdf = gpd.GeoDataFrame(df, geometry=[agent.geometry for agent in agents])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(column="intensity", cmap=cmap, legend=True, edgecolor="black", ax=ax)
    
    # Mark source cells with a red outline
    for agent in agents:
        if agent.source:
            x, y = agent.geometry.exterior.xy
            ax.plot(x, y, color='red', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

def visualize_cell_velocity_vectors(agents, h_velocities, v_velocities, title="Cell Velocity Vectors", threshold=1e-6, scale=0.5):
    """Create a visualization with arrows showing net flow direction in each cell"""
    plt.figure(figsize=(10, 10))
    
    # Plot cell boundaries
    for agent in agents:
        xs, ys = agent.geometry.exterior.xy
        plt.plot(xs, ys, color='black', linewidth=0.5)
        
        # Plot source cells with red fill
        if agent.source:
            plt.fill(xs, ys, color='red', alpha=0.2)
    
    # Calculate and plot velocity vectors for each cell
    for agent in agents:
        # Calculate center of cell
        center_x = agent.col + 0.5
        center_y = agent.row + 0.5
        
        # Calculate net velocity components
        net_vx = 0.0
        net_vy = 0.0
        
        # East edge (positive x direction) - vx > 0 means outflow
        if agent.velocity_e:
            net_vx += agent.velocity_e["vx"]
        
        # West edge (negative x direction) - vx > 0 means outflow from left cell, so inflow to this cell
        if agent.velocity_w:
            net_vx -= agent.velocity_w["vx"]
        
        # North edge (positive y direction) - vy > 0 means outflow
        if agent.velocity_n:
            net_vy += agent.velocity_n["vy"]
        
        # South edge (negative y direction) - vy > 0 means outflow from bottom cell, so inflow to this cell
        if agent.velocity_s:
            net_vy -= agent.velocity_s["vy"]
        
        # Only plot arrows with magnitude above threshold
        magnitude = (net_vx**2 + net_vy**2)**0.5
        if magnitude > threshold:
            plt.arrow(center_x, center_y, scale * net_vx, scale * net_vy, 
                     head_width=0.1*magnitude, head_length=0.15*magnitude, 
                     fc='blue', ec='blue', width=0.02*magnitude)
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def run_projection_only(h_velocities, v_velocities, steps=20, projection_iters=20):
    """Run simulation with only projection steps (no advection)"""
    print(f"Running {steps} projection-only steps with {projection_iters} iterations per step")
    
    # Visualize initial state with edge velocity vectors
    visualize_grid(agents, h_velocities, v_velocities, "Initial State Before Projection", threshold=1e-8)
    
    for step in range(steps):
        print(f"Step {step+1}/{steps}")
        
        # Skip advection step - only run projection
        h_velocities, v_velocities = project_velocities(h_velocities, v_velocities, projection_iters)
        
        # Visualize every few steps
        if (step+1) % 1 == 0 or step == steps-1:
            visualize_grid(agents, h_velocities, v_velocities, 
                         f"After {step+1} Projection-Only Steps", threshold=1e-8)
    
    return h_velocities, v_velocities

def run_advection_projection_simulation(h_velocities, v_velocities, steps=20, dt=0.5, projection_iters=1):
    """Run simulation alternating between advection and projection steps"""
    print(f"Running {steps} steps of advection-projection with dt={dt}, projection_iters={projection_iters}")
    
    # Visualize initial state
    visualize_grid(agents, h_velocities, v_velocities, "Initial State", threshold=1e-8)
    
    for step in range(steps):
        print(f"Step {step+1}/{steps}")
        
        # Step 1: Advection
        h_velocities, v_velocities = advect_velocities(h_velocities, v_velocities, dt)
        
        # Visualize after advection (optional)
        if (step+1) % 5 == 0:
            visualize_grid(agents, h_velocities, v_velocities, 
                         f"After Step {step+1} (Advection)", threshold=1e-8)
        
        # Step 2: Projection (single iteration)
        h_velocities, v_velocities = project_velocities(h_velocities, v_velocities, projection_iters)
        
        # Visualize after projection
        if (step+1) % 1 == 0 or step == steps-1:
            visualize_grid(agents, h_velocities, v_velocities, 
                         f"After Step {step+1} (Full)", threshold=1e-8)
    
    return h_velocities, v_velocities


# Create and initialize the grid
agents = create_grid(9, 40)
h_velocities, v_velocities = assign_edge_velocities(agents)
initialize_source_velocities(agents, h_velocities, v_velocities, 0.5)
set_boundary_conditions(h_velocities, v_velocities)

# Create global edges dictionary
global_edges = create_global_edges_dict(h_velocities, v_velocities)

# Run alternating advection-projection simulation
h_velocities, v_velocities = run_advection_projection_simulation(h_velocities, v_velocities, steps=20, dt=0.5, projection_iters=1)

