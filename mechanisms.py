import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import time

from disease import advect_concentrations,drop_concentration, advect_btn_concentrations,update_clam_population,initialize_clam_cancer

def wind(agents):
    """
    Calculate wind velocity for each agent based on its position.
    
    Parameters:
    -----------
    agents : list
        List of FlowPolygonAgent objects
    h_velocities : dict
        Dictionary of horizontal velocities at horizontal edges
    v_velocities : dict
        Dictionary of vertical velocities at vertical edges
        
    Returns:
    --------
    agents : list
        Updated list of agents with wind velocity attributes
    """
    #randomized wind direction
    wind_dir = np.random.choice(["N", "S", "E", "W"])
    for agent in agents: 
        wind_speed =  np.random.uniform(0.00001,0.00005)
        if wind_dir == "N":
            agent.velocity_n["vy"] += wind_speed
        elif wind_dir == "S":
            agent.velocity_s["vy"] -= wind_speed
        elif wind_dir == "E":
            agent.velocity_e["vx"] += wind_speed
        elif wind_dir == "W":
            agent.velocity_w["vx"] -= wind_speed
    return agents

def advect_velocities_step1(h_velocities, v_velocities,agents):
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

            # Check if departure point is out of bounds
            if (x_dep < 0 or x_dep > n_cols_h - 1 or 
                y_dep - 0.5 < 0 or y_dep - 0.5 > n_rows - 1):
                # Keep the original velocity if backtracking goes out of bounds
                continue  # Skip interpolation, keeping the original value 
            
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

def advect_velocities(h_velocities, v_velocities, agents,dt=0.5):
    """complete advection process"""
    # step 1: create grid representation
    h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions = advect_velocities_step1(h_velocities, v_velocities,agents)
    
    # step 2: calculate secondary velocity components
    h_secondary_vx, v_secondary_vy = advect_velocities_step2(h_vel_grid, v_vel_grid, v_locked_positions, h_locked_positions)
    
    # step 3: perform advection
    h_vel_grid_new, v_vel_grid_new = advect_velocities_step3(h_vel_grid, v_vel_grid, 
                                                          h_secondary_vx, v_secondary_vy,
                                                          v_locked_positions, h_locked_positions, dt=dt)
    
    # update velocities
    return update_velocities_from_grid(h_velocities, v_velocities, h_vel_grid_new, v_vel_grid_new)

def project_velocities(h_velocities, v_velocities, agents, overrelaxation=1.5, iterations=10):
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
    
    #print(f"Red cells: {len(red_agents)}, Black cells: {len(black_agents)}")
    
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
                project_single_cell(agent, overrelaxation=1.0)
    
    return h_velocities, v_velocities

def project_single_cell(agent,overrelaxation=1.0):
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

    divergence *= overrelaxation
 
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

def run_simulation(h_velocities, v_velocities, agents, 
                  num_iterations, 
                  advection_loops, 
                  projection_loops,
                  plot_interval, 
                  overrelaxation,
                  dt,
                  drug_drop,
                  drug_concentration,
                  drug_drop_iteration,
                  visualize_fn=None,
                  save_plots=False,
                  output_dir="./output"):
    """
    Run a fluid dynamics simulation with configurable parameters.
    
    Parameters:
    -----------
    h_velocities : dict
        Dictionary of horizontal velocities
    v_velocities : dict
        Dictionary of vertical velocities
    agents : list
        List of FlowPolygonAgent objects
    num_iterations : int
        Total number of simulation iterations
    advection_loops : int
        Number of advection steps per iteration
    projection_loops : int
        Number of projection iterations per step
    plot_interval : int
        How often to visualize results (every N iterations)
    dt : float
        Timestep for advection
    visualize_fn : function
        Optional function to visualize the simulation state
        Should accept (h_velocities, v_velocities, agents, iteration)
    save_plots : bool
        Whether to save plots to disk (ignored when using plt.show)
    output_dir : str
        Directory to save plots if save_plots is True (ignored when using plt.show)
    
    Returns:
    --------
    h_velocities, v_velocities : updated velocity dictionaries
    """
    # Store initial state (iteration 0)
    if visualize_fn and plot_interval > 0:
        visualize_fn(h_velocities, v_velocities, agents)
        plt.show()
        plt.close()
    
    # Start timing
    start_time = time.time()


    # Main simulation loop
    for iteration in range(1, num_iterations + 1): 
        if iteration == 1: 
            # Initialize clam cancer at the beginning of the simulation
            initialize_clam_cancer(agents,10)
        
        if iteration == drug_drop_iteration:
            # Drop drug concentration at specified location
            agents = drop_concentration(agents, drug_concentration, drug_drop[0], drug_drop[1])
            print(f"Dropping drug at iteration {iteration} at {drug_drop}")
        # Update wind velocities
        agents = wind(agents)
        # Perform projection steps to enforce incompressibility
        for _ in range(projection_loops):
            h_velocities, v_velocities = project_velocities(
                h_velocities, v_velocities, agents, overrelaxation, iterations=1
            )
        # Perform multiple advection steps if specified
        for _ in range(advection_loops):
            h_velocities, v_velocities = advect_velocities(h_velocities, v_velocities, agents, dt=dt) 


        #agents = concentration_spread(agents,dt=dt)
        agents = advect_concentrations(agents, h_velocities, v_velocities, dt=dt)
        agents = advect_btn_concentrations(agents, h_velocities, v_velocities, dt=dt)


        agents = update_clam_population(agents)
        # Calculate velocity statistics more robustly
        # Extract all velocity components into a single list for simpler max/min calculation
        all_velocities = []
        velocity_magnitudes = []
        
        # Get all velocity components from horizontal edges
        for edge_dict in h_velocities.values():
            vy = edge_dict.get("vy", 0)  # Primary component
            all_velocities.append(vy)
            velocity_magnitudes.append(abs(vy))
            
            if "vx" in edge_dict:  # Secondary component (if exists)
                vx = edge_dict["vx"]
                all_velocities.append(vx)
                velocity_magnitudes.append(abs(vx))
        
        # Get all velocity components from vertical edges
        for edge_dict in v_velocities.values():
            vx = edge_dict.get("vx", 0)  # Primary component
            all_velocities.append(vx)
            velocity_magnitudes.append(abs(vx))
            
            if "vy" in edge_dict:  # Secondary component (if exists)
                vy = edge_dict["vy"]
                all_velocities.append(vy)
                velocity_magnitudes.append(abs(vy))
        
        # Calculate overall stats
        max_velocity = max(velocity_magnitudes) if velocity_magnitudes else 0
        min_velocity = min(velocity_magnitudes) if velocity_magnitudes else 0
        avg_velocity = sum(velocity_magnitudes) / len(velocity_magnitudes) if velocity_magnitudes else 0
        
        # Print simplified statistics
        print(f"Iteration {iteration}: Max Velocity: {max_velocity:.4f}, Min Velocity: {min_velocity:.8f}, Avg Velocity: {avg_velocity:.8f}")

        # Apply global dampening if average velocity exceeds threshold
        threshold = 0.0008
        target_avg = 0.0005
        
        if avg_velocity > threshold:
            print(f"  Need dampening: current avg = {avg_velocity:.6f}, target = {target_avg:.6f}")
            
            # Apply progressive dampening that targets high velocities more aggressively
            for key, edge_dict in h_velocities.items():
                # Skip locked edges and edges connected to source cells
                edge_locked = edge_dict.get("locked", False)
                
                # Check if this edge is part of a source cell
                row, col = key
                edge_is_source = False
                for agent in agents:
                    if hasattr(agent, "source") and agent.source:
                        # Check if this edge belongs to the source cell
                        if (agent.row == row or agent.row == row-1) and agent.col == col:
                            edge_is_source = True
                            break
                
                if not edge_locked and not edge_is_source:
                    # Apply progressive dampening to each velocity component
                    if "vy" in edge_dict:
                        # Calculate a custom dampening factor based on velocity magnitude
                        magnitude = abs(edge_dict["vy"])
                        if magnitude > 0:
                            # Progressive dampening: stronger for larger velocities
                            # velocities > 0.005 get heavily dampened
                            # velocities near average get moderately dampened
                            # small velocities get lightly dampened
                            factor = 1.0
                            if magnitude > 0.005:
                                factor = 0.3  # 70% reduction for very large velocities
                            elif magnitude > 0.002:
                                factor = 0.6  # 40% reduction for large velocities
                            elif magnitude > threshold:
                                factor = 0.8  # 20% reduction for above-threshold velocities
                            else:
                                factor = 0.9  # 10% reduction for smaller velocities
                            
                            edge_dict["vy"] *= factor
                    
                    if "vx" in edge_dict:
                        magnitude = abs(edge_dict["vx"])
                        if magnitude > 0:
                            factor = 1.0
                            if magnitude > 0.005:
                                factor = 0.3
                            elif magnitude > 0.002:
                                factor = 0.6
                            elif magnitude > threshold:
                                factor = 0.8
                            else:
                                factor = 0.9
                            
                            edge_dict["vx"] *= factor
            
            # Apply the same progressive dampening to vertical velocities
            for key, edge_dict in v_velocities.items():
                # Skip locked edges and edges connected to source cells
                edge_locked = edge_dict.get("locked", False)
                
                # Check if this edge is part of a source cell
                row, col = key
                edge_is_source = False
                for agent in agents:
                    if hasattr(agent, "source") and agent.source:
                        # Check if this edge belongs to the source cell
                        if agent.row == row and (agent.col == col or agent.col == col-1):
                            edge_is_source = True
                            break
                
                if not edge_locked and not edge_is_source:
                    # Apply progressive dampening to each velocity component
                    if "vx" in edge_dict:
                        magnitude = abs(edge_dict["vx"])
                        if magnitude > 0:
                            factor = 1.0
                            if magnitude > 0.005:
                                factor = 0.3
                            elif magnitude > 0.002:
                                factor = 0.6
                            elif magnitude > threshold:
                                factor = 0.8
                            else:
                                factor = 0.9
                            
                            edge_dict["vx"] *= factor
                    
                    if "vy" in edge_dict:
                        magnitude = abs(edge_dict["vy"])
                        if magnitude > 0:
                            factor = 1.0
                            if magnitude > 0.005:
                                factor = 0.3
                            elif magnitude > 0.002:
                                factor = 0.6
                            elif magnitude > threshold:
                                factor = 0.8
                            else:
                                factor = 0.9
                            
                            edge_dict["vy"] *= factor
            
            print(f"  Applied progressive dampening based on velocity magnitude")

        # Visualize if it's a plotting interval
        if visualize_fn and plot_interval > 0 and iteration % plot_interval == 0:
            print(f"Iteration {iteration}/{num_iterations}")
            visualize_fn(h_velocities, v_velocities, agents, iteration)
            plt.show()
            plt.close()
    
    # End timing
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return h_velocities, v_velocities