#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
import time 
#All disease mechanisms are stored here.





def advect_concentrations(agents, h_velocities, v_velocities, dt=0.5):
    """
    Advect concentration values stored at cell centers using semi-Lagrangian method.
    
    Parameters:
    -----------
    agents : list
        List of FlowPolygonAgent objects
    h_velocities : dict
        Dictionary of horizontal velocities at horizontal edges
    v_velocities : dict
        Dictionary of vertical velocities at vertical edges
    dt : float
        Timestep for advection
        
    Returns:
    --------
    agents : list
        Updated list of agents with new concentration values
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create a grid to store concentration values
    concentration_grid = np.zeros((n_rows, n_cols))
    
    # Fill the grid with concentration values from agents
    for agent in agents:
        if hasattr(agent, "concentration"):
            concentration_grid[agent.row, agent.col] = agent.concentration
    
    # Create new concentration values (don't update immediately to avoid affecting other calculations)
    new_concentrations = {}
    
    # Process only water agents (those with concentration > 0)
    water_agents = [agent for agent in agents if hasattr(agent, "concentration") and agent.concentration > 0]
    
    for agent in water_agents:
        row, col = agent.row, agent.col
        
        # Interpolate velocity at cell center from edge velocities
        vx = 0.0
        vy = 0.0
        count_x = 0
        count_y = 0
        
        # Check west edge (for vx)
        if agent.velocity_w:
            vx += agent.velocity_w["vx"]
            count_x += 1
            
        # Check east edge (for vx)
        if agent.velocity_e:
            vx += agent.velocity_e["vx"]
            count_x += 1
            
        # Check south edge (for vy)
        if agent.velocity_s:
            vy += agent.velocity_s["vy"]
            count_y += 1
            
        # Check north edge (for vy)
        if agent.velocity_n:
            vy += agent.velocity_n["vy"]
            count_y += 1
        
        # Average the velocities
        if count_x > 0:
            vx /= count_x
        if count_y > 0:
            vy /= count_y
        
        # Current position (cell center)
        x = col + 0.5
        y = row + 0.5
        
        # Calculate departure point by backtracking
        x_dep = x - dt * vx
        y_dep = y - dt * vy
        
        # Check if departure point is out of bounds
        if (x_dep < 0 or x_dep >= n_cols or
            y_dep < 0 or y_dep >= n_rows):
            # Keep the original concentration if backtracking goes out of bounds
            new_concentrations[agent] = agent.concentration
            continue
        
        # Find the cell containing the departure point
        x_floor = int(x_dep)
        y_floor = int(y_dep)
        
        # Ensure we're within grid bounds
        x_floor = max(0, min(n_cols - 2, x_floor))
        y_floor = max(0, min(n_rows - 2, y_floor))
        
        # Compute fractional position within the cell
        x_frac = x_dep - x_floor
        y_frac = y_dep - y_floor
        
        # Ensure fractions are between 0 and 1
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        
        # Get concentration values at the four corners
        c00 = concentration_grid[y_floor, x_floor]          # Bottom-left
        c01 = concentration_grid[y_floor, min(x_floor + 1, n_cols - 1)]      # Bottom-right
        c10 = concentration_grid[min(y_floor + 1, n_rows - 1), x_floor]      # Top-left
        c11 = concentration_grid[min(y_floor + 1, n_rows - 1), min(x_floor + 1, n_cols - 1)]  # Top-right
        
        # Perform bilinear interpolation
        c0 = c00 * (1 - x_frac) + c01 * x_frac  # Bottom edge
        c1 = c10 * (1 - x_frac) + c11 * x_frac  # Top edge
        c_interp = c0 * (1 - y_frac) + c1 * y_frac
        
        # Store the new concentration value
        blend = 0.6 
        new_concentrations[agent] = ((1-blend)*agent.concentration) + blend*c_interp
    
    # Update all agent concentration values
    for agent, new_concentration in new_concentrations.items():
        agent.concentration = new_concentration

    return agents

def update_clam_population(agents):
    