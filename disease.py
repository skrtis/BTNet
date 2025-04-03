#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from numpy import random
import time 
#All disease mechanisms are stored here.

# concentration dropper, drops a concentration at a defined location
def drop_concentration(agents, concentration, locations):
    """
    Drop a concentration at multiple specific locations on the grid.
    
    Parameters:
    -----------
    agents : list
        List of FlowPolygonAgent objects
    concentration : float
        Concentration value to drop
    locations : list of tuples or tuple
        List of (row, col) positions for dropping the concentration
        Can also accept a single (row, col) tuple for backward compatibility
        
    Returns:
    --------
    agents : list
        Updated list of agents with new concentration values
    """
    # Convert single location to list for uniform processing
    if isinstance(locations, tuple) and len(locations) == 2:
        locations = [locations]
    elif locations is None:
        return agents  # No locations provided
    
    # Track which locations were actually found in the grid
    found_locations = []
    
    # Process each drop location
    for location in locations:
        # Skip empty or invalid locations
        if not location or len(location) != 2:
            continue
            
        row, col = location
        
        # Find the agent at the specified location and update its concentration
        for agent in agents:
            if agent.row == row and agent.col == col:
                agent.concentration += concentration
                found_locations.append((row, col))
                break
    
    # Add a very small amount to all water cells to avoid zero-concentration cells
    # which might get excluded in the advection process
    for agent in agents:
        if agent.water == True:
            agent.concentration += 1e-7
    
    # Report the locations where concentration was dropped
    if found_locations:
        print(f"Dropped concentration {concentration} at locations: {found_locations}")
    else:
        print("Warning: No valid drop locations were found in the grid")
        
    return agents

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
    water_agents = [agent for agent in agents if hasattr(agent, "concentration") and agent.water == True]
    
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
        
        # Store the new concentration value (change stickiness with blending)
        blend = 1
        new_concentrations[agent] = ((1-blend)*agent.concentration) + blend*c_interp
    
    # Update all agent concentration values
    for agent, new_concentration in new_concentrations.items():
        agent.concentration = new_concentration

    return agents

def advect_btn_concentrations(agents, h_velocities, v_velocities, dt=0.5):
    """
    Advect btn_concentration values stored at cell centers using semi-Lagrangian method.
    
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
        Updated list of agents with new btn_concentration values
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create a grid to store btn_concentration values
    concentration_grid = np.zeros((n_rows, n_cols))
    
    # Fill the grid with btn_concentration values from agents
    for agent in agents:
        if hasattr(agent, "btn_concentration"):
            concentration_grid[agent.row, agent.col] = agent.btn_concentration
    
    # Create new btn_concentration values (don't update immediately to avoid affecting other calculations)
    new_concentrations = {}
    
    # Process only water agents (those with btn_concentration > 0)
    water_agents = [agent for agent in agents if hasattr(agent, "btn_concentration") and agent.btn_concentration > 0]
    
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
            # Keep the original btn_concentration if backtracking goes out of bounds
            new_concentrations[agent] = agent.btn_concentration
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
        
        # Get btn_concentration values at the four corners
        c00 = concentration_grid[y_floor, x_floor]          # Bottom-left
        c01 = concentration_grid[y_floor, min(x_floor + 1, n_cols - 1)]      # Bottom-right
        c10 = concentration_grid[min(y_floor + 1, n_rows - 1), x_floor]      # Top-left
        c11 = concentration_grid[min(y_floor + 1, n_rows - 1), min(x_floor + 1, n_cols - 1)]  # Top-right
        
        # Perform bilinear interpolation
        c0 = c00 * (1 - x_frac) + c01 * x_frac  # Bottom edge
        c1 = c10 * (1 - x_frac) + c11 * x_frac  # Top edge
        c_interp = c0 * (1 - y_frac) + c1 * y_frac
        
        # Store the new btn_concentration value (change stickiness with blending)
        blend = 0.3 
        new_concentrations[agent] = ((1-blend)*agent.btn_concentration) + blend*c_interp
    
    # Update all agent btn_concentration values
    for agent, new_concentration in new_concentrations.items():
        agent.btn_concentration = new_concentration

    return agents

# clam population lists
population1 = [(19, 0), (18, 0), (18, 1), (19, 1), (20, 1)]
population2 = [(22, 9), (21, 9), (21, 10), (22, 10), (23, 11), (23, 12), (23, 13), (24, 13), (22, 14), (23, 14), (24, 14), (22, 15), (23, 15), (24, 15)]
population3 = [(52, 12), (51, 12), (51, 13), (51, 14), (51, 15)]
population4 = [(41, 16), (41, 17), (42, 18)]
population5 = [(64, 21), (63, 21), (63, 22), (62, 23), (63, 23), (61, 23), (61, 24), (62, 24), (60, 24), (60, 25), (61, 25), (59, 25), (59, 26), (60, 26), (58, 27), (59, 27), (57, 28), (58, 28), (56, 29), (57, 29), (57, 30)]
population6 = [(54, 21), (53, 21), (53, 22), (54, 22), (52, 21), (52, 22), (54, 23), (51, 21), (51, 22), (51, 23), (54, 24), (51, 24), (53, 25), (54, 25), (51, 25), (52, 26), (53, 26), (51, 26)]
population7 = [(8, 21), (7, 21), (7, 22), (8, 22), (7, 23), (7, 24), (6, 25), (7, 25), (6, 26), (7, 26), (6, 27), (7, 27), (5, 28), (6, 28), (5, 29), (6, 29)]
population8 = [(24, 24), (23, 24), (23, 25), (24, 25), (22, 24), (22, 25), (22, 26), (23, 26), (21, 25), (21, 26), (21, 27), (22, 27), (20, 26), (20, 27)]
population9 = [(16, 28), (15, 28), (15, 29), (16, 29), (17, 29), (14, 28), (14, 29), (14, 30), (15, 30), (16, 30), (17, 30), (18, 30), (13, 28), (13, 29), (13, 30)]
population10 = [(52, 30), (51, 30), (51, 31), (52, 31), (53, 31), (50, 30), (50, 32), (54, 31), (49, 30), (49, 31), (50, 33), (51, 33), (51, 34), (51, 35), (52, 35), (52, 36), (51, 37), (52, 37), (50, 38), (51, 38), (52, 38), (49, 39), (50, 39), (51, 39), (52, 39), (48, 40), (49, 40), (50, 40), (51, 40), (47, 41), (48, 41), (49, 41), (48, 42), (49, 42)]
population11 = [(30, 30), (29, 31), (30, 31), (31, 31), (28, 31), (31, 32), (32, 32), (27, 32), (31, 33), (32, 33), (27, 33), (30, 34), (31, 34), (32, 34), (26, 34), (29, 35), (30, 35), (26, 35), (27, 35), (28, 35), (28, 36), (26, 36), (27, 36)]
population12 = [(3, 31), (2, 31), (2, 32), (3, 32), (4, 32), (1, 31), (1, 32), (3, 33), (4, 33), (5, 33), (3, 34), (4, 34), (5, 34), (3, 35), (4, 35), (5, 35), (3, 36), (4, 36), (5, 36), (4, 37), (5, 37), (4, 38)]
population13 = [(34, 39), (33, 39), (34, 40), (32, 39), (32, 40), (33, 41), (34, 41), (32, 41)]
population14 = [(30, 42), (31, 43), (31, 44), (31, 45)]
population15 = [(2, 43), (1, 43), (1, 44), (0, 44), (0, 45), (1, 45), (0, 46), (1, 46), (0, 47), (1, 47), (1, 48), (2, 48), (1, 49), (2, 49), (3, 48), (3, 49), (4, 47), (4, 48), (5, 47), (5, 48)]
population16 = [(27, 47), (26, 47), (26, 48), (27, 48), (26, 49), (27, 49)]
population17 = [(16, 47)]
population18 = [(6, 51), (6, 52), (7, 52)]
population19 = [(28, 52), (27, 52), (26, 52)]
population20 = [(38, 53), (38, 54)]
population21 = [(40, 56), (39, 57), (40, 57), (39, 58), (38, 59), (39, 59)]
population22 = [(35, 57), (34, 57)]
population23 = [(34, 59), (34, 60)]

populations = [
    population1, population2, population3, population4, population5,
    population6, population7, population8, population9, population10,
    population11, population12, population13, population14, population15,
    population16, population17, population18, population19, population20,
    population21, population22, population23
]

def update_clam_population(agents, iteration=0):
    # BTN natural decay in water cells (50% every 100 iterations)
    if iteration % 50 == 0 and iteration > 0:
        for agent in agents:
            if agent.water == True and not agent.clam_presence:
                agent.btn_concentration *= 0.5

    for agent in agents:
        if agent.clam_presence == True:
            # Calculate new infections
            F = 0.0524 #m^3 per day
            cells_per_clam = F*agent.btn_concentration
            # Log-logistic parameters
            D50 = 800   # dose at which infection probability is 0.5 (ID50, in BTN cells per clam)
            beta = 1.0   # shape parameter; adjust to control steepness of the curve

            # Calculate infection probability using the log-logistic function
            # This formulation is equivalent to: 1 / (1 + (D50/cells_per_clam)**beta)
            infection_probability = 1 / (1 + np.exp(-beta * (np.log(cells_per_clam) - np.log(D50))))

            # Number of newly infected clams
            new_infections = int(agent.healthy_clams * infection_probability)

            # Update populations
            agent.healthy_clams -= new_infections
            agent.latent_clams += new_infections

            # transition some from latent to infected
            transition_prob = 0.01
            num_transitioned = int(agent.latent_clams * transition_prob)
            agent.latent_clams -= num_transitioned
            agent.infected_clams += num_transitioned

            # Calculate deaths (assuming death rate is 1% to 5% of infected clams per 20 timestep)
            new_deaths = int(agent.infected_clams * np.random.uniform(0.005, 0.01))
            agent.infected_clams -= new_deaths
            agent.dead_clams += new_deaths
            
            # Dynamic K_d: concentration per cell at which the drug is 50% effective
            K_d = (agent.latent_clams+agent.infected_clams)*(0.11)/ 1250000  # equivalent to agent.infected_clams * concentration each clam needs (quarter pounders) and volume of area (1/1250000)

            # Calculate drug effect (a value between 0 and 1)
            if K_d>0:
                drug_effect = agent.concentration / (agent.concentration + K_d)
            else:
                drug_effect = 0

            # Set the maximum recovery rate (e.g., maximum fraction of infected clams that can recover per iteration)
            max_recovery_rate = 0.5  # This means that at full drug effect, up to 50% of infected clams could recover

            # Calculate the actual recovery probability per infected clam
            recovery_prob = drug_effect * max_recovery_rate

            # Determine the number of recovered clams stochastically
            recovered_infected = np.random.binomial(agent.infected_clams, recovery_prob)
            agent.infected_clams -= recovered_infected
            recovered = np.random.binomial(agent.latent_clams, recovery_prob)
            agent.latent_clams -= recovered
            agent.healthy_clams += recovered_infected + recovered

            # Remove some of the drug concentration based on how much was "used"
            agent.concentration -= (recovered_infected+recovered)*0.11/1250000
            agent.concentration = max(agent.concentration, 0)  # ensure concentration doesn't go negative
            
            # Update BTN release based on current infected population
            agent.btn_concentration += (agent.infected_clams * 100) / 1250000 # n/v = c
        

    return agents

def initialize_clam_cancer(agents, population_id, population_id2=None):
    """
    Initialize BTN disease in specified clam population(s)
    
    Parameters:
    -----------
    agents : list
        List of agent objects
    population_id : int
        Primary population ID to infect (1-based index)
    population_id2 : int or None
        Optional second population ID to infect
    
    Returns:
    --------
    agents : list
        Updated list of agents
    """
    print(f"Initializing BTN disease in population {population_id}" + 
          (f" and {population_id2}" if population_id2 is not None else ""))
    
    # First set a very small background concentration in all water cells
    for agent in agents: 
        if agent.water == True or agent.clam_presence == True:
            agent.btn_concentration = 1e-5

    # Infect primary population
    if 1 <= population_id <= len(populations):
        for agent in agents:
            for clams in populations[population_id-1]:
                if agent.row == clams[0] and agent.col == clams[1]:
                    agent.infected_clams = 250000
                    agent.healthy_clams = 500000
                    agent.dead_clams = 0
                    agent.latent_clams = 500000
                    # Set initial BTN concentration based on infected clams
                    agent.btn_concentration = (agent.infected_clams * 100)/1250000

    # Infect secondary population if specified
    if population_id2 is not None and 1 <= population_id2 <= len(populations):
        for agent in agents:
            for clams in populations[population_id2-1]:
                if agent.row == clams[0] and agent.col == clams[1]:
                    agent.infected_clams = 250000
                    agent.healthy_clams = 500000
                    agent.dead_clams = 0
                    agent.latent_clams = 500000
                    # Set initial BTN concentration based on infected clams
                    agent.btn_concentration = (agent.infected_clams * 100)/1250000
                    
    return agents












