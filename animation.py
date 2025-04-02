import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import from grid_generator
from grid_generator import (
    extract_grid_indices,
    transform_indices,
    create_grid_agents,
    assign_edge_velocities
)

# Import from mechanisms
from mechanisms import (
    advect_velocities,
    project_velocities,
    update_velocities_from_grid,
    wind,
    apply_progressive_dampening  # Add this import
)

# Import from disease module
from disease import (
    advect_btn_concentrations, 
    advect_concentrations, 
    drop_concentration, 
    update_clam_population, 
    initialize_clam_cancer, 
    populations
)

def setup_simulation(population_id=10):
    """Initialize the simulation environment"""
    # Load and transform grid data
    grid_data = extract_grid_indices("malpeque_tiles.geojson")
    transformed_data = transform_indices(grid_data)
    
    # Create agents and assign velocities
    agents = create_grid_agents(transformed_data)
    h_velocities, v_velocities = assign_edge_velocities(agents)
    
    # Initialize clam disease parameters if needed
    #agents = initialize_clam_cancer(agents, population_id)
    
    return agents, h_velocities, v_velocities

def init_data_storage():
    """Initialize data storage for tracking clam population statistics over time"""
    # Create a dictionary to store time series data for each population
    population_data = {
        f"population{i+1}": {
            "time": [],
            "healthy": [],
            "latent": [],  # Add latent data field
            "infected": [],
            "dead": []
        } for i in range(len(populations))
    }
    return population_data

def plot_concentration(ax, agents, concentration_attr="concentration", title="Concentration", cbar_ax=None, cmap='viridis'):
    """Plot concentration heatmap with proper colorbar handling"""
    ax.clear()
    ax.set_title(title)
    
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create a grid for concentrations
    concentration_grid = np.zeros((n_rows, n_cols))
    
    # Fill the grid with concentration values and track max value
    max_value = 0
    for agent in agents:
        if hasattr(agent, concentration_attr):
            value = getattr(agent, concentration_attr)
            concentration_grid[agent.row, agent.col] = value
            max_value = max(max_value, value)
    
    # Use imshow to display concentration as a heatmap
    im = ax.imshow(concentration_grid, origin='lower', cmap=cmap, 
                  interpolation='nearest', aspect='equal',
                  extent=[0, n_cols, 0, n_rows],
                  vmin=0, vmax=max_value if max_value > 0 else None)
    
    # Add colorbar if axis provided
    if cbar_ax is not None:
        # Clear existing colorbar
        cbar_ax.clear()
        plt.colorbar(im, cax=cbar_ax, label=f"{concentration_attr.capitalize()} (max: {max_value:.6f})")
    else:
        plt.colorbar(im, ax=ax, label=f"{concentration_attr.capitalize()} (max: {max_value:.6f})", shrink=0.6)
    
    # Add cell outlines for reference
    for i in range(n_cols + 1):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(n_rows + 1):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
    
    # Mark non-water cells with crosshatching
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                                fill=False, hatch='x', 
                                edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # Configure plot
    ax.set_xlabel('Column', fontsize=8)
    ax.set_ylabel('Row', fontsize=8)
    ax.grid(False)
    ax.tick_params(labelsize=7)

def plot_population_data(ax, pop_name, time_data, healthy_data, infected_data, dead_data, latent_data):
    """Plot population data on a given axis"""
    ax.clear()
    ax.set_title(f"{pop_name}", fontsize=8)
    
    # Plot the data
    ax.plot(time_data, healthy_data, 'g-', label='Healthy', linewidth=1)
    ax.plot(time_data, latent_data, 'y-', label='Latent', linewidth=1)  # Add latent clams
    ax.plot(time_data, infected_data, 'r-', label='Infected', linewidth=1)
    ax.plot(time_data, dead_data, 'k-', label='Dead', linewidth=1)
    
    # Format the plot
    ax.set_xlabel('Time', fontsize=6)
    ax.set_ylabel('Count', fontsize=6)
    ax.tick_params(labelsize=6)
    
    # Add legend to first plot only
    if pop_name == "Population 1":
        ax.legend(fontsize=6)

def create_visualization_layout(population_id=10):
    """Create the visualization layout with actual data from the simulation"""
    # Create figure with the layout
    fig = plt.figure(figsize=(20, 12))
    
    # Define the subplot layout using mosaic
    mosaic_layout = []
    
    # Make concentration plots the same size (3 cols each)
    first_row = ['C', 'C', 'C', 'B', 'B', 'B']
    second_row = ['C', 'C', 'C', 'B', 'B', 'B']
    third_row = ['C', 'C', 'C', 'B', 'B', 'B']
    mosaic_layout.append(first_row)
    mosaic_layout.append(second_row)
    mosaic_layout.append(third_row)
    
    # Create rows for population plots, 6 per row
    pop_count = 1
    for row in range(1, 5):   
        row_layout = []
        for col in range(6):
            if pop_count <= 23:
                row_layout.append(str(pop_count))
                pop_count += 1
            else:
                row_layout.append('.')
        mosaic_layout.append(row_layout)
    
    # Create the subplot mosaic with adjusted height ratios - make concentration plots taller
    axd = fig.subplot_mosaic(mosaic_layout, 
                            gridspec_kw={'height_ratios': [1, 1, 1, 0.8, 0.8, 0.8, 0.8]})
    
    # Add dedicated colorbar axes
    divider1 = make_axes_locatable(axd['C'])
    divider2 = make_axes_locatable(axd['B'])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    axd['C_cbar'] = cax1
    axd['B_cbar'] = cax2
    
    # Initialize simulation
    print("Initializing simulation...")
    agents, h_velocities, v_velocities = setup_simulation(population_id)
    print(f"Created {len(agents)} agents")
    
    # Plot concentration data with specific colormaps
    plot_concentration(axd['C'], agents, "concentration", "Drug Concentration (mg/m^3)", cbar_ax=axd['C_cbar'], cmap='viridis')
    plot_concentration(axd['B'], agents, "btn_concentration", "BTN Concentration (cells/m^3)", cbar_ax=axd['B_cbar'], cmap='cividis')
    
    # Initialize data storage for population plots - empty for initial state
    population_data = init_data_storage()
    
    # Add a title to the figure
    fig.suptitle(f"Clam Disease Simulation (Patient Zero: Population {population_id})", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axd, agents, h_velocities, v_velocities, population_data

def update_simulation(agents, h_velocities, v_velocities, population_data, frame_num, 
                     dt=0.1, projection_loops=10, overrelaxation=1.5, 
                     drug_drop=None, drug_concentration=5, drug_drop_frame=50,
                     population_id=5, population_id2=None, btn_zero_frame=30):
    """
    Update the simulation state for a single frame of animation.
    
    Parameters:
    -----------
    agents : list
        List of FlowPolygonAgent objects
    h_velocities, v_velocities : dict
        Dictionaries of horizontal and vertical velocities
    population_data : dict
        Dictionary to store population statistics over time
    frame_num : int
        Current frame number
    dt : float
        Timestep for advection
    projection_loops : int
        Number of projection iterations per step
    overrelaxation : float
        Overrelaxation parameter for pressure projection
    drug_drop : tuple(int, int) or None
        (row, col) position to add drug concentration, or None to skip
    drug_concentration : float
        Concentration value to drop
    drug_drop_frame : int
        Frame number at which to drop the concentration
    population_id : int
        ID of the primary population to initialize disease in
    population_id2 : int or None
        ID of the secondary population to initialize disease in (if any)
    btn_zero_frame : int
        Frame number at which to initialize the disease
        
    Returns:
    --------
    agents, h_velocities, v_velocities, population_data
        Updated simulation state
    """
    # Initialize clam cancer if this is the right frame
    if frame_num == btn_zero_frame:
        print(f"Initializing BTN patient zero at frame {frame_num} in population {population_id}" +
              (f" and {population_id2}" if population_id2 is not None else ""))
        initialize_clam_cancer(agents, population_id=population_id, population_id2=population_id2)

    # Drop drug concentration if this is the right frame
    if drug_drop and frame_num == drug_drop_frame:
        print(f"Dropping drug concentration at frame {frame_num} at position {drug_drop}")
        agents = drop_concentration(agents, drug_concentration, drug_drop[0], drug_drop[1])
    
    # Rest of the function remains the same...
    # Update wind velocities
    agents = wind(agents)
    
    # Perform projection steps to enforce incompressibility
    for _ in range(projection_loops):
        h_velocities, v_velocities = project_velocities(
            h_velocities, v_velocities, agents, overrelaxation, iterations=1
        )
    
    # Perform advection
    h_velocities, v_velocities = advect_velocities(h_velocities, v_velocities, agents, dt=dt)
    # Advect concentrations
    agents = advect_concentrations(agents, h_velocities, v_velocities, dt=dt)
    agents = advect_btn_concentrations(agents, h_velocities, v_velocities, dt=dt)
    
    # Update clam populations
    agents = update_clam_population(agents,frame_num)

    #--- ----
    # Calculate velocity statistics to check if dampening is needed
    all_velocities = []
    velocity_magnitudes = []
    
    # Get all velocity components from horizontal edges
    for edge_dict in h_velocities.values():
        vy = edge_dict.get("vy", 0)
        all_velocities.append(vy)
        velocity_magnitudes.append(abs(vy))
        
        if "vx" in edge_dict:
            vx = edge_dict["vx"]
            all_velocities.append(vx)
            velocity_magnitudes.append(abs(vx))
    
    # Get all velocity components from vertical edges
    for edge_dict in v_velocities.values():
        vx = edge_dict.get("vx", 0)
        all_velocities.append(vx)
        velocity_magnitudes.append(abs(vx))
        
        if "vy" in edge_dict:
            vy = edge_dict["vy"]
            all_velocities.append(vy)
            velocity_magnitudes.append(abs(vy))
    
    # Calculate average velocity
    avg_velocity = sum(velocity_magnitudes) / len(velocity_magnitudes) if velocity_magnitudes else 0
    
    # Apply dampening if needed
    threshold = 0.0008
    target_avg = 0.0005
    h_velocities, v_velocities = apply_progressive_dampening(h_velocities, v_velocities, agents,
                                                          avg_velocity, threshold, target_avg) 
    #--- ----

    # Update population statistics
    for i, pop_coords in enumerate(populations):
        pop_name = f"population{i+1}"
        population_stats = {
            "healthy": 0,
            "latent": 0,  # Add latent field
            "infected": 0,
            "dead": 0
        }
        
        # Sum up clam counts for each population
        for coords in pop_coords:
            row, col = coords
            for agent in agents:
                if agent.row == row and agent.col == col:
                    population_stats["healthy"] += agent.healthy_clams
                    population_stats["latent"] += agent.latent_clams  # Add latent clams
                    population_stats["infected"] += agent.infected_clams
                    population_stats["dead"] += agent.dead_clams
                    break
        
        # Update the population data
        population_data[pop_name]["time"].append(frame_num)
        population_data[pop_name]["healthy"].append(population_stats["healthy"])
        population_data[pop_name]["latent"].append(population_stats["latent"])  # Add latent clams
        population_data[pop_name]["infected"].append(population_stats["infected"])
        population_data[pop_name]["dead"].append(population_stats["dead"])
    
    return agents, h_velocities, v_velocities, population_data

def create_animation(
    num_frames=300,           # Total frames to generate in the animation
    steps_per_frame=1,        # Number of simulation steps per frame
    interval=200,             # Delay between frames when viewing (ms)
    population_id=10,
    population_id2=None,
    dt=1e3, 
    projection_loops=10, 
    overrelaxation=1.5,
    drug_drop=(50, 35), 
    drug_drop_frame=50,
    drug_concentration=40,
    btn_zero_frame=30
):
    """
    Create an animation of the clam disease simulation
    
    Parameters:
    -----------
    num_frames : int
        Total number of frames to generate in the final animation
    steps_per_frame : int
        Number of simulation steps to run per visual frame
    interval : int
        Interval between frames in milliseconds
    population_id : int
        ID of the primary population to initialize with disease
    population_id2 : int or None
        ID of the secondary population to initialize with disease (if any)
    dt : float
        Timestep for advection
    projection_loops : int
        Number of projection iterations per step
    overrelaxation : float
        Overrelaxation parameter for pressure projection
    drug_drop : tuple(int, int) or None
        (row, col) position to add drug concentration, or None to skip drug drop
    drug_drop_frame : int
        Frame number at which to drop the concentration (in simulation steps)
    drug_concentration : float
        Amount of drug concentration to add at drug_drop position
    btn_zero_frame : int
        Frame number at which to initialize BTN disease (in simulation steps)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the animation
    ani : matplotlib.animation.FuncAnimation
        Animation object
    """
    # Set up the visualization layout
    fig, axd, agents, h_velocities, v_velocities, population_data = create_visualization_layout(population_id)
    
    # Keep track of total simulation steps
    total_steps = num_frames * steps_per_frame
    
    # Define the update function for animation
    def update_frame(frame_num):
        # Calculate the simulation step range for this frame
        start_step = frame_num * steps_per_frame
        end_step = start_step + steps_per_frame
        
        print(f"Processing frame {frame_num} (simulation steps {start_step} to {end_step-1})")
        
        # Update simulation state
        nonlocal agents, h_velocities, v_velocities, population_data
        
        # Run multiple simulation steps for each frame
        for step in range(start_step, end_step):
            # Pass all parameters to update_simulation
            agents, h_velocities, v_velocities, population_data = update_simulation(
                agents, 
                h_velocities, 
                v_velocities, 
                population_data, 
                step,  # Use actual simulation step number
                dt=dt, 
                projection_loops=projection_loops, 
                overrelaxation=overrelaxation, 
                drug_drop=drug_drop, 
                drug_concentration=drug_concentration,
                drug_drop_frame=drug_drop_frame,
                population_id=population_id,
                population_id2=population_id2,
                btn_zero_frame=btn_zero_frame
            )
        
        # Update concentration plots with dedicated colorbar axes and specific colormaps
        plot_concentration(axd['C'], agents, "concentration", "Drug Concentration (mg/m^3)", cbar_ax=axd['C_cbar'], cmap='viridis')
        plot_concentration(axd['B'], agents, "btn_concentration", "BTN Concentration (cells/m^3)", cbar_ax=axd['B_cbar'], cmap='cividis')
        
        # Update population plots
        for i in range(len(populations)):
            pop_name = f"population{i+1}"
            
            # Get population data (keeping all history for line plots)
            plot_population_data(
                axd[str(i+1)], 
                f"Population {i+1}",
                population_data[pop_name]["time"],
                population_data[pop_name]["healthy"],
                population_data[pop_name]["infected"],
                population_data[pop_name]["dead"],
                population_data[pop_name]["latent"]
            )
        
        # Update figure title to show current simulation step
        fig.suptitle(f"Clam Disease Simulation - Pop {population_id} - Step {end_step-1}", fontsize=16)
        
        # Adjust layout
        fig.tight_layout()
        
    # Create the animation with modified frame count
    ani = animation.FuncAnimation(
        fig, update_frame, frames=num_frames,
        interval=interval, blit=False, repeat=False
    )
    
    return fig, ani

if __name__ == "__main__":
    # Create output directory for animations
    output_dir = Path("output_animations")
    output_dir.mkdir(exist_ok=True)
    
    # List of populations to test as single infection sources
    test_populations = [10]
    
    # Define scenarios - one for each population, no drug treatment
    scenarios = []
    
    for pop_id in test_populations:
        scenarios.append({
            "name": f"pop{pop_id}_only",
            "population_id": pop_id,
            "population_id2": None,  # No secondary infection
            "btn_zero_frame": 1,    # Start disease early
            "drug_drop": None,       # No drug treatment
            "drug_drop_frame": 9999, # Never happens
            "drug_concentration": 0  # No drug
        })
    
    # Common parameters for all simulations
    common_params = {
        "num_frames": 100,       # 150 frames in final animation
        "steps_per_frame": 5,    # Run 5 simulation steps per frame
                                 # Total simulation steps = 150 × 3 = 450
        "interval": 200,
        "dt": 350,
        "projection_loops": 20,
        "overrelaxation": 1.5
    }
    
    # Run each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n===== Running scenario {i+1}/{len(scenarios)}: {scenario['name']} =====")
        print(f"Testing initial infection in population {scenario['population_id']}")
        
        # Combine common parameters with scenario-specific ones
        params = common_params.copy()
        params.update({k: v for k, v in scenario.items() if k != "name"})
        
        # Create and save the animation
        fig, ani = create_animation(**params)
        
        output_file = output_dir / f"no_drug_initial_{scenario['population_id']}.mp4"
        print(f"Saving animation to {output_file}")
        ani.save(str(output_file), writer='ffmpeg', fps=5, dpi=150)
        
        # Clean up to free memory
        plt.close(fig)
        print(f"Completed scenario: {scenario['name']}")
    
    print("\nAll animations completed!")




















