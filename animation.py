import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import os

# Import necessary functions from our modules
from grid_generator import (
    extract_grid_indices,
    transform_indices,
    create_grid_agents,
    assign_edge_velocities
)
from mechanisms import (
    advect_velocities, 
    project_velocities,
)
# Import concentration functions from disease.py instead
from disease import (
    drop_concentration,
    advect_concentrations
)
from abm import visualize_concentration

def initialize_simulation():
    """Initialize the simulation environment with grid, agents, and velocities."""
    grid_data = extract_grid_indices("malpeque_tiles.geojson")
    transformed_data = transform_indices(grid_data)
    agents = create_grid_agents(transformed_data)
    h_velocities, v_velocities = assign_edge_velocities(agents)    

    # Initialize drug concentrations to zero
    for agent in agents:
        agent.concentration = 0.0

    return agents, h_velocities, v_velocities

def one_time_step(agents, h_velocities, v_velocities, 
                 iteration=0, 
                 advection_loops=1, 
                 projection_loops=30,
                 overrelaxation=1.6,
                 dt=0.1,
                 drug_drop=(40,40),
                 drug_concentration=100,
                 drug_drop_iteration=10):
    """Execute one time step of the simulation."""
    # Make deep copies to avoid modifying the originals
    current_agents = copy.deepcopy(agents)
    current_h_vel = copy.deepcopy(h_velocities)
    current_v_vel = copy.deepcopy(v_velocities)
    
    # Drop drug at the specified iteration
    if iteration == drug_drop_iteration:
        current_agents = drop_concentration(current_agents, drug_concentration, drug_drop[0], drug_drop[1])
        print(f"Dropping drug at iteration {iteration} at {drug_drop}")
    
    # Perform advection steps
    for _ in range(advection_loops):
        current_h_vel, current_v_vel = advect_velocities(current_h_vel, current_v_vel, current_agents, dt=dt)
    
    # Advect the concentrations
    current_agents = advect_concentrations(current_agents, current_h_vel, current_v_vel, dt=dt)
    
    # Project velocities to ensure incompressibility
    for _ in range(projection_loops):
        project_velocities(current_h_vel, current_v_vel, current_agents, overrelaxation)
    
    # Calculate concentration statistics
    max_concentration = max(agent.concentration for agent in current_agents)
    min_concentration = min(agent.concentration for agent in current_agents)
    total_concentration = sum(agent.concentration for agent in current_agents)
    
    print(f"Iteration {iteration}: Max: {max_concentration:.4f}, Min: {min_concentration:.4f}, Total: {total_concentration:.4f}")
    
    return current_agents, current_h_vel, current_v_vel

def create_concentration_visualization(agents, ax=None, vmin=0, vmax=None):
    """Create a concentration visualization plot on the given axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create concentration grid
    concentration_grid = np.zeros((n_rows, n_cols))
    for agent in agents:
        if hasattr(agent, "concentration"):
            concentration_grid[agent.row, agent.col] = agent.concentration
    
    # Calculate max concentration for dynamic colormap if not provided
    if vmax is None:
        vmax = np.max(concentration_grid)
        if vmax <= 0:
            vmax = 1.0  # Default when no concentration
    
    # Display concentration as heatmap
    im = ax.imshow(concentration_grid, origin='lower', cmap='viridis', 
                  interpolation='nearest', aspect='equal',
                  vmin=vmin, vmax=vmax)
    
    # Mark non-water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                               fill=False, hatch='////', 
                               edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # Configure plot
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(False)
    
    return im, concentration_grid

def update_animation(frame, agents, h_velocities, v_velocities, ax, fig, sim_params, colorbar):
    """Animation update function with updated colormap scaling."""
    # Clear the previous plot
    ax.clear()
    
    # Run one simulation step with offset to avoid double iteration 0
    current_agents, current_h_vel, current_v_vel = one_time_step(
        agents, h_velocities, v_velocities,
        iteration=frame + 1,  # Offset by 1 to avoid duplicate iteration 0
        **sim_params
    )
    
    # Update the original agents with new concentration values
    for i, agent in enumerate(agents):
        agent.concentration = current_agents[i].concentration
    
    # Find the current maximum concentration for dynamic scaling
    max_conc = max(agent.concentration for agent in agents)
    if max_conc <= 0:
        max_conc = 1.0  # Set default when no concentration
    
    # Create visualization with dynamic vmax
    im, _ = create_concentration_visualization(agents, ax=ax, vmin=0, vmax=max_conc)
    
    # Update both the colorbar label AND its scale
    colorbar.set_label(f'Concentration (Max: {max_conc:.2f})')
    colorbar.mappable.set_clim(vmin=0, vmax=max_conc)  # This actually updates the scale
    
    # Set title with frame number
    ax.set_title(f'Concentration Values - Frame {frame+1}')
    
    return im,

def run_animation(num_frames=50, interval=50, fps=10, 
                  save_path='output/concentration_animation.mp4',  # Changed to MP4
                  show_animation=False):
    """Run and save the animation."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize the simulation
    agents, h_velocities, v_velocities = initialize_simulation()
    
    # Simulation parameters
    sim_params = {
        'advection_loops': 1,
        'projection_loops': 50,
        'overrelaxation': 1.6,
        'dt': 0.1,
        'drug_drop': (35, 35),
        'drug_concentration': 100,
        'drug_drop_iteration': 5
    }
    
    # Create figure and axes for animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a placeholder visualization to initialize the colorbar
    agents_copy = copy.deepcopy(agents)
    im, _ = create_concentration_visualization(agents_copy, ax=ax, vmin=0, vmax=sim_params['drug_concentration'])
    
    # Create a single colorbar that will persist throughout the animation
    colorbar = fig.colorbar(im, ax=ax, label='Concentration (Max: 0.00)')
    
    # Create animation
    ani = FuncAnimation(
        fig, 
        update_animation, 
        frames=num_frames,
        fargs=(agents, h_velocities, v_velocities, ax, fig, sim_params, colorbar),
        interval=interval,
        blit=False,
        repeat=False
    )
    
    # Save animation based on file extension
    print(f"Saving animation to {save_path}...")
    try:
        # For MP4 format using ffmpeg
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=100, 
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation as MP4: {e}")
        
        # Fallback to GIF
        gif_path = os.path.splitext(save_path)[0] + '.gif'
        print(f"Attempting to save as GIF instead: {gif_path}")
        try:
            ani.save(gif_path, writer='pillow', fps=fps)
            print("GIF saved successfully!")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
    
    # Show the animation in interactive mode only if requested
    if show_animation:
        plt.show()
    else:
        plt.close(fig)  # Close the figure to free memory
    
    return ani

def run_animation(num_frames=50, interval=50, fps=10, 
                  save_path='output/clam_health_animation.mp4',
                  show_animation=False,
                  infected_population_id=1):
    """Run and save an animation showing clam populations health metrics over time."""
    # Import necessary disease functions
    from disease import update_clam_population, initialize_clam_cancer, populations
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize the simulation
    agents, h_velocities, v_velocities = initialize_simulation()
    
    # Track data for all 23 populations
    population_data = {i+1: {'healthy': [], 'infected': [], 'dead': []} for i in range(23)}
    
    # Initialize clam populations - each agent with clam_presence gets healthy clams
    for agent in agents:
        if hasattr(agent, "clam_presence") and agent.clam_presence:
            agent.healthy_clams = 2500  # Initial healthy clams per cell
            agent.infected_clams = 0
            agent.dead_clams = 0
            agent.btn_concentration = 0.0
    
    # Initialize the infection in specified population
    agents = initialize_clam_cancer(agents, infected_population_id)
    
    # Create figure with 5x5 grid (we'll use 23 of the 25 subplots)
    fig, axs = plt.subplots(5, 5, figsize=(20, 16))
    axs = axs.flatten()  # Flatten for easier indexing
    
    # Hide unused subplots
    for i in range(23, len(axs)):
        fig.delaxes(axs[i])
    
    # Initialize line plots for each population
    lines = {}
    for i in range(23):
        pop_id = i + 1
        ax = axs[i]
        ax.set_title(f"Population {pop_id} (n={len(populations[i])} cells)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Clam Count")
        
        # Create empty line objects
        healthy_line, = ax.plot([], [], 'g-', label='Healthy')
        infected_line, = ax.plot([], [], 'r-', label='Infected')
        dead_line, = ax.plot([], [], 'k-', label='Dead')
        
        # Store reference to lines
        lines[pop_id] = {
            'healthy': healthy_line,
            'infected': infected_line,
            'dead': dead_line
        }
        ax.legend(loc='upper right', fontsize='small')
    
    fig.tight_layout()
    fig.suptitle("Clam Population Health Over Time", fontsize=16)
    
    # Function to update animation
    def update(frame):
        # Update clam populations
        agents = update_clam_population(agents, frame)
        
        # Simulate advection of bacteria concentration
        agents = advect_concentrations(agents, h_velocities, v_velocities, dt=0.1)
        
        # Collect data for each population
        iterations = list(range(frame + 1))
        for pop_id in range(1, 24):
            healthy, infected, dead = 0, 0, 0
            
            # Sum up clam counts for this population
            for coords in populations[pop_id-1]:
                for agent in agents:
                    if agent.row == coords[0] and agent.col == coords[1]:
                        healthy += agent.healthy_clams
                        infected += agent.infected_clams
                        dead += agent.dead_clams
            
            # Store the data
            population_data[pop_id]['healthy'].append(healthy)
            population_data[pop_id]['infected'].append(infected)
            population_data[pop_id]['dead'].append(dead)
            
            # Update the lines
            lines[pop_id]['healthy'].set_data(iterations, population_data[pop_id]['healthy'])
            lines[pop_id]['infected'].set_data(iterations, population_data[pop_id]['infected'])
            lines[pop_id]['dead'].set_data(iterations, population_data[pop_id]['dead'])
            
            # Adjust axes limits
            axs[pop_id-1].set_xlim(0, max(10, frame + 1))
            axs[pop_id-1].set_ylim(0, max(100, healthy + infected + dead + 100))
        
        # Return all line objects
        return [line for pop_lines in lines.values() for line in pop_lines.values()]
    
    # Create animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=interval,
        blit=True
    )
    
    # Save animation
    print(f"Saving animation to {save_path}...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=100, 
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation as MP4: {e}")
        gif_path = os.path.splitext(save_path)[0] + '.gif'
        try:
            ani.save(gif_path, writer='pillow', fps=fps)
            print("GIF saved successfully!")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
    
    if show_animation:
        plt.show()
    else:
        plt.close(fig)
    
    return ani

if __name__ == "__main__":
    run_animation(
        num_frames=100,
        interval=200, 
        fps=10,
        save_path='output/clam_health_animation.mp4',
        show_animation=False,
        infected_population_id=2  # Start infection in population 2
    )