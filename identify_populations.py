import numpy as np
from collections import deque
import os
import random
import matplotlib.pyplot as plt

def identify_populations(agents):
    """
    Identify connected populations of clams.
    
    A population is defined as a group of clam agents that are connected
    by edges, vertices, or through other clam agents in the population.
    
    Parameters:
    -----------
    agents : list
        List of agent objects with row and col attributes
        
    Returns:
    --------
    dict
        Dictionary mapping population names to lists of (row, col) coordinates
    """
    # Create a list of all clam coordinates for quick lookup
    clam_coords = []
    for agent in agents:
        # Look for clam_presence=True instead of water=True
        if hasattr(agent, "clam_presence") and agent.clam_presence:
            clam_coords.append((agent.row, agent.col))
    
    clam_coords_set = set(clam_coords)
    
    # Define the 8 directions (horizontal, vertical, and diagonal connections)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Perform breadth-first search to identify connected components
    visited = set()
    populations = []
    
    for row, col in clam_coords:
        if (row, col) in visited:
            continue
        
        # Start a new population
        population = []
        queue = deque([(row, col)])
        visited.add((row, col))
        
        # BFS to find all connected clams
        while queue:
            r, c = queue.popleft()
            population.append((r, c))
            
            # Check all 8 directions for neighbors (including diagonals)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (nr, nc) in clam_coords_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        populations.append(population)
    
    # Create named dictionary of populations
    named_populations = {}
    for i, pop in enumerate(populations):
        named_populations[f"population{i+1}"] = pop
    
    print(f"Found {len(populations)} distinct clam populations")
    
    return named_populations

def visualize_populations(agents, populations):
    """
    Visualize the identified populations with different colors,
    showing the underlying map (water/land) for reference.
    
    Parameters:
    -----------
    agents : list
        List of agent objects
    populations : dict
        Dictionary mapping population names to lists of (row, col) coordinates
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create a base grid showing water/land
    base_grid = np.zeros((n_rows, n_cols))
    for agent in agents:
        if hasattr(agent, "water") and agent.water:
            base_grid[agent.row, agent.col] = 1
    
    # Create a grid to store population IDs
    pop_grid = np.zeros((n_rows, n_cols))
    
    # Assign a unique ID to each population
    for i, (name, pop) in enumerate(populations.items()):
        for r, c in pop:
            pop_grid[r, c] = i + 1  # Add 1 to avoid confusion with background (0)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Base map with water and land - origin='lower' puts (0,0) at bottom left
    water_cmap = plt.cm.colors.ListedColormap(['tan', 'lightblue'])
    ax1.imshow(base_grid, cmap=water_cmap, origin='lower')
    ax1.set_title('Base Map (Blue = Water, Tan = Land)')
    
    # Add grid lines to the base map
    for i in range(n_cols + 1):
        ax1.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    for i in range(n_rows + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Plot 2: Population colors
    # Create color map with random colors for each population
    n_pops = len(populations)
    colors = ['white']  # Background color
    
    # Use a fixed seed for reproducible colors
    random.seed(42)
    for _ in range(n_pops):
        # Generate semi-transparent colors
        r, g, b = random.random(), random.random(), random.random()
        colors.append((r, g, b, 0.7))  # Add alpha channel for transparency
    
    # Create a masked array to only color population cells
    masked_pop_grid = np.ma.masked_where(pop_grid == 0, pop_grid)
    
    # Plot the base map first - with origin='lower'
    ax2.imshow(base_grid, cmap=water_cmap, origin='lower')
    
    # Then overlay the populations with custom colors - with origin='lower'
    pop_cmap = plt.cm.colors.ListedColormap(colors[1:])  # Skip the background color
    population_plot = ax2.imshow(masked_pop_grid, cmap=pop_cmap, vmin=1, vmax=n_pops+1, origin='lower')
    
    # Add grid lines
    for i in range(n_cols + 1):
        ax2.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    for i in range(n_rows + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Add population labels for larger populations
    for name, pop in populations.items():
        if len(pop) > 0:  # Only label populations with more than 3 cells
            center_r = sum(r for r, c in pop) / len(pop)
            center_c = sum(c for r, c in pop) / len(pop)
            ax2.text(center_c, center_r, name, 
                   ha='center', va='center', fontweight='bold', color='black')
    
    # Add colorbar
    cbar = fig.colorbar(population_plot, ax=ax2)
    cbar.set_label('Population ID')
    
    # Set titles and labels
    ax2.set_title(f'Clam Populations ({n_pops} total)')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('output/clam_populations_with_map.png', dpi=300)
    plt.show()
    
    return pop_grid

def get_population_statistics(populations):
    """
    Get statistics about the identified populations.
    
    Parameters:
    -----------
    populations : dict
        Dictionary mapping population names to lists of (row, col) coordinates
    
    Returns:
    --------
    dict
        Dictionary with population statistics
    """
    stats = {
        'total_populations': len(populations),
        'total_clams': sum(len(pop) for pop in populations.values()),
        'population_sizes': {name: len(pop) for name, pop in populations.items()},
        'largest_population': max(populations.items(), key=lambda x: len(x[1]))[0],
        'smallest_population': min(populations.items(), key=lambda x: len(x[1]))[0]
    }
    
    return stats

def main():
    """Main function to demonstrate population identification."""
    from grid_generator import extract_grid_indices, transform_indices, create_grid_agents
    
    # Load grid data
    grid_data = extract_grid_indices("malpeque_tiles.geojson")
    transformed_data = transform_indices(grid_data)
    agents = create_grid_agents(transformed_data)
    
    # If agents don't have clam_presence attribute, add it for testing
    # In production, this should come from your actual clam distribution
    if not hasattr(agents[0], "clam_presence"):
        print("Adding dummy clam_presence attribute for testing")
        for agent in agents:
            if hasattr(agent, "water") and agent.water:
                # For testing, just copy the water attribute
                agent.clam_presence = True
            else:
                agent.clam_presence = False
    
    # Identify populations
    populations = identify_populations(agents)
    
    # Get population statistics
    stats = get_population_statistics(populations)
    print(f"Found {stats['total_populations']} populations with {stats['total_clams']} total clams")
    print(f"Largest population: {stats['largest_population']} with {stats['population_sizes'][stats['largest_population']]} clams")
    print(f"Smallest population: {stats['smallest_population']} with {stats['population_sizes'][stats['smallest_population']]} clams")
    
    # Visualize populations
    visualize_populations(agents, populations)

    print(populations) 
    return populations

if __name__ == "__main__":
    main()