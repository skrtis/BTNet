import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import time
import json
import os
from shapely.geometry import Polygon

# Import methods from grid_generator.py
from grid_generator import (
    FlowPolygonAgent,
    extract_grid_indices,
    transform_indices,
    create_grid_agents,
    assign_edge_velocities
)

# Import methods from mechanisms.py
from mechanisms import (
    advect_velocities,
    project_velocities,
    project_single_cell,
    run_simulation
)

grid_data = extract_grid_indices("malpeque_tiles.geojson")
transformed_data = transform_indices(grid_data)
total_agents = create_grid_agents(transformed_data)
h_velocities, v_velocities = assign_edge_velocities(total_agents)

def visualize_flow(h_velocities, v_velocities, agents, iteration=0):
    """
    Visualize the staggered grid flow field
    
    Parameters:
    -----------
    h_velocities : dict
        Dictionary of horizontal velocities (for horizontal edges)
    v_velocities : dict
        Dictionary of vertical velocities (for vertical edges)
    agents : list
        List of FlowPolygonAgent objects
    iteration : int
        Current iteration number for title display
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create a new figure
    plt.figure(figsize=(14, 10))
    
    # 1. Draw the grid cells (water, non-water, sources)
    for agent in agents:
        row, col = agent.row, agent.col
        
        # Determine cell color based on cell type
        if agent.source:
            color = 'orange'  # Source cells
        elif not agent.water:
            color = 'gray'    # Non-water (land) cells
        elif agent.clam_presence:
            color = 'lightblue'  # Water cells with clams
        else:
            color = 'white'   # Regular water cells
            
        # Draw the cell as a rectangle
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='black', alpha=0.3)
        plt.gca().add_patch(rect)
    
    # 2. Plot velocity vectors
    
    # Plot horizontal velocities (at vertical edges)
    for (row, col), vel in v_velocities.items():
        # Skip locked edges (boundaries)
        if vel["locked"]:
            continue
            
        vx = vel["vx"]
        # Plot as red arrow at vertical edge (col, row+0.5)
        if abs(vx) > 0.01:  # Only plot if velocity is significant
            plt.arrow(col, row+0.5, vx*0.5, 0, 
                     head_width=0.1, head_length=0.05, fc='red', ec='red', 
                     length_includes_head=True)
    
    # Plot vertical velocities (at horizontal edges)
    for (row, col), vel in h_velocities.items():
        # Skip locked edges (boundaries)
        if vel["locked"]:
            continue
            
        vy = vel["vy"]
        # Plot as green arrow at horizontal edge (col+0.5, row)
        if abs(vy) > 0.01:  # Only plot if velocity is significant
            plt.arrow(col+0.5, row, 0, vy*0.5, 
                     head_width=0.1, head_length=0.05, fc='green', ec='green', 
                     length_includes_head=True)
    
    # Add special highlight for source cells and their vectors
    for agent in agents:
        if agent.source:
            row, col = agent.row, agent.col
            plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, 
                                              facecolor='none', 
                                              edgecolor='red', 
                                              linewidth=2))
    
    # Set plot properties
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(0, n_cols)
    plt.ylim(0, n_rows)
    
    # Add legend
    source_patch = plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.3)
    land_patch = plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3)
    water_patch = plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.3)
    clam_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3)
    h_vel_arrow = plt.Line2D([0], [0], color='red', lw=2)
    v_vel_arrow = plt.Line2D([0], [0], color='green', lw=2)
    
    plt.legend([source_patch, land_patch, water_patch, clam_patch, h_vel_arrow, v_vel_arrow],
               ['Source', 'Land', 'Water', 'Clams', 'Horizontal velocity', 'Vertical velocity'],
               loc='upper right')
    
    plt.title(f'Flow Visualization - Iteration {iteration}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Adjust layout to make room for legend
    plt.tight_layout()


run_simulation(h_velocities, v_velocities, total_agents, 
                  num_iterations=1, 
                  advection_loops=1, 
                  projection_loops=1,
                  plot_interval=1, 
                  dt=0.5,
                  visualize_fn=visualize_flow,
                  save_plots=True,
                  output_dir="./output")