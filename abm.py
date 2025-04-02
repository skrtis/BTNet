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

from disease import (
    populations
)


grid_data = extract_grid_indices("malpeque_tiles.geojson")
transformed_data = transform_indices(grid_data)
total_agents = create_grid_agents(transformed_data)
h_velocities, v_velocities = assign_edge_velocities(total_agents)

def visualize_collocated(h_velocities, v_velocities, agents, iteration=0):
   # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    # 2) SECOND SUBPLOT: Cell-centered velocity vectors
    ax = axs[1]
    
    # First draw the same grid background for reference
    for agent in agents:
        row, col = agent.row, agent.col
        
        # Determine cell color based on cell type
        if agent.source:
            color = 'orange'
        elif not agent.water:
            color = 'gray'
        elif agent.clam_presence:
            color = 'lightblue'
        else:
            color = 'white'
            
        # Draw the cell as a rectangle
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
    
    # Calculate and plot cell-centered velocities for water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            continue
        
        row, col = agent.row, agent.col
        
        # Calculate average x-velocity from adjacent vertical edges
        vx = 0.0
        count_x = 0
        if agent.velocity_w:
            vx += agent.velocity_w["vx"]
            count_x += 1
        if agent.velocity_e:
            vx += agent.velocity_e["vx"]
            count_x += 1
        if count_x > 0:
            vx /= count_x
        
        # Calculate average y-velocity from adjacent horizontal edges
        vy = 0.0
        count_y = 0
        if agent.velocity_n:
            vy += agent.velocity_n["vy"]
            count_y += 1
        if agent.velocity_s:
            vy += agent.velocity_s["vy"]
            count_y += 1
        if count_y > 0:
            vy /= count_y
        
        # Plot the cell-centered velocity vector
        if (abs(vx) > 1e-10 or abs(vy) > 1e-10):
            # Cell center coordinates
            center_x = col + 0.5
            center_y = row + 0.5
            
            # Scale factor for specific vector visibility
            scale =1e3
            
            # Plot the arrow
            ax.arrow(center_x, center_y, vx * scale, vy * scale,
                    head_width=0.1, head_length=0.05, fc='blue', ec='blue',
                    length_includes_head=True)
    
    # Configure second subplot
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_title('Cell-Centered Velocity Vectors')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add simple legend to second subplot
    center_vel_arrow = plt.Line2D([0], [0], color='blue', lw=2)
    ax.legend([center_vel_arrow], ['Cell-centered velocity'], 
              loc='upper right', fontsize='small')

    # Overall figure configuration
    plt.suptitle(f'Flow and Concentration Visualization - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    return fig 

def visualize_flow(h_velocities, v_velocities, agents, iteration=0, vmin=None, vmax=None):
    """
    Visualize the flow field with three panels:
    1) Original staggered grid visualization
    2) Cell-centered velocity vectors
    3) Concentration values
    
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
    vmin : float, optional
        Minimum value for concentration colormap
    vmax : float, optional
        Maximum value for concentration colormap
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    
    # 1) FIRST SUBPLOT: Original staggered grid visualization
    ax = axs[0]
    
    # Draw the grid cells (water, non-water, sources)
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
        ax.add_patch(rect)
    
    # Plot horizontal velocities (at vertical edges)
    for (row, col), vel in v_velocities.items():
        # Skip locked edges (boundaries)
        if vel["locked"]:
            continue
            
        vx = vel["vx"]
        # Plot as red arrow at vertical edge (col, row+0.5)
        if abs(vx) > 1e-10:  # Only plot if velocity is significant
            ax.arrow(col, row+0.5, vx*0.5, 0, 
                     head_width=0.1, head_length=0.05, fc='red', ec='red', 
                     length_includes_head=True)
    
    # Plot vertical velocities (at horizontal edges)
    for (row, col), vel in h_velocities.items():
        # Skip locked edges (boundaries)
        if vel["locked"]:
            continue
            
        vy = vel["vy"]
        # Plot as green arrow at horizontal edge (col+0.5, row)
        if abs(vy) > 1e-10:  # Only plot if velocity is significant
            ax.arrow(col+0.5, row, 0, vy*0.5, 
                     head_width=0.1, head_length=0.05, fc='green', ec='green', 
                     length_includes_head=True)
    
    # Add special highlight for source cells
    for agent in agents:
        if agent.source:
            row, col = agent.row, agent.col
            ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                       facecolor='none', 
                                       edgecolor='red', 
                                       linewidth=2))
    
    # Configure first subplot
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_title('Staggered Grid Flow Visualization')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add legend to first subplot
    source_patch = plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.3)
    land_patch = plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3)
    water_patch = plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.3)
    clam_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3)
    h_vel_arrow = plt.Line2D([0], [0], color='red', lw=2)
    v_vel_arrow = plt.Line2D([0], [0], color='green', lw=2)
    
    ax.legend([source_patch, land_patch, water_patch, clam_patch, h_vel_arrow, v_vel_arrow],
              ['Source', 'Land', 'Water', 'Clams', 'Horizontal velocity', 'Vertical velocity'],
              loc='upper right', fontsize='small')

    # 2) SECOND SUBPLOT: Cell-centered velocity vectors
    ax = axs[1]
    
    # First draw the same grid background for reference
    for agent in agents:
        row, col = agent.row, agent.col
        
        # Determine cell color based on cell type
        if agent.source:
            color = 'orange'
        elif not agent.water:
            color = 'gray'
        elif agent.clam_presence:
            color = 'lightblue'
        else:
            color = 'white'
            
        # Draw the cell as a rectangle
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
    
    # Calculate and plot cell-centered velocities for water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            continue
        
        row, col = agent.row, agent.col
        
        # Calculate average x-velocity from adjacent vertical edges
        vx = 0.0
        count_x = 0
        if agent.velocity_w:
            vx += agent.velocity_w["vx"]
            count_x += 1
        if agent.velocity_e:
            vx += agent.velocity_e["vx"]
            count_x += 1
        if count_x > 0:
            vx /= count_x
        
        # Calculate average y-velocity from adjacent horizontal edges
        vy = 0.0
        count_y = 0
        if agent.velocity_n:
            vy += agent.velocity_n["vy"]
            count_y += 1
        if agent.velocity_s:
            vy += agent.velocity_s["vy"]
            count_y += 1
        if count_y > 0:
            vy /= count_y
        
        # Plot the cell-centered velocity vector
        if (abs(vx) > 1e-10 or abs(vy) > 1e-10):
            # Cell center coordinates
            center_x = col + 0.5
            center_y = row + 0.5
            
            # Scale factor for specific vector visibility
            scale = 1
            
            # Plot the arrow
            ax.arrow(center_x, center_y, vx * scale, vy * scale,
                    head_width=0.1, head_length=0.05, fc='blue', ec='blue',
                    length_includes_head=True)
    
    # Configure second subplot
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_title('Cell-Centered Velocity Vectors')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add simple legend to second subplot
    center_vel_arrow = plt.Line2D([0], [0], color='blue', lw=2)
    ax.legend([center_vel_arrow], ['Cell-centered velocity'], 
              loc='upper right', fontsize='small')

    # 3) THIRD SUBPLOT: Concentration values
    ax = axs[2]
    
    # Create a grid for concentrations
    concentration_grid = np.zeros((n_rows, n_cols))
    
    # Fill the grid with concentration values
    for agent in agents:
        if hasattr(agent, "concentration"):
            concentration_grid[agent.row, agent.col] = agent.concentration
    
    # Use imshow to display concentration as a heatmap with specified range
    im = ax.imshow(concentration_grid, origin='lower', cmap='viridis', 
                  interpolation='nearest', aspect='equal',
                  extent=[0, n_cols, 0, n_rows],
                  vmin=vmin, vmax=vmax)  # Set min and max values for color scale
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Concentration')
    
    # Add cell outlines for reference (optional)
    for i in range(n_cols + 1):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(n_rows + 1):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
    
    # Mark non-water cells with crosshatching
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                                fill=False, hatch='////', 
                                edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # Configure third subplot
    ax.set_title('Concentration Values')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(False)
    
    # Overall figure configuration
    plt.suptitle(f'Flow and Concentration Visualization - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_concentration(h_velocities, v_velocities, agents, iteration=0, vmin=0, vmax=None):

    """
    Visualize only the concentration values as a heatmap.
    
    Parameters:
    -----------
    h_velocities : dict
        Dictionary of horizontal velocities (not used but kept for compatibility)
    v_velocities : dict
        Dictionary of vertical velocities (not used but kept for compatibility)
    agents : list
        List of FlowPolygonAgent objects
    iteration : int
        Current iteration number for title display
    vmin : float, optional
        Minimum value for concentration colormap
    vmax : float, optional
        Maximum value for concentration colormap
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid for concentrations
    concentration_grid = np.zeros((n_rows, n_cols))
    
    # Fill the grid with concentration values
    for agent in agents:
        if hasattr(agent, "concentration"):
            concentration_grid[agent.row, agent.col] = agent.concentration
    
    # Use imshow to display concentration as a heatmap with specified range
    im = ax.imshow(concentration_grid, origin='lower', cmap='viridis', 
                  interpolation='nearest', aspect='equal',
                  extent=[0, n_cols, 0, n_rows],
                  vmin=0, vmax=vmax)  # Set min and max values for color scale
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Concentration')
    
    # Add cell outlines for reference (optional)
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
    ax.set_title('Concentration Values')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(False)
    
    # Overall figure configuration
    plt.suptitle(f'Concentration Visualization - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_btn(h_velocities, v_velocities, agents, iteration=0, vmin=0, vmax=None):
    """
    Visualize only the BTN concentration values as a heatmap.
    
    Parameters:
    -----------
    h_velocities : dict
        Dictionary of horizontal velocities (not used but kept for compatibility)
    v_velocities : dict
        Dictionary of vertical velocities (not used but kept for compatibility)
    agents : list
        List of FlowPolygonAgent objects
    iteration : int
        Current iteration number for title display
    vmin : float, optional
        Minimum value for concentration colormap
    vmax : float, optional
        Maximum value for concentration colormap
    """
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid for concentrations
    btn_concentration_grid = np.zeros((n_rows, n_cols))
    
    # Debug: Count how many agents have btn_concentration
    has_btn = 0
    max_btn = 0
    
    # Fill the grid with concentration values
    for agent in agents:
        if hasattr(agent, "btn_concentration"):
            btn_concentration_grid[agent.row, agent.col] = agent.btn_concentration
            has_btn += 1
            max_btn = max(max_btn, agent.btn_concentration)
    
    # Print debug info
    print(f"Iteration {iteration}: {has_btn} agents have btn_concentration, max value: {max_btn}")
    
    # If no maximum specified and we have data, set a reasonable maximum
    if vmax is None and max_btn > 0:
        vmax = max_btn
    
    # Use imshow to display concentration as a heatmap with specified range
    im = ax.imshow(btn_concentration_grid, origin='lower', cmap='viridis', 
                  interpolation='nearest', aspect='equal',
                  extent=[0, n_cols, 0, n_rows],
                  vmin=0, vmax=vmax)  # Set min and max values for color scale
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=f'BTN Concentration (Max: {max_btn:.2f})')
    
    # Add cell outlines for reference (optional)
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
    
    # Mark clam cells with a different border or marker
    
    # Configure plot
    ax.set_title('BTN Concentration Values (mg/L)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(False)
    
    # Overall figure configuration
    plt.suptitle(f'BTN Concentration Visualization - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_everything(h_velocities, v_velocities, agents, iteration=0):
    """
    Comprehensive visualization including BTN concentration, drug concentration,
    and population statistics - with larger concentration plots and smaller population graph.
    
    Parameters:
    -----------
    h_velocities : dict
        Dictionary of horizontal velocities
    v_velocities : dict
        Dictionary of vertical velocities
    agents : list
        List of FlowPolygonAgent objects
    iteration : int
        Current iteration number for title display
    """
    # Check if we have population data
    has_clams = any(hasattr(agent, 'clam_presence') and agent.clam_presence for agent in agents)
    
    # Create figure with appropriate layout
    fig = plt.figure(figsize=(18, 14))
    
    if has_clams:
        # Define the subplot layout using GridSpec with adjusted heights
        # Make the concentration plots take up more space (rows 0-2)
        # Make the population plot smaller (row 3 only)
        gs = fig.add_gridspec(4, 3, height_ratios=[3, 3, 3, 2])
        
        # Main visualization panels - now spanning multiple rows
        ax_btn = fig.add_subplot(gs[0:2, 0])  # Spans rows 0-1, column 0
        ax_drug = fig.add_subplot(gs[0:2, 1])  # Spans rows 0-1, column 1
        ax_flow = fig.add_subplot(gs[0:2, 2])  # Spans rows 0-1, column 2
        
        # Population statistics panels - smaller and at bottom
        pop_stats_ax = fig.add_subplot(gs[3, :])
    else:
        # Simpler layout if no clams
        gs = fig.add_gridspec(1, 3)
        ax_btn = fig.add_subplot(gs[0, 0])
        ax_drug = fig.add_subplot(gs[0, 1])
        ax_flow = fig.add_subplot(gs[0, 2])
    
    # Get grid dimensions
    n_rows = max(agent.row for agent in agents) + 1
    n_cols = max(agent.col for agent in agents) + 1
    
    # --- BTN Concentration Plot ---
    btn_concentration_grid = np.zeros((n_rows, n_cols))
    max_btn = 0
    
    for agent in agents:
        if hasattr(agent, "btn_concentration"):
            btn_concentration_grid[agent.row, agent.col] = agent.btn_concentration
            max_btn = max(max_btn, agent.btn_concentration)
    
    im_btn = ax_btn.imshow(btn_concentration_grid, origin='lower', cmap='inferno', 
                  interpolation='nearest', aspect='equal',
                  extent=[0, n_cols, 0, n_rows],
                  vmin=0)
    
    # Add colorbar
    plt.colorbar(im_btn, ax=ax_btn, label=f'BTN Concentration (max: {max_btn:.8f})')
    
    # Mark non-water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                               fill=False, hatch='x', 
                               edgecolor='white', alpha=0.7)
            ax_btn.add_patch(rect)
    
    # Mark clam cells with cyan border
    for agent in agents:
        if hasattr(agent, "clam_presence") and agent.clam_presence:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                               fill=False, linewidth=1.5, 
                               edgecolor='cyan', alpha=0.8)
            ax_btn.add_patch(rect)
    
    ax_btn.set_title('BTN Concentration', fontsize=14)
    ax_btn.set_xlabel('Column')
    ax_btn.set_ylabel('Row')
    
    # --- Drug Concentration Plot ---
    drug_concentration_grid = np.zeros((n_rows, n_cols))
    max_drug = 0
    
    for agent in agents:
        if hasattr(agent, "concentration"):
            drug_concentration_grid[agent.row, agent.col] = agent.concentration
            max_drug = max(max_drug, agent.concentration)
    
    im_drug = ax_drug.imshow(drug_concentration_grid, origin='lower', cmap='viridis', 
                   interpolation='nearest', aspect='equal',
                   extent=[0, n_cols, 0, n_rows],
                   vmin=0)
    
    # Add colorbar
    plt.colorbar(im_drug, ax=ax_drug, label=f'Drug Concentration (max: {max_drug:.2f})')
    
    # Mark non-water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            row, col = agent.row, agent.col
            rect = plt.Rectangle((col, row), 1, 1, 
                               fill=False, hatch='x', 
                               edgecolor='black', alpha=0.7)
            ax_drug.add_patch(rect)
    
    ax_drug.set_title('Drug Concentration', fontsize=14)
    ax_drug.set_xlabel('Column')
    ax_drug.set_ylabel('Row')
    
    # --- Flow Visualization ---
    # Draw the grid cells
    for agent in agents:
        row, col = agent.row, agent.col
        
        # Determine cell color based on cell type
        if hasattr(agent, "source") and agent.source:
            color = 'orange'  # Source cells
        elif not agent.water:
            color = 'gray'    # Non-water (land) cells
        elif hasattr(agent, "clam_presence") and agent.clam_presence:
            color = 'lightblue'  # Water cells with clams
        else:
            color = 'white'   # Regular water cells
            
        # Draw the cell
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='black', alpha=0.3)
        ax_flow.add_patch(rect)
    
    # Calculate and plot cell-centered velocities for water cells
    for agent in agents:
        if not hasattr(agent, "water") or not agent.water:
            continue
        
        row, col = agent.row, agent.col
        
        # Calculate average x-velocity
        vx = 0.0
        count_x = 0
        if hasattr(agent, "velocity_w") and agent.velocity_w:
            vx += agent.velocity_w["vx"]
            count_x += 1
        if hasattr(agent, "velocity_e") and agent.velocity_e:
            vx += agent.velocity_e["vx"]
            count_x += 1
        if count_x > 0:
            vx /= count_x
        
        # Calculate average y-velocity
        vy = 0.0
        count_y = 0
        if hasattr(agent, "velocity_n") and agent.velocity_n:
            vy += agent.velocity_n["vy"]
            count_y += 1
        if hasattr(agent, "velocity_s") and agent.velocity_s:
            vy += agent.velocity_s["vy"]
            count_y += 1
        if count_y > 0:
            vy /= count_y
        
        # Plot velocity vector if significant
        if (abs(vx) > 1e-10 or abs(vy) > 1e-10):
            center_x = col + 0.5
            center_y = row + 0.5
            
            # Scale for visibility
            scale = 1e2
            
            ax_flow.arrow(center_x, center_y, vx * scale, vy * scale,
                      head_width=0.1, head_length=0.05, fc='blue', ec='blue',
                      length_includes_head=True)
    
    ax_flow.set_title('Water Flow', fontsize=14)
    ax_flow.set_xlabel('Column')
    ax_flow.set_ylabel('Row')
    ax_flow.set_xlim(0, n_cols)
    ax_flow.set_ylim(0, n_rows)
    
    # --- Population Statistics (if clams are present) ---
    if has_clams:
        # Collect population statistics
        population_stats = []
        total_healthy = 0
        total_infected = 0
        total_dead = 0
        
        for i, pop_coords in enumerate(populations):
            pop_stats = {"id": i+1, "healthy": 0, "infected": 0, "dead": 0}
            
            for coords in pop_coords:
                row, col = coords
                for agent in agents:
                    if agent.row == row and agent.col == col:
                        pop_stats["healthy"] += agent.healthy_clams if hasattr(agent, "healthy_clams") else 0
                        pop_stats["infected"] += agent.infected_clams if hasattr(agent, "infected_clams") else 0
                        pop_stats["dead"] += agent.dead_clams if hasattr(agent, "dead_clams") else 0
                        break
            
            total_healthy += pop_stats["healthy"]
            total_infected += pop_stats["infected"]
            total_dead += pop_stats["dead"]
            
            if pop_stats["healthy"] + pop_stats["infected"] + pop_stats["dead"] > 0:
                population_stats.append(pop_stats)
        
        # Create bar chart for healthy/infected/dead counts
        pop_ids = [f"Pop {stats['id']}" for stats in population_stats]
        healthy_counts = [stats["healthy"] for stats in population_stats]
        infected_counts = [stats["infected"] for stats in population_stats]
        dead_counts = [stats["dead"] for stats in population_stats]
        
        x = np.arange(len(pop_ids))
        width = 0.25
        
        pop_stats_ax.bar(x - width, healthy_counts, width, label='Healthy', color='green')
        pop_stats_ax.bar(x, infected_counts, width, label='Infected', color='red')
        pop_stats_ax.bar(x + width, dead_counts, width, label='Dead', color='black')
        
        pop_stats_ax.set_title('Clam Population Statistics', fontsize=14)
        pop_stats_ax.set_xlabel('Population')
        pop_stats_ax.set_ylabel('Count')
        pop_stats_ax.set_xticks(x)
        pop_stats_ax.set_xticklabels(pop_ids)
        pop_stats_ax.legend()
        
        # Add summary text
        summary_text = f"Total: {total_healthy} Healthy, {total_infected} Infected, {total_dead} Dead"
        pop_stats_ax.text(0.5, -0.1, summary_text, transform=pop_stats_ax.transAxes, 
                         horizontalalignment='center', fontsize=12)
    
    # Overall figure configuration
    plt.suptitle(f'Simulation Status - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    return fig

run_simulation(h_velocities, v_velocities, total_agents, 
               num_iterations=1000, 
               advection_loops=1, 
               projection_loops=30,
               plot_interval=50,  # More frequent plots to better see the disease progression
               overrelaxation=1.5,
               dt=350,
               drug_drop=(45,45),
               drug_concentration=10,
               drug_drop_iteration=5,
               visualize_fn=visualize_collocated,  # Use our new visualization function
               save_plots=True,
               output_dir="./output")


