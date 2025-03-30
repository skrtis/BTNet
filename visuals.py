import solara
import matplotlib.pyplot as plt
import copy

# Import only necessary grid functions
from grid_generator import (
    extract_grid_indices,
    transform_indices,
    create_grid_agents,
    assign_edge_velocities
)

# Import the correct visualization function
from abm import visualize_concentration

# Import mechanisms for simulation steps
from mechanisms import advect_velocities, project_velocities

# Initialize data directly
raw_grid_data = extract_grid_indices("malpeque_tiles.geojson")
transformed_data = transform_indices(raw_grid_data)
agents = create_grid_agents(transformed_data)
h_velocities, v_velocities = assign_edge_velocities(agents)

# Store in a dictionary for easy access and create reactive state
current_iteration = solara.reactive(0)
grid_data = solara.reactive({
    "agents": agents,
    "h_velocities": h_velocities,
    "v_velocities": v_velocities
})

def simulation_step():
    """Advance the simulation by one step"""
    # Make deep copies of the current state
    current_agents = copy.deepcopy(grid_data.value["agents"])
    current_h_vel = copy.deepcopy(grid_data.value["h_velocities"])
    current_v_vel = copy.deepcopy(grid_data.value["v_velocities"])
    
    # Advance simulation by one timestep
    # Parameters based on the commented run_simulation in abm.py
    dt = 1.0
    overrelaxation = 1.0
    advection_loops = 1
    projection_loops = 10  # Make sure this is an integer
    
    # Apply advection
    advect_velocities(current_h_vel, current_v_vel, current_agents, dt)
    
    # Apply projection - explicitly cast to int to avoid type errors
    project_velocities(current_h_vel, current_v_vel, current_agents, overrelaxation, projection_loops)
    
    # Update the grid data with new state
    grid_data.value = {
        "agents": current_agents,
        "h_velocities": current_h_vel,
        "v_velocities": current_v_vel
    }
    
    # Increment iteration counter
    current_iteration.value += 1

def run_multiple_steps(steps=10):
    """Run the simulation for multiple steps"""
    for _ in range(steps):
        simulation_step()

@solara.component
def StaggeredVelocitiesPlot():
    """Component to display the staggered velocities plot"""
    # Create and return the visualization
    fig = visualize_concentration(
        grid_data.value["h_velocities"], 
        grid_data.value["v_velocities"], 
        grid_data.value["agents"], 
        current_iteration.value
    )
    
    # Return the figure as a Solara component
    return solara.FigureMatplotlib(fig)

@solara.component
def SimulationControls():
    """Controls for the simulation"""
    with solara.Card("Simulation Controls"):
        with solara.Column():
            solara.Text(f"Current iteration: {current_iteration.value}")
            
            # Single step button
            solara.Button(
                label="Step Once",
                icon_name="mdi-step-forward",
                on_click=simulation_step  # Note: no parentheses here
            )
            
            # Multiple step button 
            solara.Button(
                label="Run 100 Steps",
                icon_name="mdi-fast-forward",
                on_click=lambda: run_multiple_steps(100)  # Using lambda to pass parameters
            )

@solara.component
def Page():
    """Main page layout"""
    with solara.Column():
        solara.Title("Flow Model Visualization")
        
        # Add simulation controls
        SimulationControls()
        
        # Add the plot component
        StaggeredVelocitiesPlot()