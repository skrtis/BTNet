import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import math

#############################################
# STEP 1: Create a Simple Grid Structure
#############################################
class Cell:
    def __init__(self, row_index, col_index):
        self.row_index = row_index
        self.col_index = col_index
        self.velocity_n = None  # North edge
        self.velocity_e = None  # East edge
        self.velocity_s = None  # South edge
        self.velocity_w = None  # West edge
        self.source = False     # If True, this cell is a source
        self.bounds = (col_index, row_index, col_index+1, row_index+1)  # minx, miny, maxx, maxy

    @property
    def geometry(self):
        """Simple geometry representation for visualization"""
        return {
            'bounds': self.bounds,
            'centroid': ((self.bounds[0] + self.bounds[2])/2, (self.bounds[1] + self.bounds[3])/2)
        }

def create_grid(rows, cols):
    """Create a rectangular grid of cells"""
    cells = []
    for i in range(rows):
        for j in range(cols):
            cells.append(Cell(i, j))
    print(f"Created grid with {rows}x{cols} = {len(cells)} cells")
    return cells

#############################################
# STEP 2: Assign Shared Edges
#############################################
def assign_shared_edges(cells, rows, cols):
    """
    For each cell, assign canonical keys for its edges.
    For vertical edges:
         West edge: key = ("V", row_index, col_index)
         East edge: key = ("V", row_index, col_index+1)
    For horizontal edges:
         North edge: key = ("H", row_index, col_index)
         South edge: key = ("H", row_index+1, col_index)
    """
    # Build a mapping from (row_index, col_index) to cell
    cell_map = {}
    for cell in cells:
        cell_map[(cell.row_index, cell.col_index)] = cell

    unique_edges = {}
    for cell in cells:
        minx, miny, maxx, maxy = cell.bounds

        # WEST edge
        key_w = ("V", cell.row_index, cell.col_index)
        locked_w = ((cell.row_index, cell.col_index - 1) not in cell_map)
        if key_w not in unique_edges:
            unique_edges[key_w] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_w,
                "midpoint": (minx, (miny+maxy)/2)
            }
        cell.velocity_w = unique_edges[key_w]

        # EAST edge
        key_e = ("V", cell.row_index, cell.col_index+1)
        locked_e = ((cell.row_index, cell.col_index+1) not in cell_map)
        if key_e not in unique_edges:
            unique_edges[key_e] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_e,
                "midpoint": (maxx, (miny+maxy)/2)
            }
        cell.velocity_e = unique_edges[key_e]

        # NORTH edge
        key_n = ("H", cell.row_index, cell.col_index)
        locked_n = ((cell.row_index-1, cell.col_index) not in cell_map)
        if key_n not in unique_edges:
            unique_edges[key_n] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_n,
                "midpoint": ((minx+maxx)/2, maxy)
            }
        cell.velocity_n = unique_edges[key_n]

        # SOUTH edge
        key_s = ("H", cell.row_index+1, cell.col_index)
        locked_s = ((cell.row_index+1, cell.col_index) not in cell_map)
        if key_s not in unique_edges:
            unique_edges[key_s] = {
                "vx": 0.0,
                "vy": 0.0,
                "locked": locked_s,
                "midpoint": ((minx+maxx)/2, miny)
            }
        cell.velocity_s = unique_edges[key_s]

    print(f"Assigned {len(unique_edges)} unique edges")
    return unique_edges

#############################################
# STEP 3: Initialize Source Vectors
#############################################
def initialize_source_vectors(cells, global_edges):
    """
    Set source vectors on the right boundary of the grid (x=100).
    This creates westward flow from the right boundary edge.
    """
    source_cells = []
    count = 0
    
    # Get cells on the right edge (col_index = 99)
    right_edge_cells = [cell for cell in cells if cell.col_index == 99]
    
    for cell in right_edge_cells:
        cell.source = True
        source_cells.append(cell)
        
        # Set the East edge (boundary edge) to have a westward velocity
        if cell.velocity_e:
            cell.velocity_e["vx"] = -1.0  # Flowing to the left (westward)
            cell.velocity_e["locked"] = True
            count += 1
    
    print(f"Initialized {count} source edges on {len(source_cells)} source cells at the right boundary")
    return source_cells

#############################################
# STEP 4: Incompressibility Functions
#############################################
def classify_cells(cells):
    """Classify cells into red/black/source for Gauss-Seidel relaxation"""
    groups = {"red": [], "black": [], "source": []}
    for cell in cells:
        if cell.source:
            groups["source"].append(cell)
        else:
            if (cell.row_index + cell.col_index) % 2 == 0:
                groups["red"].append(cell)
            else:
                groups["black"].append(cell)
    return groups

def do_incompressibility_iteration(cells):
    """One step of incompressibility projection"""
    for cell in cells:
        if cell.source:
            continue
        
        # Calculate fluxes
        flux_n = cell.velocity_n["vy"] if (cell.velocity_n and not isinstance(cell.velocity_n["vy"], str)) else 0.0
        flux_e = cell.velocity_e["vx"] if (cell.velocity_e and not isinstance(cell.velocity_e["vx"], str)) else 0.0
        flux_w = cell.velocity_w["vx"] if (cell.velocity_w and not isinstance(cell.velocity_w["vx"], str)) else 0.0
        flux_s = cell.velocity_s["vy"] if (cell.velocity_s and not isinstance(cell.velocity_s["vy"], str)) else 0.0
        divergence = flux_n + flux_e + flux_w + flux_s

        # Skip cells with negligible divergence
        if abs(divergence) < 1e-10:
            continue

        # Find valid edges to adjust
        valid_edges = []
        if cell.velocity_n and not cell.velocity_n["locked"]:
            valid_edges.append(("N", cell.velocity_n))
        if cell.velocity_e and not cell.velocity_e["locked"]:
            valid_edges.append(("E", cell.velocity_e))
        if cell.velocity_w and not cell.velocity_w["locked"]:
            valid_edges.append(("W", cell.velocity_w))
        if cell.velocity_s and not cell.velocity_s["locked"]:
            valid_edges.append(("S", cell.velocity_s))
            
        n_avail = len(valid_edges)
        if n_avail == 0:
            continue
            
        # Calculate and apply correction
        correction = divergence / n_avail
        for direction, edge in valid_edges:
            if direction in ("N", "S"):
                edge["vy"] -= correction
            else:
                edge["vx"] -= correction

def do_red_black_incompressibility(cells, num_steps=5):
    """Run several steps of red-black incompressibility projection"""
    groups = classify_cells(cells)
    for step in range(num_steps):
        do_incompressibility_iteration(groups["red"])
        do_incompressibility_iteration(groups["black"])
    print(f"Completed {num_steps} steps of incompressibility projection")

#############################################
# STEP 5: Semi-Lagrangian Advection
#############################################
def advect_velocities(global_edges, cells, dt):
    """Optimized advection that properly includes source edges for interpolation"""
    # Identify source edges
    source_edges = set()
    for cell in cells:
        if cell.source:
            if cell.velocity_n: source_edges.add(id(cell.velocity_n))
            if cell.velocity_s: source_edges.add(id(cell.velocity_s))
            if cell.velocity_e: source_edges.add(id(cell.velocity_e))
            if cell.velocity_w: source_edges.add(id(cell.velocity_w))
    
    # Build arrays for interpolation
    h_positions = []
    h_values = []
    v_positions = []  
    v_values = []
    
    # CRITICAL CHANGE: Include ALL velocities, including source velocities, for interpolation
    for key, edge in global_edges.items():
        if isinstance(edge["vx"], str) or isinstance(edge["vy"], str):
            continue
            
        if key[0] == "H":
            h_positions.append(edge["midpoint"])
            h_values.append(edge["vy"])
        elif key[0] == "V":
            v_positions.append(edge["midpoint"])
            v_values.append(edge["vx"])
    
    # Process edges (but still don't modify source edges)
    new_edges = {}
    
    # First copy all boundary and source edges
    for key, edge in global_edges.items():
        if isinstance(edge["vx"], str) or isinstance(edge["vy"], str) or id(edge) in source_edges:
            new_edges[key] = edge.copy()
    
    # Build interpolators once per time step
    vy_interpolator = LinearNDInterpolator(np.array(h_positions), np.array(h_values), fill_value=0.0)
    vx_interpolator = LinearNDInterpolator(np.array(v_positions), np.array(v_values), fill_value=0.0)
    
    # Process horizontal edges
    h_edges = [(key, edge) for key, edge in global_edges.items() 
              if key[0] == "H" and key not in new_edges]
    
    # Batch process horizontal edges
    if h_edges:
        h_midpoints = np.array([edge["midpoint"] for _, edge in h_edges])
        vx_at_h_edges = vx_interpolator(h_midpoints)
        
        h_departures = np.zeros_like(h_midpoints)
        for i, (_, edge) in enumerate(h_edges):
            h_departures[i, 0] = h_midpoints[i, 0] - dt * vx_at_h_edges[i]
            h_departures[i, 1] = h_midpoints[i, 1] - dt * edge["vy"]
        
        vy_at_departures = vy_interpolator(h_departures)
        
        for i, (key, edge) in enumerate(h_edges):
            new_edge = edge.copy()
            new_edge["vy"] = vy_at_departures[i] if not np.isnan(vy_at_departures[i]) else 0.0
            new_edges[key] = new_edge
    
    # Process vertical edges
    v_edges = [(key, edge) for key, edge in global_edges.items() 
              if key[0] == "V" and key not in new_edges]
    
    # Batch process vertical edges
    if v_edges:
        v_midpoints = np.array([edge["midpoint"] for _, edge in v_edges])
        vy_at_v_edges = vy_interpolator(v_midpoints)
        
        v_departures = np.zeros_like(v_midpoints)
        for i, (_, edge) in enumerate(v_edges):
            v_departures[i, 0] = v_midpoints[i, 0] - dt * edge["vx"]
            v_departures[i, 1] = v_midpoints[i, 1] - dt * vy_at_v_edges[i]
        
        vx_at_departures = vx_interpolator(v_departures)
        
        for i, (key, edge) in enumerate(v_edges):
            new_edge = edge.copy()
            new_edge["vx"] = vx_at_departures[i] if not np.isnan(vx_at_departures[i]) else 0.0
            new_edges[key] = new_edge
    
    return new_edges

#############################################
# STEP 6: Visualization Functions
#############################################
def compute_cell_intensities(cells):
    """Compute velocity magnitude for each cell"""
    intensities = []
    for cell in cells:
        intensity = 0.0
        if cell.velocity_n and not isinstance(cell.velocity_n.get("vy"), str):
            intensity += abs(cell.velocity_n["vy"])
        if cell.velocity_s and not isinstance(cell.velocity_s.get("vy"), str):
            intensity += abs(cell.velocity_s["vy"])
        if cell.velocity_e and not isinstance(cell.velocity_e.get("vx"), str):
            intensity += abs(cell.velocity_e["vx"])
        if cell.velocity_w and not isinstance(cell.velocity_w.get("vx"), str):
            intensity += abs(cell.velocity_w["vx"])
        intensities.append(intensity)
    return intensities

def plot_cell_heatmap(cells, title="Cell Velocity Intensity"):
    """Plot a heatmap of cell velocity intensity"""
    intensities = compute_cell_intensities(cells)
    
    # Create a 2D grid for the heatmap
    grid = np.zeros((100, 100))
    for cell, intensity in zip(cells, intensities):
        grid[cell.row_index, cell.col_index] = intensity
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, origin='lower', cmap='hot')
    plt.colorbar(label='Velocity Magnitude')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

def plot_velocity_field(global_edges, cells, threshold=1e-6, scale=1.0):
    """Plot the velocity field with arrows"""
    plt.figure(figsize=(14, 14))
    
    # Draw cell boundaries
    for i in range(101):
        plt.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5)
        plt.axvline(x=i, color='lightgray', linestyle='-', linewidth=0.5)
    
    # Highlight source cells
    source_cells = [cell for cell in cells if cell.source]
    for cell in source_cells:
        minx, miny, maxx, maxy = cell.bounds
        plt.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], 
                 color='lightblue', alpha=0.3)
    
    # Draw velocity arrows
    h_arrows = 0
    v_arrows = 0
    
    for key, edge in global_edges.items():
        if edge.get("locked", False) and not any(edge is c.velocity_w for c in source_cells):
            continue
            
        x, y = edge["midpoint"]
        
        if key[0] == "V":  # Vertical edge (east/west)
            if abs(edge["vx"]) > threshold:
                plt.arrow(x, y, scale * edge["vx"], 0, 
                         head_width=0.3, head_length=0.3, fc='green', ec='green')
                v_arrows += 1
                
        elif key[0] == "H":  # Horizontal edge (north/south)
            if abs(edge["vy"]) > threshold:
                plt.arrow(x, y, 0, scale * edge["vy"], 
                         head_width=0.3, head_length=0.3, fc='red', ec='red')
                h_arrows += 1
    
    plt.title(f"Velocity Field (Showing {h_arrows} horizontal, {v_arrows} vertical vectors)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()

#############################################
# STEP 7: Simulation Loop
#############################################
class FlowSimulation:
    def __init__(self, rows=100, cols=100):
        self.cells = create_grid(rows, cols)
        self.global_edges = assign_shared_edges(self.cells, rows, cols)
        self.source_cells = initialize_source_vectors(self.cells, self.global_edges)
        self.dt = 0.1
    
    def step(self):
        """Run one complete simulation step (advection + projection)"""
        # 1. Advect velocities
        self.global_edges = advect_velocities(self.global_edges, self.cells, self.dt)
        
        # 2. Project to maintain incompressibility
        do_red_black_incompressibility(self.cells, num_steps=5)
    
    def advect_only_step(self):
        """Run only the advection part of a step, with source reinitialization"""
        # First perform advection
        self.global_edges = advect_velocities(self.global_edges, self.cells, self.dt)
        
        # Then reinitialize source vectors
        self.reinitialize_source_vectors()
    
    def project_only_step(self):
        """Run only the projection part of a step"""
        do_red_black_incompressibility(self.cells, num_steps=5)
    
    def run_simulation(self, num_steps, mode="full"):
        """Run multiple simulation steps"""
        print(f"Running {num_steps} simulation steps in {mode} mode...")
        
        for i in range(num_steps):
            if i % 10 == 0:
                print(f"Step {i+1}/{num_steps}")
                
            if mode == "full":
                self.step()
            elif mode == "advect_only":
                self.advect_only_step()
            elif mode == "project_only":
                self.project_only_step()
            else:
                raise ValueError(f"Unknown simulation mode: {mode}")
        
        print("Simulation complete!")
    
    def reinitialize_source_vectors(self):
        """Reinitialize the source vectors to ensure they maintain their values"""
        for cell in self.source_cells:
            if cell.velocity_e:
                cell.velocity_e["vx"] = -1.0  # Flowing to the left (westward) from right boundary
                cell.velocity_e["locked"] = True

#############################################
# STEP 8: Run the Simulation with Displays Every 5 Steps
#############################################
if __name__ == "__main__":
    # Create and initialize the simulation for advection only
    sim = FlowSimulation(rows=100, cols=100)
    
    # Print stats about non-zero velocities
    def print_velocity_stats(edges):
        h_count = sum(1 for k, e in edges.items() if k[0] == "H" and abs(e["vy"]) > 1e-6)
        v_count = sum(1 for k, e in edges.items() if k[0] == "V" and abs(e["vx"]) > 1e-6)
        print(f"Non-zero velocities: {h_count} horizontal, {v_count} vertical")
    
    # Display initial state
    print("Initial state (before any advection steps)")
    print_velocity_stats(sim.global_edges)
    plot_velocity_field(sim.global_edges, sim.cells, threshold=0, scale=1.0)
    
    # Run advection with displays every 5 steps
    total_steps = 10
    step_size = 1
    
    for step in range(0, total_steps, step_size):
        # Run 5 steps at a time
        for i in range(step_size):
            sim.advect_only_step()
            
        # Display current state
        print(f"After {step + step_size} advection steps")
        print_velocity_stats(sim.global_edges)
        plot_velocity_field(sim.global_edges, sim.cells, threshold=0, scale=1.0)