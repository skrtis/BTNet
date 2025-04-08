# Malpeque Disease ABM Simulation

![BTNet Logo](images/BTNet_logo_dark_green.jpg)

## Overview

This repository contains an agent-based model (ABM) simulation for studying the spread of Malpeque disease in clam populations, including fluid dynamics, disease transmission, and treatment interventions.

## Demo

<video src="output/demo.mp4" controls title="Malpeque Disease Simulation Demo" width="640"></video>

## Features

- Fluid dynamics simulation with advection and projection
- Realistic ocean current modeling
- Disease transmission with compartment modelling
- Treatment intervention scenarios
- Population health tracking and visualization
- Animated visualizations of disease spread

## Repository Structure

- [`abm.py`](abm.py): Core agent-based model implementation with visualization functions
- [`animation.py`](animation.py): Animation generation module for creating mp4 visualizations
- [`disease.py`](disease.py): Disease transmission and progression logic
- [`grid_generator.py`](grid_generator.py): Creates the simulation grid environment
- [`mechanisms.py`](mechanisms.py): Fluid dynamics and physical mechanisms
- [`identify_populations.py`](identify_populations.py): Tools for identifying distinct clam populations
- [`visuals.py`](visuals.py): Additional visualization utilities

## Getting Started

1. Clone this repository
2. Install required dependencies
3. Run a simulation:

```python
python animation.py