markdown
# PyMMF: Python Micromagnetic Framework by Roke

Uses a PyCUDA framework to simulate.

Port of AFS SSH repo to Git repo. Currently being tested on Ubuntu 18.04 with:

## Requirements
- cuda 11.8.0
- anaconda3/2023.03
- ffmpeg/4.0.2
- texlive/2022

## Getting Started

1. Install requirements
2. Create conda environment with necessary repositories (quite a lot)
3. Activate repository
4. Run the simulation
    - Use `python skyrmion_simulation.py` via Python
    - Navigate to the script directory with `cd skyrmion_simulation/python_scripts`
    - Output is provided via console and into the `OUTPUT` directory

5. Modify masks  
    - Modify black and white pixel masks (easily done via paint or similar applications)
    - Input Skyrmions in `needed_files` and rerun with your own specifications

Most parameters can be found inside the class definitions and `__init__` methods in `skyrmion_simulation.py`.

Several Standard Modes are Available
`sim.sim_type` -> The specific parameters modified can be found in the spin class `__init__`.

## Dynamic Current Calculation & Visualization

1. Follow steps 1, 2, and 3 as for Skyrmion Simulation
2. Run with `python current_calculation.py` after navigating to the directory with `cd current_calculation`.

## For PhysNet UHH Users

Skip the install requirements and run via `ss_physnet.sh`.

- `ss_physnet.sh` starts the skyrmion simulation with necessary modules (can be used for jobs; configure with your email address and log directory)
- `cc_physnet.sh` starts the current calculation with necessary modules (can be used for jobs; configure with your email address and log directory)

## To Do (For Me)

- Implement temporary directories cleverly into main calculations:
    - Skyrmion simulation
    - Current Calculation
- Functionalize more of the code inside the analysis scripts (split skyrmion simulation into different parts)
- Achieve support for Windows (manage paths with `os.path` or a different module)
