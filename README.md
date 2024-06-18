# PyMMF
python micromagnetic framework by Roke
Uses a PyCUDA framework to simulate 

port of afs ssh repo to git repo

is currently being tested on Ubuntu 18.04
with:

requirements:
cuda 11.8.0
anaconda3/2023.03
ffmpeg/4.0.2
texlive/2022



GET RUNNING:
1.: install requirements
2.: create conda env with necessary repos (quiet a lot)
3.: activate repo
4.: run with "python skyrmion_simulation.py" via python (after "cd skyrmion_simulation/python_scripts") -> Output is via console and into OUTPUT dir
5.: modify masks (black and white pixels, easy via paint i.e.) and input skyrmions in needed_files and rerun with your own specifications
    most parameters can be found inside the class definitions and inits in skyrmion_simulation.py
    several standard modes are available: sim.sim_type -> the specific parameters modified can be found in spin class init



DYNAMIC CURRENT CALCULATION + VISUALIZATION:
Steps 1.2.3. like for Skyrmion Simulation
run with python current_calculation.py after cd current_calculation


FOR PHYSNET UHH USERS:

SKIP THE INSTALL REQUIREMENTS AND RUN VIA SS_PHYSNET.SH
ss_physnet.sh starts skyrmion simulation with necessary modules -> can be used for jobs, configure with email address (and your own log dir)
cc_physnet.sh starts current calculation with necessary modules -> can be used for jobs, configure with email address (and your own log dir)


TODO (For me):
Implement temp dirs cleverly into main calculations -> Skyrmion simulation and Current Calculation
functionalize more of the code inside the analysis scripts
(split skyrmion simulation up into different parts)
achieve support for windows -> paths with os.path or different module# PyMMF
