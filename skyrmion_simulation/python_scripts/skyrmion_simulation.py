# PyMMF by Anton Kettner, 2024
# INFO:
# - This script is used to simulate skyrmions in a 2D system using PyCUDA
# - sim class                       --> the basic simulation parameters and METHODS OF THE SIMULATION
# - spin class                      --> spin field, its parameters and the Skyrmion array, METHODS TO ALTER AND ANALYZE THE SPINFIELD
# - cst class for setting constants --> PHYSICAL CONSTANTS AND PARAMETERS
# - math class                      --> MATHEMATICAL OPERATIONS
# - output class                    --> CONSTANTS: output location, METHODS: status bar stats, conversion to mp4, save images

# Standard library imports
import logging  # enabling display of logging.info messages

# import subprocess
import gc
import os
import signal
import sys
import glob
import shutil
import time
import math
import collections
import multiprocessing
import argparse as ap

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden
from numpy.linalg import norm as value
from PIL import Image  # Das csv - Paket wird zum Speichern der CSV - Datei benutzt
from tqdm import tqdm

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import matplotlib

# if os.environ.get("DISPLAY", "") == "":
#     logging.info("No display found. Using non-interactive Agg backend.")
matplotlib.use("Agg")
# else:
#     logging.info("Display found. Using interactive Qt5Agg backend.")
#     matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import matplotlib.patches as patches
# from matplotlib.collections import PolyCollection
import matplotlib
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import uniform_direction
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# import pycuda.gpuarray as gpuarray

# Navigate two levels up from the current script's directory
parent_directory = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Add the parent directory to sys.path to access modules from there
sys.path.insert(0, parent_directory)

# Local application imports
from current_calculation import current_calculation as cc
from analysis.python_scripts.input_output import create_input_output_plot
from analysis.python_scripts.trajectory_trace_new_wall_retention import make_wall_retention_plot

# from trajectory_trace_with_pics import trajectory_trace
from analysis.python_scripts.trajectory_trace_simple import trajectory_trace
from analysis.python_scripts.q_r_vs_time_plot import create_q_r_vs_time_plot
from analysis.python_scripts.angle_distance_plot import current_vs_distance_plot


class cst:
    """
    COMMENT AI GENERATED!!!
    A class that contains the physical constants and material parameters used in the simulation of skyrmions.

    ATTRIBUTES
    s (float)                      : Spin quantum number.
    g_el_neg (float)               : Electron g-factor.
    mu_b (float)                   : Bohr magneton in eV/T.
    mu_free_spin (float)           : Magnetic moment of free spins in eV/T.
    mu_0 (float)                   : Vacuum permeability in Vs/(Am).
    gamma_el (float)               : Gyromagnetic ratio in 1/(ns*T).
    alpha (float)                  : Gilbert damping constant.
    beta (float)                   : Dimensionless non-adiabaticity parameter.
    a (float)                      : Lattice constant in meters.
    h (float)                      : Atomic layer height in meters.
    p (float)                      : Spin polarization.
    e (float)                      : Elementary charge in C.
    coords (dict)                  : Dictionary containing the coordinates x, y, and z.
    rot_matrix_90 (numpy.ndarray)  : Rotation matrix for 90 degrees.
    Temp (float)                   : Temperature in Kelvin.
    v_s_to_j_c_factor (float)      : Conversion factor from spin velocity to current density.
    A_density (float)              : Exchange interaction density in J/m.
    DM_density (float)             : Dzyaloshinskii-Moriya interaction density in J/m^2.
    K_mu (float)                   : Anisotropy in the z-direction in J/m^3.
    B_ext (float)                  : External magnetic field in T.
    B_fields (numpy.ndarray)       : Array of external magnetic fields.
    M_s (float)                    : Saturation magnetization in A/m.
    K_density (float)              : Anisotropy density in J/m^3.
    mu_s (float)                   : Magnetic moment per spin in eV/T.
    skyr_name_ext (str)            : Extension name for skyrmion.
    B_a_quadr (float)              : Effective exchange field for quadratic lattice.
    B_d_quadr (float)              : Effective DM field for quadratic lattice.
    B_k_quadr (float)              : Effective anisotropy field for quadratic lattice.
    B_a_hex (float)                : Effective exchange field for hexagonal lattice.
    B_d_hex (float)                : Effective DM field for hexagonal lattice.
    B_k_hex (float)                : Effective anisotropy field for hexagonal lattice.
    E_a_quadr (float)              : Effective exchange energy for quadratic lattice.
    E_d_quadr (float)              : Effective DM energy for quadratic lattice.
    E_k_quadr (float)              : Effective anisotropy energy for quadratic lattice.
    E_a_hex (float)                : Effective exchange energy for hexagonal lattice.
    E_d_hex (float)                : Effective DM energy for hexagonal lattice.
    E_k_hex (float)                : Effective anisotropy energy for hexagonal lattice.
    NNs (int)                      : Number of nearest neighbors.
    hex_image_scalefactor (int)    : Scale factor for hexagonal images.
    A_Field (float)                : Exchange field.
    DM_Field (float)               : DM field.
    K_Field (float)                : Anisotropy field.
    dAdAtom (float)                : Area per atom.
    dVdAtom (float)                : Volume per atom.
    NN_vecs (numpy.ndarray)        : Nearest neighbor vectors.
    NN_pos_even_row (numpy.ndarray): Nearest neighbor positions for even rows.
    NN_pos_odd_row (numpy.ndarray) : Nearest neighbor positions for odd rows.
    rotate_anticlock (bool)        : Flag to rotate vectors anticlockwise.
    DM_vecs (numpy.ndarray)        : Dzyaloshinskii-Moriya vectors.

    METHODS
    __init__(cls, rotate_anticlock=False): Initializes the class with an optional argument to rotate vectors anticlockwise.
    rotate_vecs_90(self, vecs)           : Rotates the given vectors by 90 degrees clockwise or anticlockwise depending on the value of rotate_anticlock.
    """

    # ---------------------------------------------------------------Attributes: Physical constants-------------------------------------------------------------------

    s = 1 / 2
    g_el_neg = 2.00231930436256
    mu_b = 5.7883818012e-5
    mu_free_spin = 3 * mu_b / 6
    mu_0 = 4 * np.pi * 1e-7
    gamma_el = 176.1
    alpha = 0.1
    beta = alpha / 2
    betas = np.array([beta], dtype=np.float32)
    a = 0.271e-9
    h = 0.4e-9  # atomic layer height according to literature -> s. thesis
    p = 1  # spin polarisation
    e = 1.602176634e-19
    coords = {"x": 0, "y": 1, "z": 2}
    rot_matrix_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)  # Rotation Matrix to create the DM-vecs
    Temp = 4  # in K

    # # Konstanten Schaeffer 2018 from Romming Pd/Fe bilayer on Ir(111)
    A_density = 2e-12  # Austauschwechselwirkung in J/m
    DM_density = 3.9e-3  # DM - Wechselwirkung in J/m^2
    K_mu = 2.5e6  # Anisotropie in Z-Richtung in J/m^3
    B_ext = 1.5  # z-Magnetfeld in T
    B_fields = np.array([B_ext])  # for multiple B_exts
    M_s = 1.1e6  # Saettigungsmagnetisierung in A/m
    K_density = K_mu  # - mu_0 * M_s**2 / 2
    mu_s = 3 * mu_b
    skyr_name_ext = "schaeffer_1.5"
    r_skyr = 1.393  # in nm

    @classmethod
    def __init__(cls, rotate_anticlock=False):
        """
        calculates the DM-vecs based on rotation and NN-vecs.

        Args:
            rotate_anticlock (bool, optional): If True, rotates the field collection
                anticlockwise by 90 degrees. Defaults to False.
        """

        cls.v_s_to_j_c_factor = -2 * cls.e / (cls.a**3 * cls.p)

        # -------------my Conversion to Effective B_field constants in a 2D QUADRATIC LATTICE from Micromagnetic Constants -----------------------

        cls.B_a_quadr = 4 * cls.A_density / (cls.a**2 * cls.M_s) * (1 / 2)

        cls.B_d_quadr = 2 * cls.DM_density / (cls.a * cls.M_s) * (1 / 2)

        cls.B_k_quadr = cls.K_density / cls.M_s

        # -------------Hagemeister 2018 Conversion zu atomistischen B_Feld-Konstanten (hexagonal lattice) -----------------------

        cls.B_a_hex = 4 * cls.A_density / (cls.a**2 * cls.M_s) * (1 / 3)

        cls.B_d_hex = 2 * cls.DM_density / (cls.a * cls.M_s) * (1 / 3)

        cls.B_k_hex = cls.K_density / cls.M_s

        # -------------my Conversion zu atomistischen Energie-Konstanten die funktioniert (quadr lattice)-----------------------

        cls.E_a_quadr = 4 * cls.A_density * cls.mu_s / (cls.a**2 * cls.M_s) * (1 / 2)

        cls.E_d_quadr = 2 * cls.DM_density * cls.mu_s / (cls.a * cls.M_s) * (1 / 2)

        cls.E_k_quadr = cls.K_density * cls.mu_s / cls.M_s

        # -------------Hagemeister 2018 Conversion zu atomistischen Energie-Konstanten (hexagonal lattice)-----------------------

        cls.E_a_hex = 4 * cls.A_density * cls.mu_s / (cls.a**2 * cls.M_s) * (1 / 3)

        cls.E_d_hex = 2 * cls.DM_density * cls.mu_s / (cls.a * cls.M_s) * (1 / 3)

        cls.E_k_hex = cls.K_density * cls.mu_s / cls.M_s

        NN_vec_1 = np.array([1, 0, 0])
        if sim.model_type == "atomistic":
            cls.NNs = 6
            cls.hex_image_scalefactor = 4
            cls.A_Field = cls.B_a_hex
            cls.DM_Field = cls.B_d_hex
            cls.K_Field = cls.B_k_hex
            # cls.angle_between_NNs = 2 * np.pi / NNs  # against the clock rotation
        if sim.model_type == "continuum":
            cls.NNs = 4
            cls.A_Field = cls.B_a_quadr
            cls.DM_Field = cls.B_d_quadr
            cls.K_Field = cls.B_k_quadr
            cls.dAdAtom = cls.a**2 * 2 * np.sqrt(3)
            cls.dVdAtom = cls.dAdAtom * cls.h
        rotation_by_angle = lambda angle: np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # Nearest neighbor vectors
        cls.NN_vecs = np.empty((cls.NNs, 3), dtype=np.float32)
        NN_angle = 2 * np.pi / cls.NNs
        for i in range(cls.NNs):
            cls.NN_vecs[i, :] = np.around((rotation_by_angle(i * NN_angle) @ NN_vec_1), decimals=5)

        if sim.model_type == "atomistic":
            cls.NN_pos_even_row = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, -1, 0]])
            cls.NN_pos_odd_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]])
        if sim.model_type == "continuum":
            cls.NN_pos_even_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
            cls.NN_pos_odd_row = cls.NN_pos_even_row

        cls.rotate_anticlock = rotate_anticlock
        cls.DM_vecs = cst.rotate_vecs_90(cls.NN_vecs)

    @classmethod
    def rotate_vecs_90(cls, vecs):
        """
        Rotate the given vectors by 90 degrees either clockwise or anticlockwise, depending on the value of `rotate_anticlock`.

        Args:
            vecs (numpy.ndarray): A 2D array of shape (N, 3) containing the vectors to be rotated.

        Returns:
            numpy.ndarray: A 2D array of shape (N, 3) containing the rotated vectors.
        """
        if not cls.rotate_anticlock:
            return np.ascontiguousarray(np.dot(cst.rot_matrix_90, vecs.T).T)
        else:
            return np.ascontiguousarray(np.dot(cst.rot_matrix_90.T, vecs.T).T)


class sim:
    """
    COMMENT AI GENERATED!!!
    A class representing the basic simulation parameters and methods for simulating skyrmions.

    ATTRIBUTES:
    - sim_type (str)                : The type of simulation -> many are possible.
    - model_type (str)              : The model type, either "atomistic" (hexagonal lattice) or "continuum" (implicating quadratic lattice).
    - calculation_method (str)      : The calculation method for the simulation, either "rk4", "euler", or "heun".
    - boundary (str)                : The boundary conditions for the simulation, either "ferro" or "open".
    - apply_bottom_angle (bool)     : Whether to apply a bottom angle to the racetrack.
    - bottom_angles (numpy.ndarray) : 1D Array of bottom angles.
    - v_s_factors (numpy.ndarray)   : 1D Array of velocity factors.
    - pivot_point (tuple)           : The pivot point for the simulation.
    - samples (int)                 : The number of samples.
    - final_skyr_No (int)           : The final number of skyrmions.
    - t_max (float)                 : The maximum simulation time in nanoseconds.
    - t_relax_skyr (float)          : Relaxation time for skyrmions.
    - t_relax_no_skyr (float)       : Relaxation time without skyrmions.
    - t_circ_buffer (float)         : Circular buffer time.
    - No_sim_img (int)              : The number of simulation images.
    - cc_steps (int)                : The number of current calculation steps.
    - len_circ_buffer (int)         : Length of the circular buffer.
    - time_per_img (float)          : Time per image.
    - t_last_skyr_frac (float)      : Fraction of time until all skyrmions are set.
    - save_pics (bool)              : Whether to save pictures.
    - save_npys (bool)              : Whether to save numpy arrays.
    - save_npy_end (bool)           : Whether to save numpy arrays at the end.
    - track_radius (bool)           : Whether to track the radius.
    - check_variance (bool)         : Whether to check the variance.
    - check_skyrmion_presence (bool): Whether to check for skyrmion presence.
    - critical_variance (float)     : Critical variance value.
    - max_error (float)             : Maximum error (wall_ret_test_new)
    - t_pics (numpy.ndarray)        : The times at which a picture is saved.
    - steps_per_avg (int)           : The number of steps per average calculation.
    - learning_rate (numpy.ndarray) : The learning rate for the simulation.
    - smallest_error_yet (float)    : The smallest error encountered so far.
    - cons_reach_threashold (int)   : The threshold for consecutive reaches.

    ATTRIBUTES INITIALIZED IN __init__:
    dt (float)                      : Time step for the simulation, adjusted based on the calculation method.
    steps_total (int)               : Total number of steps in the simulation.
    steps_per_pic (int)             : Number of steps between each picture taken during the simulation.
    steps_per_avg (int)             : Number of steps per averaging interval.
    atomistic_upscaling_factor (int): Upscaling factor for atomistic simulations.

    METHODS:
    - __init__()                       : Initializes the simulation parameters.
    - calc_steps_per_avg()             : Calculates the number of steps per average.
    - spa_guess(x)                     : Provides an initial guess for steps per average calculation.
    - compile_kernel(current_ext_field): Compiles the CUDA kernel with the given external field.
    - get_kernel_functions(mod)        : Retrieves the kernel functions from the compiled module.
    """

    # ---------------------------------------------------------------Attributes: Basic Sim Params-------------------------------------------------------------------
    model_type = "atomistic"
    calculation_method = "heun"
    boundary = "open"
    apply_bottom_angle = False
    bottom_angles = np.array([0])
    v_s_factors = np.array([25])
    pivot_point = (250, 100)
    final_skyr_No = 1
    t_max = 1
    t_relax_skyr = 0
    t_relax_no_skyr = 0.3
    t_circ_buffer = 0.01
    No_sim_img = 20
    cc_steps = 600000
    t_last_skyr_frac = 1
    save_pics = True
    save_npys = False
    save_npy_end = True
    track_radius = True
    check_variance = True
    check_skyrmion_presence = True
    critical_variance = 1e-6
    steps_per_avg = 1
    learning_rate = np.array([1, 1])
    smallest_error_yet = 1000
    cons_reach_threashold = 10

    # ---------------------------------------------------------------Methods-------------------------------------------------------------------

    @classmethod
    def __init__(cls, sim_type="x_current"):
        """
        INFO(max timesteps for each calculation method)
        - "euler": Can be used with a max time step of 0.0000033.
        - "rk4"  : Can be used with a max time step of 0.000051.
        - "heun" : Can be used with a max time step of 0.000018.
        """

        # SIMULATION TYPES: "skyrmion_creation" or "wall_retention" or "wall_ret_test_close" or "wall_ret_test_far" or
        #                   "angled_vs_on_edge" or "x_current" or "x_current_SkH_test" or "pinning_tests" or "ReLU" or "ReLU_larger_beta"
        cls.sim_type = sim_type

        if cls.calculation_method == "euler":
            cls.dt = 0.0000033 / 1.8
        elif cls.calculation_method == "rk4":
            cls.dt = 0.000051 / 1.8
        elif cls.calculation_method == "heun":
            cls.dt = 0.000018

        # DEFAULT VELOCITY FIELD V_S PROPERTIES
        cls.v_s_dynamic = False
        cls.v_s_active = True
        cls.v_s_to_wall = False
        cls.v_s_positioning = False
        cls.v_s_factor = 200

        # only needed for wall_ret_test_new
        cls.distances = np.array([])

        if cls.sim_type == "wall_retention":
            # Output dir name
            cls.fig = "Thesis_Fig_9"

            # sim_vars
            # cls.t_max = 20
            cls.t_max = 2
            # cls.No_sim_img = 1000
            cls.No_sim_img = 100
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_100100atomic.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = math.floor((cst.r_skyr * 1e-9 / cst.a * 1.6))
            cls.skyr_set_y = cls.y_size / 2  # 200 bei big

            # v_s properties
            cls.v_s_active = False
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.check_variance = False
            cls.v_s_to_wall = False
            cls.v_s_positioning = False

        elif cls.sim_type == "angled_vs_on_edge":
            cls.fig = "Thesis_Fig_12"
            # sim_vars
            cls.t_max = 50
            cst.beta = cst.alpha
            cls.No_sim_img = 1000

            # set the map
            cls.mask_dir = "needed_files/Mask_track_test.png"
            # cls.mask_dir = "needed_files/Mask_track_test_atomic_step.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = 450
            cls.skyr_set_y = 160

            # v_s properties
            cls.bottom_angles = np.linspace(3, 9, 49, endpoint=True) * -1
            cls.bottom_angles = np.linspace(3, 9, 4, endpoint=True) * -1
            cls.v_s_factors = np.linspace(0.5, 30.5, 31, endpoint=True) * -1
            cls.v_s_factors = np.linspace(0.5, 30.5, 4, endpoint=True) * -1
            cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
            cst.betas = np.ones((cls.samples)) * cst.alpha
            cst.B_fields = np.ones((cls.samples)) * cst.B_ext
            cls.v_s_to_wall = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "skyrmion_creation":
            cls.fig = "Thesis_Fig_8"

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # vary r_skyr a little bit
            cls.t_max = 0.3
            cls.No_sim_img = 50

            # set the skyrmion parameters
            cls.final_skyr_No = 1
            cls.r_skyr = 2 / cst.a * 1e-9 / 10  # r = 1.5 nm --> /10 because of the algorithm also working with already set skyrmions
            cls.check_skyrmion_presence = False
            cls.skyr_set_x = math.floor((cls.x_size) / 2)
            cls.skyr_set_y = math.floor((cls.y_size) / 2)

            # v_s properties
            cls.v_s_factors = np.array([0.5])
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = True
            cls.v_s_to_wall = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "x_current":
            cls.fig = cls.sim_type
            # sim_vars
            cls.t_max = 0.3
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)
            cls.No_sim_img = 10

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = cls.x_size / 2
            cls.skyr_set_y = cls.y_size / 2

            # v_s properties
            cls.v_s_factors = np.array([1])
            cls.v_s_to_wall = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "x_current_SkH_test":

            cls.fig = "Thesis_Fig_11"
            # sim_vars
            cls.t_max = 40
            cst.betas = np.array([0.5, 1, 2]) * cst.alpha
            # cst.beta = cst.alpha
            cls.No_sim_img = 200

            # set the map
            cls.mask_dir = "needed_files/Mask_track_beta_vs_alpha.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            # cls.skyr_set_x = cls.x_size / 2
            cls.skyr_set_x = 50
            cls.skyr_set_y = cls.y_size / 2

            cls.v_s_factors = np.ones((3)) / cst.a / 1e9  # -> should be 10 v_s
            cls.samples = cls.v_s_factors.shape[0]
            cls.v_s_to_wall = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.apply_bottom_angle = False

            cst.B_fields = np.ones((cls.samples)) * cst.B_ext

        elif cls.sim_type == "wall_ret_test_close":
            # Output dir name
            cls.fig = "Thesis_Fig_10_close"
            cls.t_max = 200
            cls.No_sim_img = 5000
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = cls.x_size / 2
            cls.skyr_set_y = cls.y_size / 2

            # v_s properties
            cls.v_s_factors = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.v_s_positioning = True
            cls.check_variance = False

            # set distances
            dist_start = int(cls.x_size - 4 * cst.r_skyr * (1e-9 / cst.a))
            dist_end = int(cls.x_size - 1.5 * cst.r_skyr * (1e-9 / cst.a))
            cls.distances = np.arange(dist_start, dist_end)
            cls.distances = np.unique(cls.distances)
            cls.distances = np.floor(np.linspace(dist_start, dist_end, 4))
            cls.max_error = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        elif cls.sim_type == "wall_ret_test_far":
            # Output dir name
            cls.fig = "Thesis_Fig_10_far"
            cls.t_max = 800
            cls.No_sim_img = 5000
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = cls.x_size / 2
            cls.skyr_set_y = cls.y_size / 2

            # v_s properties
            cls.v_s_factors = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.v_s_positioning = True
            cls.check_variance = False

            # set distances
            dist_start = int(cls.x_size - 7 * cst.r_skyr * (1e-9 / cst.a))
            dist_end = int(cls.x_size - 4 * cst.r_skyr * (1e-9 / cst.a))
            cls.distances = np.arange(dist_start, dist_end)
            cls.distances = np.unique(cls.distances)
            cls.distances = np.floor(np.linspace(dist_start, dist_end, 4))
            cls.max_error = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        elif cls.sim_type == "pinning_tests":
            # CHOOSE ONE OF THOSE 4 options
            # cls.fig = "Thesis_Fig_13_1"
            # cls.mask_dir = "needed_files/Mask_track_narrowing_through.png"

            cls.fig = "Thesis_Fig_13_2"
            cls.mask_dir = "needed_files/Mask_track_narrowing_stuck.png"

            # cls.fig = "Thesis_Fig_14_1"
            # cls.mask_dir = "needed_files/Mask_track_corner_through.png"

            # cls.fig = "Thesis_Fig_14_2"
            # cls.mask_dir = "needed_files/Mask_track_corner_stuck.png"

            cls.t_max = 30
            cls.t_max = 3
            cls.No_sim_img = 200
            cls.No_sim_img = 20
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)
            cls.skyr_set_x = 20
            cls.skyr_set_y = 80
            cls.bottom_angles = np.array([0])
            cls.v_s_factors = np.array([2])
            cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = True
            cls.v_s_centering = False

        elif cls.sim_type == "ReLU":
            # FINAL RESULTS:
            cls.fig = "Thesis_Fig_15"
            cls.mask_dir = "needed_files/Mask_final_ReLU_simplification_bigger_11.png"
            cls.final_skyr_No = 15
            cls.t_max = 100
            cls.No_sim_img = 50
            cls.skyr_set_x = 15
            cls.skyr_set_y = 140
            cls.v_s_factors = np.array([1])
            cls.samples = cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = True
            cls.v_s_centering = False
            cls.v_s_to_wall = False
            cst.beta = cst.alpha / 2

        elif cls.sim_type == "ReLU_larger_beta":
            # FINAL RESULTS:
            cls.fig = "Thesis_Fig_19"
            cls.mask_dir = "needed_files/Mask_final_ReLU_high_beta_modular.png"
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            cls.final_skyr_No = 50
            cls.t_max = 200
            cls.No_sim_img = 1000
            cst.beta = cst.alpha * 2
            cst.betas = np.array([cst.beta], dtype=np.float32)
            cls.skyr_set_x = 15
            cls.skyr_set_y = 140
            cls.v_s_factors = np.array([1])
            cls.samples = cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = True
            cls.v_s_centering = False
            cls.v_s_to_wall = False

        elif cls.sim_type == "wall_ret_test":
            # Output dir name
            cls.fig = "wall_ret_test_middle"
            cls.t_max = 40
            cls.No_sim_img = 1000
            cst.beta = cst.alpha
            cst.betas = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x = cls.x_size / 2
            cls.skyr_set_y = cls.y_size / 2

            # v_s properties
            cls.v_s_factors = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active = True
            cls.v_s_dynamic = False
            cls.v_s_centering = False
            cls.v_s_positioning = True
            cls.check_variance = False

            # set distances
            cls.distances = np.array([int(cls.x_size - 3 * cst.r_skyr * (1e-9 / cst.a))])
            cls.max_error = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        else:
            raise ValueError("Invalid simulation type.")

        # calculated params from base params
        cls.len_circ_buffer = min(max(int(cls.t_circ_buffer * cls.No_sim_img / cls.t_max), 5), 50)
        cls.time_per_img = cls.t_max / cls.No_sim_img
        cls.samples = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
        cls.len_circ_buffer = min(max(int(cls.t_circ_buffer * cls.No_sim_img / cls.t_max), 5), 50)
        cls.time_per_img = cls.t_max / cls.No_sim_img
        cls.t_pics = np.linspace(0, cls.t_max, cls.No_sim_img + 1, endpoint=True)[1:]

        # Berechnung der Anzahl an Bildern, nach denen je ein Skyrmion gesetzt wird
        if not cls.final_skyr_No == 0:
            cls.every__pic_set_skyr = np.floor(cls.No_sim_img * cls.t_last_skyr_frac / cls.final_skyr_No)
        else:
            cls.every__pic_set_skyr = cls.No_sim_img + 1

        # some additional parameters where dt is needed
        cls.total_steps = int(cls.t_max / cls.dt)
        cls.steps_per_pic = int(cls.total_steps / len(cls.t_pics))
        cls.steps_per_avg, cls.total_steps = cls.calc_steps_per_avg()

        # adjust dt to have the same t_max as before:
        cls.dt = cls.t_max / cls.total_steps

        # cls.mask_dir = "needed_files/Mask_track_free.png"
        cls.orig_mask_dir = cls.mask_dir

        # load the mask and get the size of the mask
        cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

        # create an empty array for the current
        cls.v_s = np.empty((cls.x_size, cls.y_size, 2), dtype=np.float32)

        # set the external fields
        cst.B_fields = np.ones((cls.samples)) * cst.B_ext

    @classmethod
    def load_mask(cls, dir):
        cls.mask = np.ascontiguousarray(
            np.array(np.array(Image.open(dir), dtype=bool)[:, :, 0]).T[:, ::-1]
        )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis

        return cls.mask.shape[0], cls.mask.shape[1]

    @classmethod
    def calc_steps_per_avg(cls):
        """
        INFO:
            if output of this function too big, simulation does not work ~ 9000 works
            sa = steps until another picture is saved for averages -> Steps per Average
        """
        # first guess for steps per average
        sa_guess = cls.spa_guess(cls.total_steps)

        # Refining guess to ensure total steps are divisible and maintain a minimum frequency ratio
        while cls.total_steps % sa_guess != 0 or cls.steps_per_pic / sa_guess < 1:
            sa_guess -= 1

        # Calculating revised total steps based on sa_guess to verify against the steps total
        total_step_fn = lambda spag: (int(cls.steps_per_pic / spag)) * (spag + 1) * len(cls.t_pics)
        revised_total_steps = total_step_fn(sa_guess)

        # Adjust the steps_per_avg_guess upwards until the actual total steps meet or exceed the required steps_total
        while revised_total_steps < cls.total_steps:
            sa_guess += 1
            revised_total_steps = total_step_fn(sa_guess)

        return sa_guess, revised_total_steps

    @staticmethod
    def spa_guess(x):
        # Adjust b and c based on observations.
        a = x / 2000
        d = x / 2000
        b = 1e-6
        c = -5
        value = a * np.tanh(b * x + c) + d
        return max(1, int(value))

    @staticmethod
    def compile_kernel():
        """
        Compiles a CUDA kernel for skyrmion simulation with the given external field.
        Args:
            current_ext_field (float): The current external magnetic field value.
        Returns:
            tuple: A tuple containing the compiled CUDA module and the texture reference for the spin field.
        Raises:
            FileNotFoundError: If the kernel file cannot be found.
            IOError          : If there is an error reading the kernel file.
        Notes:
            - The function reads the CUDA kernel from a file based on the current model type and calculation method.
            - It defines several constants in the CUDA script, including physical constants and simulation parameters.
            - The function allocates memory on the GPU and transfers necessary data arrays to the GPU.
            - The texture reference for the spin field is set up for use in the CUDA kernel.
        """

        # Get the kernel directory for the current model type and calculation method
        kernel_dir = GPU.kernel_dirs[sim.calculation_method]

        # Read the kernel from the file
        with open(kernel_dir, "r") as file:
            kernel = file.read()

        # String to include constants into the Cuda-Script ------> with constant v_s: f"__constant__ float3 v_s = {v_s};"
        GPU_constants = (
            f"__constant__ float A        = {cst.A_Field};\n"
            f"__constant__ float DM       = {cst.DM_Field};\n"
            f"__constant__ float K        = {cst.K_Field};\n"
            f"__constant__ float B_ext    = {cst.B_ext};\n"
            f"__constant__ float Temp     = {cst.Temp};\n"
            f"__constant__ float alpha    = {cst.alpha};\n"
            f"__constant__ float beta     = {cst.beta};\n"
            f"__constant__ float gamma_el = {cst.gamma_el};\n"
            f"__constant__ float dt       = {sim.dt:.9f};\n"
            f"__constant__ int size_x     = {sim.x_size};\n"
            f"__constant__ int size_y     = {sim.y_size};\n"
            f"__constant__ int No_NNs     = {cst.NNs};\n"
            f"__constant__ float NN_vec[{cst.NN_vecs.size}];\n"
            f"__constant__ int NN_pos_even_row[{cst.NN_pos_even_row.size}];\n"
            f"__constant__ int NN_pos_odd_row[{cst.NN_pos_odd_row.size}];\n"
            f"__constant__ float DM_vec[{cst.DM_vecs.size}];\n"
            f"#define block_size {GPU.block_dim_x * GPU.block_dim_y}\n"
        )

        mod = SourceModule(GPU_constants + kernel)

        # set the texture reference for the spin field
        texref = mod.get_texref("v_s")
        texref.set_array(GPU.cuda_v_s)
        cuda.mem_alloc(sim.v_s.nbytes)

        # send the NN array and the DM array to the GPU
        NN_vec_id = mod.get_global("NN_vec")[0]
        NN_pos_even_row_id = mod.get_global("NN_pos_even_row")[0]
        NN_pos_odd_row_id = mod.get_global("NN_pos_odd_row")[0]
        DM_vec_id = mod.get_global("DM_vec")[0]
        cuda.memcpy_htod(NN_vec_id, cst.NN_vecs)
        cuda.memcpy_htod(NN_pos_even_row_id, cst.NN_pos_even_row.astype(np.int32))
        cuda.memcpy_htod(NN_pos_odd_row_id, cst.NN_pos_odd_row.astype(np.int32))
        cuda.memcpy_htod(DM_vec_id, cst.DM_vecs)

        return mod, texref

    @staticmethod
    def get_kernel_functions(mod):
        """
        Retrieves the kernel functions for the skyrmion simulation based on the specified calculation method.
        Args:
            mod: The module containing the kernel functions.
        Returns:
            tuple: A tuple containing:
                - simSteps: A list of kernel functions for the simulation steps, or a single kernel function if using the Euler method.
                - avgStep: The kernel function for averaging steps.
                - q_topo: The kernel function for calculating the topological charge.
        Raises:
            AttributeError: If the specified calculation method is not recognized.
        """
        simSteps = None
        if sim.calculation_method == "rk4":
            simSteps = [
                mod.get_function(name)
                for name in [
                    "SimStep_k1",
                    "SimStep_k2",
                    "SimStep_k3",
                    "SimStep_k4_nextspin",
                ]
            ]
        elif sim.calculation_method == "euler":
            simSteps = mod.get_function("SimStep")
        elif sim.calculation_method == "heun":
            simSteps = [
                mod.get_function("EulerSpin"),
                mod.get_function("HeunSpinAlteration"),
            ]

        avgStep = mod.get_function("AvgStep")
        q_topo = mod.get_function("CalQTopo")

        return simSteps, avgStep, q_topo


class spin:
    """
    class containing the major operations on the spinfield and corresponding parameters for a skyrmion simulation as attributes.

    Attributes:
    mask_path (str): The path to the masks image file.
    j_local_path (str): The path to the local current density file.
    skyr_path (str): The path to the skyrmion file.
    mask (numpy.ndarray): The mask image array.
    field_size_x (int): The size of the spinfield in the x direction.
    field_size_y (int): The size of the spinfield in the y direction.
    block_size_x (int): The size of the thread block in the x direction.
    block_size_y (int): The size of the thread block in the y direction.
    grid_size_x (int): The size of the grid in the x direction.
    grid_size_y (int): The size of the grid in the y direction.
    skyr (numpy.ndarray): The skyrmion array.
    skyr_center_idx (tuple): The center index of the skyrmion.
    sykr_set_x (int): The x position of the skyrmion.
    sykr_set_y (int): The y position of the skyrmion.
    v_s_factor (int): The factor for the velocity of charged particles.
    j (numpy.ndarray): The current density array.
    v_s (numpy.ndarray): The velocity of charged particles due to j.
    avgTex (numpy.ndarray): The average of all z components.
    spins_evolved (numpy.ndarray): array containing the evolved spins.
    mask_gpu (int): The allocated memory on the GPU for the mask array.
    avgTex_gpu (int): The allocated memory on the GPU for the average image array.
    cuda_v_s (numpy.ndarray): The spin velocity array in a format compatible with CUDA.
    v_s_centering (bool): If True, the v_s altered until the center of mass of the skyrmion reaches the center of the field.
    v_s_dynamic (bool): If True, the v_s is calculated dynamically for the used mask with the current calculation script.
    """

    # ----------------------------------------------------------------------------Attributes: Spinfield params ----------------------------------------------------------------------------

    # PATHS
    j_dir = "current_temp/current.npy"
    skyr_dir = f"needed_files/skyr_{cst.skyr_name_ext}_{sim.model_type}.npy"

    # # SKYRMION PLACEMENT IN ATOMS
    # skyr_set_x = 250
    # skyr_set_y = 240

    # LOADING SKYRMION
    try:
        skyr = np.load(skyr_dir)
    except:
        logging.info(f"No Skyrmion file at {skyr_dir} existing.")
        skyr = np.array([[[0, 0, 0]]])

    # # POTENTIAL SKYRMION ADJUSTMENTS
    # # INVERT CHIRALITY
    # skyr = np.dstack((-skyr[:, :, 0], -skyr[:, :, 1], skyr[:, :, 2]))

    # # SIZE ADJUSTMENT
    # factor = 2
    # skyr   = np.kron(skyr.copy(), np.ones((factor, factor, 1)))     # increase
    # skyr   = skyr[::factor, ::factor, :]                            # decrease

    # ---------------------------------------------------------------Methods-------------------------------------------------------------------

    @classmethod
    def __init__(cls):
        """
        Initializes the class with the given parameters.
        """
        cls.r_skyr, cls.w_skyr = cls.find_skyr_params(cls.skyr[:, :, 2], (cls.skyr.shape[0] / 2, cls.skyr.shape[1] / 2))

    @classmethod
    def norm(cls, array):
        """
        Normalizes the given array.
        """
        array[sim.mask] /= value(array[sim.mask], axis=-1, keepdims=True)
        return array

    @classmethod
    def masking(cls, array):
        """
        Masks the given array with sim.mask.
        """
        return array * sim.mask[..., np.newaxis]

    @classmethod
    def set_constant_v_s(cls, v_s_sample_factor, angle=0):
        x_old = -0.05 * sim.v_s_factor * v_s_sample_factor
        x_new = np.cos(np.radians(angle)) * x_old
        y_new = np.sin(np.radians(angle)) * x_old
        sim.v_s = np.dstack(
            (
                np.full(
                    (sim.x_size, sim.y_size),
                    x_new,
                ),
                np.full(
                    (sim.x_size, sim.y_size),
                    y_new,
                ),
            )
        ).astype(np.float32)

        return sim.v_s

    @classmethod
    def update_current_and_mask(cls, bottom_angle, v_s_sample_factor, mask_dir, j_dir):
        cls.j_dir = j_dir
        sim.mask_dir = mask_dir

        # load mask
        sim.load_mask(mask_dir)

        # dynamic v_s
        if sim.v_s_dynamic and sim.v_s_active:

            try:
                cls.j = np.load(cls.j_dir)  # [a*nm/ps]
                logging.info("current_temp file found")
                logging.warning(f"j_shape: {cls.j.shape}")
                sim.v_s = np.ascontiguousarray(
                    (
                        cls.j
                        * 7.801e-13
                        #  * cst.a
                        #  * 1e3
                        * sim.v_s_factor
                        * v_s_sample_factor
                    )
                )
                # -2 * cst.e * v_s / (cst.a**3 * cst.p)
            except:
                logging.info("current_temp file not found, resulting to constant current in x dir")
                sim.v_s = cls.set_constant_v_s(v_s_sample_factor=v_s_sample_factor, angle=bottom_angle)

        else:
            logging.info("constant current density with mask")
            sim.v_s = cls.set_constant_v_s(v_s_sample_factor=v_s_sample_factor, angle=bottom_angle)

        # Correct tuple creation
        v_s_test_loc = (int(sim.x_size / 2), int(sim.y_size / 2))

        # Conditional modification of the tuple
        if sim.sim_type == "first_results_replica" or sim.sim_type == "ReLU":
            v_s_test_loc = (v_s_test_loc[0], int(sim.y_size / 4) + v_s_test_loc[1])

        # Logging information
        try:
            logging.info(f"test v_s at {v_s_test_loc}: {sim.v_s[v_s_test_loc[0], v_s_test_loc[1]] * cst.a * 1e9} m/s\n")
            logging.info(f"test v_s at {v_s_test_loc}: {sim.v_s[v_s_test_loc[0], v_s_test_loc[1]] * 1e9} a/s\n")
            logging.info(f"j_c at {v_s_test_loc}: {cst.v_s_to_j_c_factor*sim.v_s[v_s_test_loc[0], v_s_test_loc[1]]} A/m^2")
        except:
            logging.warning(f"v_s test failed at location: {v_s_test_loc}")

    @classmethod
    def update_current(cls, bottom_angle, v_s_sample_factor, j_dir="None"):
        """
        updates v_s, the velocity field with
        """
        cls.j_dir = j_dir

        # for dynamic v_s
        if sim.v_s_dynamic and sim.v_s_active:

            try:
                cls.j = np.load(cls.j_dir)  # [a*nm/ps]
                logging.info("current_temp file found")
                logging.warning(f"j_shape: {cls.j.shape}")
                sim.v_s = np.ascontiguousarray(
                    (
                        cls.j
                        * 7.801e-13
                        #  * cst.a
                        #  * 1e3
                        * sim.v_s_factor
                        * v_s_sample_factor
                    )
                )
            except:
                logging.info("current_temp file not found, resulting to constant current in x dir")
                sim.v_s = cls.set_constant_v_s(v_s_sample_factor=v_s_sample_factor, angle=bottom_angle)

        else:
            logging.info(f"setting constant current density({bottom_angle} °, {v_s_sample_factor})")
            sim.v_s = cls.set_constant_v_s(v_s_sample_factor=v_s_sample_factor, angle=bottom_angle)

        return sim.v_s

    @classmethod
    def racetrack_bottom_angle(cls, center, angle):
        """
        relevant for "angled_wall_comparison"
        Modifies the input mask to set values below a specified line to False.

        Parameters:
        - mask: 2D numpy array
        - center: Tuple specifying a point on the line
        - angle: Angle of the line in degrees. 0 is horizontal, positive values are counter-clockwise.

        Returns:
        - Modified mask
        """
        # sets the mask to the original mask
        sim.load_mask(sim.orig_mask_dir)

        if sim.apply_bottom_angle:
            # ---------------------------------------------------------------Set everything below line in area to false---------------------------------------------
            # leaving this space on both sides
            s = 130

            # width of the mask
            width = sim.mask.shape[0]

            # slope of the line
            m = np.tan(np.radians(angle))

            # For each column x, compute the y value of the line and set values below it to False.
            for x in range(width - 2 * s):
                y_line = int(m * (x + s - center[0]) + (sim.y_size - center[1]))
                sim.mask[x + s, :y_line] = False

            # ---------------------------------------------------------------Set everything below this y to false---------------------------------------------
            y = int(m * (s - center[0]) + (sim.y_size - center[1]))  # y value at x = 0
            for x in range(width):
                sim.mask[x, :y] = False

        return sim.mask

    @classmethod
    def initialize_spinfield(cls):
        """
        Sets the initial spinfield for a given simulation as well as the spins_id as a class attribute.

        Parameters:
        sim_no (int): The index of the current simulation.

        Returns:
        spinfield (numpy.ndarray): The initial spinfield for the simulation.
        calqtopo (float): The topological charge density of the spinfield.
        """
        logging.info(f"setting init_spinfield")

        # define init state from random numbers between -0.1 and 0.1 in x and y direction and 1 in z direction
        # x, y = -0.1 to 0.1    # z = 1
        x_and_y = np.random.rand(sim.x_size, sim.y_size, 2) * 0.2 - 0.1
        z = np.ones((sim.x_size, sim.y_size, 1))
        spinfield = np.dstack((x_and_y, z))

        # transform into antiferromagnet
        if sim.sim_type == "antiferromagnet_simulation":
            spinfield[::2, ::2, 2] = -1
            spinfield[1::2, 1::2, 2] = -1

        # # INITSTATE SOME RANDOM NUMBERS AT X,Y
        # x = 200
        # y = 200
        # rnd_size = 10
        # var = int(rnd_size / 2)
        # spinfield[x - var : x + var, y - var : y + var] = math.rnd_vecs(
        #     (rnd_size, rnd_size)
        # )

        # # INITSTATE ALL RANDOM NUMBERS IN X,Y
        # spinfield = math.rnd_vecs((spin.field_size_x, spin.field_size_y))

        # mask and norm the spins
        spinfield = spin.masking(spin.norm(spinfield))

        # set boundary
        if sim.boundary == "ferro":
            logging.info("boundary: ferromagnetic")
            spinfield = spin.set_ferromagnetic_boundary(spinfield)

        elif sim.boundary == "open":
            logging.info("boundary: open\n")

        spinfield = spinfield.astype(np.float32)

        # allocate memory on the GPU for everything and the spinfield
        GPU.transfer_to_GPU(spinfield)

        return spinfield

    @staticmethod
    def sigmoid(x, exp_Factor=-20):
        """
        Calculates the sigmoid function for a given input:
        interpolating for insertion of new skyrmion

        Parameters:
        x (float): The input value.
        exp_Factor (float): The exponential factor. Default is -20.

        Returns:
        float: The output of the sigmoid function for the given input.
        """
        return 1 / (1 + np.exp(exp_Factor * (min(x, 1) - 1)))

    @staticmethod
    def rnd_vecs(shape):
        """
        Generates random vectors on the surface of a unit sphere for each point in the given shape.

        Parameters:
        shape (tuple): Shape of the array (e.g., (100, 200)).

        Returns:
        numpy array: A shape[0] x shape[1] x 3 array of random vectors.
        """
        # Number of points

        vectors = uniform_direction(3).rvs(shape[0] * shape[1], random_state=np.random.default_rng())
        vectors = vectors.reshape(shape[0], shape[1], 3)
        return vectors

    @staticmethod
    def set_ferromagnetic_boundary(spinfield):
        """
        Sets all the spins right at the edge of the mask that are now 0 in all directions to 1 in z direction.

        Parameters:
        spinfield (numpy.ndarray): A 3D numpy array representing the spin field.
            (first two dimensions are the x and y coordinates, third dimension is the x, y, and z component of the spin)

        Returns:
        numpy.ndarray: A 3D numpy array representing the spin field with the ferromagnetic boundary set.
        """

        NN_pos_odd_row = cst.NN_pos_odd_row[:, :-1]
        NN_pos_even_row = cst.NN_pos_even_row[:, :-1]

        for i in range(sim.x_size):
            for j in range(sim.y_size):
                if spinfield[i, j, :].all() == 0:
                    # Determine the nearest neighbor positions based on whether the row is even or odd
                    NN_pos = NN_pos_even_row if j % 2 == 0 else NN_pos_odd_row

                    # Iterate over the nearest neighbor positions
                    for dx, dy in NN_pos:
                        nx, ny = i + dx, j + dy

                        # Check if the nearest neighbor is within the spinfield and is not zero
                        if 0 <= nx < sim.x_size and 0 <= ny < sim.y_size and not spinfield[nx, ny, :].all() == 0:
                            spinfield[i, j, :] = np.array([0, 0, 1])
                            break  # Exit the loop as soon as a non-zero neighbor is found
        return spinfield

    @staticmethod
    def calculate_learning_rate(t, step_size=10, base_lr=0.1, max_lr=1.1):
        """
        Adjust learning rate in a cyclical manner based on step size.
        :param old_error: Previous error value.
        :param new_error: Current error value.
        :param past_learning_rate: Previous learning rate.
        :param step_size: Number of steps in half a cycle.
        :param base_lr: Minimum learning rate.
        :param max_lr: Maximum learning rate.
        :return: Adjusted learning rate.
        """
        # Calculate the cycle
        cycle = np.floor(1 + t / (2 * step_size))
        # Calculatechange in time within a cycle
        x = np.abs(t / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))

        return np.array([lr, lr])

    @staticmethod
    def set_skyr(spins, skyrmion, x, y):
        """
        Places a skyrmion at the specified position.

        Args:
        spins (numpy.ndarray): The spin configuration.
        x, y (int): The x and y coordinates of the center of the skyrmion.

        Returns:
        numpy.ndarray: The updated spin configuration with the skyrmion placed at the specified position.
        """
        # set x, y to be ints
        x = int(x)
        y = int(y)
        logging.info("setting skyrmion")
        skyr_size_x = skyrmion.shape[0]
        skyr_size_y = skyrmion.shape[1]
        # skyr_radius_set = math.ceil(2 * spin.r_skyr / (cst.a * 1e9))
        logging.info(f"r_skyr: {spin.r_skyr}")
        skyr_radius_set = spin.r_skyr * 10

        # skyr array boundary on each side
        boundary = (skyr_size_x + skyr_size_y) / 2 / 20

        skyr_center_x = int(skyr_size_x / 2)
        skyr_center_y = int(skyr_size_y / 2)

        for i in range(skyr_size_x):
            # Schleife ueber alle Positionen von dem einzelnen Skyrmion
            for j in range(skyr_size_y):
                if boundary < i < skyr_size_x - boundary and boundary < j < skyr_size_y - boundary:
                    # Index des Spins in spins, der dem Spin in singleSkyr entspricht
                    idx = x - int(skyr_center_x) + i
                    idy = y - int(skyr_center_y) + j

                    # normierter Abstand von i und j von der Skyrmionmitte --> bei 1 am Rand des Skyrmions
                    # dist = 4 * spin.r_skyr
                    if j % 2 == 0 and sim.sim_type == "skyrmion_creation":
                        x_pos = i + 1 / 2
                    else:
                        x_pos = i
                    if sim.model_type == "atomistic":
                        dist = value([(x_pos - skyr_center_x), (j - skyr_center_y) * np.sqrt(3) / 2]) / (skyr_radius_set)
                    elif sim.model_type == "continuum":
                        dist = value([(i - skyr_center_x), (j - skyr_center_y)]) / (skyr_radius_set)
                    else:
                        dist = "undefined"
                        raise ValueError("Invalid model type. Please choose either 'atomistic' or 'continuum'.")

                    # Abfrage, ob der Spin in spins liegt
                    if 0 <= idx < sim.x_size and 0 <= idy < sim.y_size:
                        # Es wird zwischen der vorherigen Spinkonfigeruration und dem Skyrmion mittels des Abstandes und einer Sigmoid-Funktion interpoliert
                        sig_skyr = (-2 * spin.sigmoid(dist)) + 1
                        sig_spinfield = 1 - sig_skyr
                        spins[idx, idy] = skyrmion[i, j] * sig_skyr + spins[idx, idy] * sig_spinfield
                        # # Normalisierung der Spins
                        # spins[idx, idy] = spins[idx, idy] / np.sqrt(np.dot(spins[idx, idy], spins[idx, idy]))

        # norm and mask the spins
        spins = spin.masking(spin.norm(spins))

        # set boundary ferro if necessary
        if sim.boundary == "ferro":
            spins = spin.set_ferromagnetic_boundary(spins)

        return spins

    @staticmethod
    def skyr_profile(r, R, w):
        """
        Describes the radial profile of a skyrmion with radius R and wall width w.
        same as cos(2*arctan((sinh(R/w)/sinh(r/w))))
        as 2*arctan((sinh(R/w)/sinh(r/w))) is the angle from the center of the skyrmion to the point r
        """
        return np.tanh((r - R) / w)

    @staticmethod
    def find_skyr_params(m_z, center):
        """
        Fit the mz component of the magnetization to determine the skyrmion radius using scipy's curve_fit.

        Parameters:
        mz (2D numpy array): The mz component of the magnetization.
        center (tuple): The coordinates of the center of the skyrmion (y, x).

        Returns:
        float: The radius R and wall width W of the skyrmion.
        """

        y, x = np.indices(m_z.shape)
        if sim.model_type == "continuum":
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        elif sim.model_type == "atomistic":
            # Adjust x and y to account for the hexagonal grid
            adjusted_x = x + 0.5 * (y % 2)
            adjusted_y = y * np.sqrt(3) / 2

            # Adjust the center coordinates
            adjusted_center = (center[0] * np.sqrt(3) / 2, center[1] + 0.5 * (center[0] % 2))

            # Calculate r using the adjusted x, y and center
            r = np.sqrt((adjusted_x - adjusted_center[1]) ** 2 + (adjusted_y - adjusted_center[0]) ** 2)
        else:
            r = 0
            raise ValueError("Invalid model type. Please choose either 'atomistic' or 'continuum'.")
        # Bin m_z values by radial distance to get average m_z for each r
        bins = np.logspace(np.log10(1), np.log10(np.max((r))), 400)
        bin_indices = np.digitize(r, bins)
        binned_mz = [np.mean(m_z[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(1, len(bins))]
        # remove nan values from bins and binned_mz
        bins = bins[:-1][~np.isnan(binned_mz)]
        binned_mz = np.array(binned_mz)[~np.isnan(binned_mz)]

        # Initial guesses for R and w
        p0 = [5, 1]

        # Fit using curve_fit
        popt, _ = curve_fit(spin.skyr_profile, bins, binned_mz, p0=p0)
        skyr_params = np.array(popt) * 1e9 * cst.a  # Convert back to lattice units

        return float(skyr_params[0]), float(skyr_params[1])  # Return the fitted R value

    # Spinfield relaxieren mit festgehaltenem Skyrmion
    @staticmethod
    def relax_but_hold_skyr(
        spins,
        numerical_steps,
        skyrs_set=0,
    ):
        """
        Relax the spin field while holding the skyrmion in place.

        Args:
            spins (ndarray): The spin field.
            numerical_steps (function): The function to perform a simulation step.
            gpu_spins (DeviceAllocation): The device memory for the spin field.
            No (int): The sample number.
            bottom_angle (float): The bottom angle.
            v_s_fac (float): The v_s factor.
            skyrs_set (int, optional): The skyrs set. Defaults to 0.

        Returns:
            ndarray: The relaxed spin field.
        """

        if skyrs_set > 0:
            return spins
        else:
            steps_relax = int(sim.t_relax_no_skyr / sim.dt)

            # Print that the relaxation is starting
            logging.info(f"relaxing spinfield for {sim.t_relax_no_skyr} ns with {steps_relax} steps, no skyr\n")

            # transfer the temp vs 0 bool to GPU
            v_s_temp = False
            cuda.memcpy_htod(GPU.v_s_active_id, np.int32(v_s_temp))

        for _ in range(steps_relax):
            GPU.perform_numerical_steps(
                numerical_steps,
            )

        logging.info("relaxation done\n")

        # reset back v_s to original value
        if skyrs_set == 0:
            cuda.memcpy_htod(GPU.v_s_active_id, np.int32(sim.v_s_active))

        # replace relax mask with original mask
        cuda.memcpy_htod(GPU.mask_id, sim.mask) if skyrs_set > 0 else None

        # fetch the spins from the GPU into new variable
        relaxed_spins = np.empty_like(spins)
        cuda.memcpy_dtoh(relaxed_spins, GPU.spins_id)

        return relaxed_spins

    @staticmethod
    def find_skyr_center(m_z_diff, density_threashold=1.5):
        """
        Determine the center of the skyrmion using a weighted center of mass approach.
        Skyrmion is in the negative z direction.
        m_z_diff is m_z, reduces by the init spins and set to 0 where bigger then 0


        Parameters:
        m_z_diff (2D numpy array): The mz component of the magnetization.

        Returns:
        tuple: The coordinates (y_center, x_center) of the skyrmion center.
        """
        density = -m_z_diff  # goes from 0 to 2

        if sim.final_skyr_No <= 1:
            x_indices, y_indices = np.indices(m_z_diff.shape)

            # print(f"density_max: {np.max(density)}") if idx == 0 else None
            # print(f"density_min: {np.min(density)}") if idx == 0 else None

            sig = density  # **20

            # density[density < 1] = 0

            x_center = np.sum(x_indices * sig) / np.sum(sig)
            y_center = np.sum(y_indices * sig) / np.sum(sig)

            return x_center, y_center
        else:
            potential_centers = np.argwhere(density > density_threashold)

            # Filter out points that are not local minima within the specified radius
            accepted_centers = []
            for center in potential_centers:
                y, x = center
                if density[y, x] == np.max(
                    density[
                        max(0, y - math.ceil(spin.r_skyr)) : y + math.ceil(spin.r_skyr),
                        max(0, x - math.ceil(spin.r_skyr)) : x + math.ceil(spin.r_skyr),
                    ]
                ):
                    accepted_centers.append(center)

            return np.array(accepted_centers)


class GPU:

    kernel_dir_euler = "kernels/ss_kernel_euler.c"
    kernel_dir_heun = "kernels/ss_kernel_heun.c"
    kernel_dir_rk4 = "kernels/ss_kernel_rk4.c"
    kernel_dirs = {"euler": kernel_dir_euler, "heun": kernel_dir_heun, "rk4": kernel_dir_rk4}

    @classmethod
    def __init__(cls):

        # BLOCK AND GRID SIZES BASED ON MASK
        cls.block_dim_x = 8
        cls.block_dim_y = 8
        cls.grid_dim_x = math.ceil(sim.x_size / cls.block_dim_x)
        cls.grid_dim_y = math.ceil(sim.y_size / cls.block_dim_y)

        # TO BLOCK MEMORY ON GPU
        cls.avgTex = np.zeros([sim.x_size, sim.y_size], dtype=np.float32)
        cls.spins_evolved = np.zeros((sim.x_size, sim.y_size, len(spin.skyr[0, 0, :])), dtype=np.float32)
        cls.q_topo_results = np.zeros(sim.x_size * sim.y_size, dtype=np.float32)

    @classmethod
    def transfer_to_GPU(cls, spinfield):
        """
        Allocates memory on the GPU and transfers the mask, the average image of z components to the GPU.

        Parameters:
        cls (object): An instance of the class.

        Returns:
        None
        """
        # Allocation of Mem on GPU and transfer of the mask, the Average image
        cls.mask_id = cuda.mem_alloc(sim.mask.nbytes)
        cuda.memcpy_htod(cls.mask_id, sim.mask)

        # cls.avgTex         = np.zeros([sim.x_size, sim.y_size], dtype=np.float32)
        cls.avgTex_id = cuda.mem_alloc(cls.avgTex.nbytes)
        cuda.memcpy_htod(cls.avgTex_id, cls.avgTex)

        # Allocation and transfer the v_s_active bool to GPU
        cls.v_s_active_id = cuda.mem_alloc(np.int32(0).nbytes)
        cuda.memcpy_htod(cls.v_s_active_id, np.int32(sim.v_s_active))

        # As textured memory needs the kernel for sending the texture, here only the definition that it is an np array
        cls.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")

        # save space for q topo calculation on GPU
        cls.q_topo_id = cuda.mem_alloc(GPU.q_topo_results.nbytes)
        cuda.memcpy_htod(cls.q_topo_id, GPU.q_topo_results)

        if sim.calculation_method == "rk4":
            cls.k1_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            cls.k2_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            cls.k3_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            # cls.pot_next_spin_id = None
            # cls.k4_gpu = cuda.mem_alloc(cls.spins_evolved.nbytes)     # not needed because no new step from there is calculated
        if sim.calculation_method == "heun":
            cls.pot_next_spin_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            # cls.k1_id = cls.k2_id = cls.k3_id = None  #! Critical maybe

        # allocate memory on GPU and transfer the spinfield
        cls.spins_id = cuda.mem_alloc(spinfield.nbytes)  # Allocate memory on GPU
        cuda.memcpy_htod(cls.spins_id, spinfield)  # Copy spins to GPU

        # # allocate memory on GPU for the current
        # cls.v_s_id = cuda.mem_alloc(sim.v_s.nbytes)

        # # transfer the v_s to GPU
        # cls.v_s_id = cuda.mem_alloc(sim.v_s.nbytes)
        # cuda.memcpy_htod(cls.v_s_id, sim.v_s)

        cuda.Context.synchronize()

    @classmethod
    def free_GPU_memory(cls):
        """
        Frees the memory allocated on the GPU.

        Parameters:
        cls (object): An instance of the class.

        Returns:
        None
        """
        # Free the memory allocated on the GPU
        cls.mask_id.free()
        cls.avgTex_id.free()
        cls.v_s_active_id.free()
        cls.q_topo_id.free()
        cls.spins_id.free()
        if sim.calculation_method == "rk4":
            cls.k1_id.free()
            cls.k2_id.free()
            cls.k3_id.free()
        elif sim.calculation_method == "heun":
            cls.pot_next_spin_id.free()
        cuda.Context.synchronize()

    @staticmethod
    def perform_numerical_step(step_func, *args):
        step_func(
            *args,
            block=(GPU.block_dim_x, GPU.block_dim_y, 1),
            grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1),
        )

    @staticmethod
    def perform_numerical_steps(numerical_steps):
        if sim.calculation_method == "rk4":
            steps_args = [
                (numerical_steps[0], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.v_s_active_id),
                (numerical_steps[1], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.k2_id, GPU.v_s_active_id),
                (numerical_steps[2], GPU.spins_id, GPU.mask_id, GPU.k2_id, GPU.k3_id, GPU.v_s_active_id),
                (numerical_steps[3], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.k2_id, GPU.k3_id, GPU.v_s_active_id),
            ]
        elif sim.calculation_method == "euler":
            # print(numerical_steps[0])
            steps_args = [(numerical_steps, GPU.spins_id, GPU.mask_id, GPU.v_s_active_id)]
        elif sim.calculation_method == "heun":
            steps_args = [
                (numerical_steps[0], GPU.spins_id, GPU.mask_id, GPU.pot_next_spin_id, GPU.v_s_active_id),
                (numerical_steps[1], GPU.spins_id, GPU.mask_id, GPU.pot_next_spin_id, GPU.v_s_active_id),
            ]
        else:
            raise ValueError(f"Unknown method: {sim.calculation_method}")

        for step_args in steps_args:
            GPU.perform_numerical_step(*step_args)


class output:
    """
    A class containing static methods for outputting data from the simulation.

    Methods:
    save_images(spinfield, No, time=0, extra=""): Saves images of the magnetization of a spinfield at a given time for each coordinate direction.
    two_d_plot(x, y, x_label, y_label): Plots a two-dimensional graph of x and y data.
    """

    # q_location_tracker = np.zeros((sim.No_sim_img, 5))

    max_parallel_processes = 10
    plot_processes = []
    ctrl_c_counter = 0

    # ---------------------------------------------------------------Methods-------------------------------------------------------------------
    @classmethod
    def __init__(cls):
        # setting the destination for the output
        # get the last part of the mask_dir
        spinfield_name = os.path.splitext(os.path.basename(sim.mask_dir))[0]
        cls.dest = f"OUTPUT/{sim.fig}_{sim.sim_type}"

        # Erstellen der Ordnerstruktur und loeschen der alten Ordner falls vorhanden
        if os.path.exists(f"{cls.dest}"):
            shutil.rmtree(f"{cls.dest}")
        os.makedirs(f"{cls.dest}")
        logging.info(f"created directory {cls.dest}")

        cls.configure_logging(dest_dir=cls.dest)

        cls.log_params()

        if sim.model_type == "atomistic":
            output.atomistic_upscaling_factor = 5
            cls.upscaling_indices, cls.locations = output.calculate_upscaling_array(sim.x_size, sim.y_size)

        # Get the seismic colormap
        seismic = matplotlib.colormaps["seismic"]

        # Create a new colormap from the seismic colormap
        cls.custom_seismic = colors.ListedColormap(seismic(np.linspace(0, 1, 256)))

        # Set the 'bad' (NaN) color to black
        cls.custom_seismic.set_bad(color="black")

        cls.skyrmion_dtype = [
            ("time", np.float32),
            ("topological_charge", np.float32),
            ("left_count", np.float32),
            ("right_count", np.float32),
        ]
        cls.skyrmion_dtype += [(f"x{i}", np.float32) for i in range(sim.final_skyr_No)]
        cls.skyrmion_dtype += [(f"y{i}", np.float32) for i in range(sim.final_skyr_No)]
        if sim.final_skyr_No <= 1:
            cls.skyrmion_dtype += [("r1", np.float32)]
            cls.skyrmion_dtype += [("w1", np.float32)]

        if sim.v_s_positioning:
            logging.warning("v_s_positioning active")
            cls.skyrmion_dtype += [("error", np.float32, 2)]
            cls.skyrmion_dtype += [("v_s_x", np.float32)]
            cls.skyrmion_dtype += [("v_s_y", np.float32)]

        # Create the tracking array
        cls.stat_tracker = np.zeros((sim.No_sim_img,), dtype=cls.skyrmion_dtype)

    @staticmethod
    def log_params():

        # LOGGING PARAMS
        logging.info(f"SIMULATION BASIC PARAMS for {sim.sim_type.upper()}")
        logging.info(f"No_skyrs: {sim.final_skyr_No}")
        logging.info(f"t_op: {sim.t_max/(sim.final_skyr_No+1)}")
        if sim.sim_type == "wall_ret_test_close" or sim.sim_type == "wall_ret_test_far":
            logging.warning(f"distances: {sim.distances}")
        logging.info(f"sim.t_pics[:5]: {sim.t_pics[:5]}")
        logging.info(f"save_avg: {sim.steps_per_avg}")
        logging.info(f"steps total based on loops: {(int(sim.steps_per_pic / sim.steps_per_avg)) * (sim.steps_per_avg + 1) * len(sim.t_pics)}")
        logging.info(f"sim_model: {sim.model_type}")
        logging.info(f"Field Shape [spins]: {sim.mask.shape}")
        nm_factor = cst.a * 1e9 * np.array([1, 1])
        if sim.model_type == "atomistic":
            nm_factor[1] = nm_factor[1] * np.sqrt(3) / 2
        logging.info(f"Field Shape [nm]: {sim.mask.shape * nm_factor}")
        logging.info(f"Boundary: {sim.boundary}")
        logging.info(f"skyr init pos: ({sim.skyr_set_x}, {sim.skyr_set_y})")
        logging.info(f"skyr radius: {spin.r_skyr:.4g} [nm]")
        try:
            logging.info(f"skyr wall width: {spin.w_skyr:.4g} [nm]")
        except:
            logging.warning("skyr wall width not set yet")
        logging.info(f"max Time: {sim.t_max:.4g} ns")
        logging.info(f"No sim img: {sim.No_sim_img}")
        logging.info(f"dt: {sim.dt:.4g} ns")
        logging.info(f"steps_total: {sim.total_steps}")
        logging.info(f"time_per_img: {sim.time_per_img:.4g} ns")
        if sim.len_circ_buffer == 30:
            logging.warning(f"len_circ_buffer: {sim.len_circ_buffer} --> timestep too small for accurate automatic shutdown, set manually")
        else:
            logging.info(f"len_circ_buffer: {sim.len_circ_buffer}")
        logging.info(f"bottom_angles: {tuple(sim.bottom_angles)}")
        logging.info(f"v_s aktiv: {sim.v_s_active}")
        logging.info(f"v_s durch skript berechnet: {sim.v_s_dynamic}")
        logging.info(f"v_s_factors: {tuple(sim.v_s_factors)}\n")
        logging.info(f"v_s to wall: {sim.v_s_to_wall}")
        if sim.v_s_to_wall:
            logging.info(f"v_s_to_wall threshold: {sim.x_threashold}\n")

        logging.info(f"PHYSICAL CONSTANTS")
        logging.info(f"mu_free_spin: {cst.mu_free_spin:.4g} eV/T")
        logging.info(f"mu_0: {cst.mu_0:.4g} Vs/(Am)")
        logging.info(f"(Mesh-size or atomic length) a: {cst.a:.4g} m")
        logging.info(f"NN_vecs: {cst.NN_vecs}")
        logging.info(f"DM_vecs: {cst.DM_vecs}")
        logging.info(f"NN_pos_even_row: {cst.NN_pos_even_row}")
        logging.info(f"NN_pos_odd_row: {cst.NN_pos_odd_row}\n")

        logging.info(f"SIMULATION PARAMETERS for sim.sim_type: {sim.sim_type}")
        logging.info(f"B_ext: {cst.B_ext:.4g} T")
        logging.info(f"gamma_el: {cst.gamma_el:.4g} 1/(T*ns)")
        logging.info(f"alpha: {cst.alpha:.4g}")
        logging.info(f"beta: {cst.beta:.4g}")
        logging.info(f"calculation method: {sim.calculation_method}\n")

        logging.info(f"MICROMAGNETIC CONSTANTS")
        logging.info(f"A_density: {cst.A_density:.4g} J/m")
        logging.info(f"DM_density: {cst.DM_density:.4g} J/m^2")
        logging.info(f"K_density: {cst.K_density:.4g} J/m^3")
        # logging.info(f"K_mu: {cst.K_mu:.4g} J/m^3")
        logging.info(f"M_s: {cst.M_s:.4g} A/m\n")

        logging.info(f"2D QUADRATIC LATTICE B_EFF CONSTANTS")
        logging.info(f"B_a_quadr: {cst.B_a_quadr:.4g} T ")
        logging.info(f"B_d_quadr: {cst.B_d_quadr:.4g} T ")
        logging.info(f"B_k_quadr: {cst.B_k_quadr:.4g} T \n")

        logging.info(f"HEXAGONAL LATTICE B_EFF CONSTANTS")
        logging.info(f"B_a_hex: {cst.B_a_hex:.4g} T ")
        logging.info(f"B_d_hex: {cst.B_d_hex:.4g} T ")
        logging.info(f"B_k_hex: {cst.B_k_hex:.4g} T \n")

        logging.info(f"ATOMIC E CONST FROM MY CONV FORMULA FOR 2D QUADRATIC LATTICE")
        logging.info(f"E_a_quadr : {cst.E_a_quadr / cst.M_s:.4g} eV")
        logging.info(f"E_dm_quadr : {cst.E_d_quadr / cst.M_s:.4g} eV")
        logging.info(f"E_k_quadr : {cst.E_k_quadr / cst.M_s:.4g} eV\n")

        logging.info(f"ATOMIC CONSTANTS REAL FOR HEXAGONAL LATTICE")
        logging.info(f"E_a_hex: {cst.E_a_hex:.4g} eV")
        logging.info(f"E_d_hex: {cst.E_d_hex:.4g} eV")
        logging.info(f"E_k_hex: {cst.E_k_hex:.4g} eV\n")

        logging.info(f"CUDA PARAMS")
        logging.info(f"block_size_y: {GPU.block_dim_y:.4g}\n")
        logging.info(f"blockspergrid: ({GPU.grid_dim_x}, {GPU.grid_dim_y}, 1)")
        logging.info(f"threadsperblock: ({GPU.block_dim_x}, {GPU.block_dim_y}, 1)\n")
        # logging.info(f"v_s: {sim.v_s} m/s")
        # logging.info("Output at _[ns]:", *np.around(checkpoint_times, 8), sep="\n")

    @staticmethod
    def count_open_fds():
        """
        Counts the number of open file descriptors for current process
        """
        fds = 0
        for fd in range(os.sysconf("SC_OPEN_MAX")):  # type: ignore
            try:
                os.fstat(fd)
                fds += 1
            except OSError:
                continue
        return fds

    @staticmethod
    def configure_logging(dest_dir="None"):
        """
        Configures the logging settings for the simulation.

        Parameters:
        dest_dir="terminal_output.log" (str): The path to the log file.

        Returns:
        None
        """

        # logging config
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        logger = logging.getLogger()
        log = dest_dir + "/terminal_output.log"
        file_handler = logging.FileHandler(log, mode="w")
        file_handler.setLevel(logging.INFO)  # Setting the log level for file handler
        file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))  # Keeping the format consistent

        # Add the FileHandler to the logger
        logger.addHandler(file_handler)

        logging.info(f"Logging to {log}")

    @staticmethod
    def calculate_upscaling_array(field_size_x, field_size_y):
        """
        Determine the upscaling locations that each pixel in the original hex array will be upscaled to."""

        logging.info(
            f"CALCULATING UPSCALING ARRAY FOR ATOMISTIC UPSCALING FACTOR {output.atomistic_upscaling_factor} and FIELD SIZE {field_size_x} x {field_size_y}\n"
        )

        # Create arrays for i and j indices
        i, j = np.indices((field_size_x, field_size_y))

        # Calculate x and y values
        x = i + 0.5 * ((j + 1) % 2)
        y = j * np.sqrt(3) / 2

        # Combine x and y values into a single array
        orig_locations = np.dstack([x, y])

        # Create arrays for i and j indices
        i, j = (
            np.indices(
                (output.atomistic_upscaling_factor * field_size_x, math.ceil(field_size_y * output.atomistic_upscaling_factor * np.sqrt(3) / 2))
            )
            / output.atomistic_upscaling_factor
        )

        # Combine x and y values into a single array
        downscaled_locations_in_orig = np.dstack([i, j])

        # Flatten the original locations and downscaled locations arrays
        orig_locations_flat = orig_locations.reshape(-1, 2)
        downscaled_locations_in_orig_flat = downscaled_locations_in_orig.reshape(-1, 2)

        # Create a KDTree from the original locations
        tree = cKDTree(orig_locations_flat)

        # Find the indices of the nearest neighbors in the original locations
        _, indices = tree.query(downscaled_locations_in_orig_flat)

        # Reshape the indices to match the shape of the upscaled array
        indices = indices.reshape(
            field_size_x * output.atomistic_upscaling_factor, math.ceil(field_size_y * output.atomistic_upscaling_factor * np.sqrt(3) / 2)
        )

        # Use the indices to fill the upscaled indices array
        upscaled_indices = np.stack(np.unravel_index(indices, (field_size_x, field_size_y)), axis=-1)

        logging.info(f"UPSCALING ARRAY SHAPE {upscaled_indices.shape}\n")

        return upscaled_indices, orig_locations

    @classmethod
    def reset_q_loc_track(cls):
        cls.stat_tracker = np.zeros((sim.No_sim_img,), dtype=cls.skyrmion_dtype)

    @staticmethod
    def save_images_x_y_z(spinfield_vectors, pic_name_base):
        """
        Save images of spins in the x, y, and z directions.

        Parameters:
        spins (ndarray): Array of spin values.
        pic_name_base (str): Base name for the saved images.

        Returns:
        None
        """
        for coord in cst.coords:
            current_pic_name = pic_name_base.replace("direction", coord)
            output.save_image(spinfield_vectors[:, :, cst.coords[coord]], current_pic_name)

    @staticmethod
    def save_image(spinfield_x_or_y_or_z, pic_name):
        """
        Save the spinfield image to a file.

        Parameters:
        spinfield_x_or_y_or_z (numpy.ndarray): The spinfield data to be saved.
        pic_name (str): The name of the output image file.

        Returns:
        None
        """

        if sim.model_type == "continuum":
            plt.imsave(
                pic_name,
                spinfield_x_or_y_or_z.T[::-1, :],
                cmap=output.custom_seismic,
                vmin=-1,
                vmax=1,
            )

        if sim.model_type == "atomistic":
            plot_process = multiprocessing.Process(
                target=output.draw_hexagonal_spinfield,
                args=(
                    spinfield_x_or_y_or_z,
                    output.custom_seismic,
                    pic_name,
                ),
            )

            plot_process.start()
            output.plot_processes.append(plot_process)

            for process in list(output.plot_processes):
                if not process.is_alive():
                    process.join()
                    output.plot_processes.remove(process)

            while len([p for p in output.plot_processes if p.is_alive()]) >= output.max_parallel_processes:
                time.sleep(0.1)  # Wait for 0.1 second

    @staticmethod
    def draw_hexagonal_spinfield(orig_array, colormap, pic_dir, vmin=-1, vmax=1):

        # logging.warning(f"upscaling_indices shape: {output.upscaling_indices.shape}")
        # logging.warning(f"orig_array shape: {orig_array.shape}")
        # Calculate the valid indices for orig_array

        if orig_array.shape == (sim.x_size, sim.y_size):
            valid_x_index = output.upscaling_indices[..., 0] % orig_array.shape[0]
            valid_y_index = output.upscaling_indices[..., 1] % orig_array.shape[1]
        else:
            temp_upscaling_indices, locations = output.calculate_upscaling_array(orig_array.shape[0], orig_array.shape[1])
            valid_x_index = temp_upscaling_indices[..., 0] % orig_array.shape[0]
            valid_y_index = temp_upscaling_indices[..., 1] % orig_array.shape[1]

        # Use the valid indices to index orig_array
        upscaled_hex_array = orig_array[valid_x_index, valid_y_index]

        hex_transformed = upscaled_hex_array.T[::-1, :].copy()
        plt.imsave(
            pic_dir,
            hex_transformed,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.close()
        gc.collect()

        return upscaled_hex_array

    @staticmethod
    def current_and_potential_plot(current, potential, dest_dir, model="continuum"):

        # to convert from tecnical to physical current
        current *= -1

        if model == "continuum":
            # Create a grid of coordinates
            x = np.linspace(0, current.shape[0] - 1, current.shape[0])
            y = np.linspace(0, current.shape[1] - 1, current.shape[1])

            # Generate grid
            Y, X = np.meshgrid(y, x)

            # Plot the vector field
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create a heatmap of the potential
            c = ax.imshow(
                potential, extent=(np.amin(Y) - 0.5, np.amax(Y) + 0.5, np.amin(X) - 0.5, np.amax(X) + 0.5), origin="lower", alpha=0.5, cmap="hot"
            )

            fig.colorbar(c, ax=ax, label="Potential")

            # Overlay the current distribution
            ax.quiver(Y, X, current[:, :, 1], current[:, :, 0], pivot="middle", headwidth=4, headlength=6)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Current distribution and potential heatmap")
            ax.invert_yaxis()

            plt.savefig(dest_dir, dpi=600)

            # plt.show()

        if model == "atomistic":

            hex_array = output.draw_hexagonal_spinfield(potential, "hot", dest_dir)

            output.draw_hex_with_plt(potential, current, hex_array)

    @staticmethod
    def draw_hex_with_plt(potential, current, hex_array_upscaled):

        scalefactor = hex_array_upscaled.shape[0] / potential.shape[0]

        locations_x = output.locations[:, :, 0].flatten()
        locations_y = output.locations[:, :, 1].flatten()
        pic_offset = 1 / (scalefactor * 2)

        # define the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define the extent of the imshow plot to match the range of your locations
        extent = (
            float(locations_x.min() - pic_offset),
            float(locations_x.max() + 0.5 - pic_offset),
            float(locations_y.min() - pic_offset),
            float(locations_y.max() + 1 - pic_offset),
        )

        plt.imshow(hex_array_upscaled.T[::-1, :], cmap="seismic", alpha=0.5, extent=extent)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Hexagonal Spinfield")
        ax.set_aspect("equal")

        current_x = current[:, :, 0].flatten()
        current_y = current[:, :, 1].flatten()

        # Overlay the current distribution with 0.5 opacity
        Q = ax.quiver(
            locations_x,
            locations_y,
            current_x,
            current_y,
            pivot="middle",
            alpha=0,
            headwidth=4,
            headlength=6,
            scale=None,
            scale_units="xy",
        )

        Q._init()
        assert isinstance(Q.scale, float)
        assert isinstance(Q.width, float)

        logging.warning(f"Q.scale: {Q.scale}\n")
        logging.warning(f"Q.width: {Q.width}\n")

        Q = ax.quiver(
            locations_x,
            locations_y,
            current_x,
            current_y,
            pivot="middle",
            alpha=0.7,
            headwidth=3,
            headlength=5,
            scale=Q.scale * 3,
            scale_units="xy",
            width=Q.width * 1.2,
        )

        plt.tight_layout()
        # plt.show()
        plt.close()

    @staticmethod
    def update_postfix_dict(postfix_dict, index_t, skyrs_set, no_subprocesses):
        # Uptade the general values
        # logging.info(f"t: {output.stat_tracker[index_t]['time']:.2f} ns; index_t: {index_t}; q: {output.stat_tracker[index_t]['topological_charge']:.2f}, q_prev: {output.stat_tracker[index_t-1]['topological_charge']:.2f}")
        postfix_dict["Q"] = round(float(output.stat_tracker[index_t]["topological_charge"]))
        postfix_dict["No set"] = skyrs_set

        # Count skyrmions on the left and right
        left_count, right_count = 0, 0
        # Update for single or multiple skyrmions
        if sim.final_skyr_No <= 1:
            postfix_dict["(x, y)"] = (
                round(float(output.stat_tracker[index_t]["x0"]), 4),
                round(float(output.stat_tracker[index_t]["y0"]), 4),
            )
            # print("updating this shit")
            # print(output.stat_tracker[index_t]["r1"])
            postfix_dict["r"] = f'{output.stat_tracker[index_t]["r1"]:.2f} nm'
            postfix_dict["w"] = f'{output.stat_tracker[index_t]["w1"]:.2f} nm'
            # print(f"tracker ist da{output.q_location_tracker[index_t]['r1']}")
        else:
            # Define the boundary for left and right side
            boundary_x = sim.x_size / 2 + 70

            for i in range(min(abs(postfix_dict["Q"]), sim.final_skyr_No)):
                x_coord = output.stat_tracker[index_t][f"x{i}"]
                if x_coord <= boundary_x:
                    left_count += 1
                else:
                    right_count += 1

            # Update the postfix dictionary with the counts
            postfix_dict["L"] = left_count
            postfix_dict["R"] = right_count
        if sim.model_type == "atomistic":
            postfix_dict["sub_no"] = no_subprocesses
        if sim.sim_type == "wall_ret_test_close" or sim.sim_type == "wall_ret_test_far":
            postfix_dict["error"] = output.stat_tracker[index_t]["error"]

        return postfix_dict, left_count, right_count

    @classmethod
    def signal_handler(cls, sig, frame):
        """
        Signal handler for the SIGINT signal (Ctrl+C).
        """
        cls.ctrl_c_counter += 1

        if not np.any(GPU.spins_evolved):
            raise KeyboardInterrupt

        if cls.ctrl_c_counter == 1:
            logging.warning("first CTRL + C, finishing this step, then converting to Video and plotting up to here...")

        if cls.ctrl_c_counter > 1:
            logging.warning("second CTRL + C, shutting down...")
            raise KeyboardInterrupt


def arg_parser():
    # Define a list of acceptable simulation types

    # SIMULATION TYPES: "skyrmion_creation" or "wall_retention" or "wall_ret_test_close" or "wall_ret_test_far" or
    # SIMULATION TYPES: "angled_vs_on_edge" or "x_current" or "x_current_SkH_test" or "pinning_tests" or "ReLU" or "ReLU_larger_beta"
    acceptable_sim_types = [
        "skyrmion_creation",
        "wall_retention",
        "wall_ret_test",
        "wall_ret_test_close",
        "wall_ret_test_far",
        "angled_vs_on_edge",
        "x_current",
        "x_current_SkH_test",
        "pinning_tests",
        "ReLU",
        "ReLU_larger_beta",
    ]

    parser = ap.ArgumentParser(description=f"run one of the following simulation types: {acceptable_sim_types}")

    parser.add_argument(
        "--sim_type",
        type=str,
        default="x_current",
        choices=acceptable_sim_types,
        help=f"the simulation type to be run, one of {acceptable_sim_types}",
    )

    # parse the arguments
    return parser.parse_args()


def simulate(sim_no, angle, v_s_fac):
    """
    Simulates the spin field evolution over time using CUDA.

    Args:
        simulation_no (int): The number of the simulation.

    Returns:
        None
    """

    # Anzahl der bereits hinzugefuegten Skyrmionen
    skyr_counter = 0

    # initialize the spinfield
    spins = spin.initialize_spinfield()

    # if var B_field exists:
    cst.B_ext = float(cst.B_fields[sim_no])
    logging.warning(f"current B field: {cst.B_ext}") if sim.sim_type == "ReLU_changed_capacity" else None

    # activate the beta that is currently in use
    cst.beta = float(cst.betas[sim_no])
    logging.warning(f"CURRENT BETA: {cst.beta}") if sim.sim_type == "x_current_SkH_test" else None

    # compile the kernel
    mod, tex = sim.compile_kernel()

    # get the fitting name(s) of the kernel function(s)
    numerical_steps, avgStep, q_topo = sim.get_kernel_functions(mod)

    # picture before the relaxation
    sample_dir = f"{output.dest}/sample_{sim_no+1}_{angle}_deg_{v_s_fac}_v_s_fac"
    pic_dir = f"{sample_dir}/z_before_relaxation.png"
    output.save_image(spins[:, :, 2], pic_dir)

    # relax the spinfield without a skyrmion
    relaxed_init_spins = spin.relax_but_hold_skyr(np.copy(spins), numerical_steps, skyr_counter)
    pic_dir = f"{sample_dir}/relaxed_z_before_skyr.png"
    output.save_image(relaxed_init_spins[:, :, 2], pic_dir)

    # save to relaxed_init npy
    np.save(f"{sample_dir}/relaxed_init_spins.npy", relaxed_init_spins)

    # calculate the topological charge of the relaxed spinfield
    q_topo(GPU.spins_id, GPU.mask_id, GPU.q_topo_id, block=(GPU.block_dim_x, GPU.block_dim_y, 1), grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1))

    # fetch the results from the GPU
    q_temp = np.empty(sim.x_size * sim.y_size, dtype=np.float32)
    cuda.memcpy_dtoh(q_temp, GPU.q_topo_id)

    # sum up the results
    q_init = np.sum(q_temp) / (2 * np.pi)

    # show the initial topological charge without a skyrmion
    logging.info(f"Q_init (no skyr): {q_init}")

    # set the postfix dict for the progressbar
    if sim.final_skyr_No <= 1:
        postfix_dict = {
            "(x, y)": (np.float32(0), np.float32(0)),
            "No set": skyr_counter,
            "Q": np.float32(0),
            "r": None,
            "w": None,
        }  # mit "Max fetched_field": spin.ff_max[-1], \  falls noetig

    else:
        postfix_dict = {
            "L": np.float32(0),
            "R": np.float32(0),
            "No set": skyr_counter,
            "Q": np.float32(0),
        }  # mit "Max fetched_field": spin.ff_max[-1], \  falls noetig

    if sim.model_type == "atomistic":
        postfix_dict["sub_no"] = np.float32(0)

    if sim.sim_type == "wall_ret_test_close" or sim.sim_type == "wall_ret_test_far":
        postfix_dict["error"] = np.empty(2, dtype=np.float32)
        start_v_s_x_y_deletion_index = 0
        error_streak_counter = 0
        t_one_pos = 0
        skyr_elims = 0
        reverts = 0
        lr_adjustment = 1
        smallest_error_yet = 1000
        reset = True
        learning_rate = np.array([1, 1])

    no_skyr_counter = 0

    circular_spinfield_buffer = collections.deque(maxlen=sim.len_circ_buffer)
    circular_spinfield_buffer.append(relaxed_init_spins)

    # write the relaxed init spins to evolved spins
    GPU.spins_evolved = relaxed_init_spins.copy()

    # error avoidance for positioning part:
    v_s_x = 0
    v_s_y = 0
    delta_r_native = 0
    temp_last_skyr_spinfield = last_skyr_spinfield = GPU.spins_evolved.copy()
    skyr_elims = 0
    index_now = 0
    start_v_s_x_y_deletion_index = 0
    error_streak_counter = 0
    v_s = np.array([])
    t_one_pos = 0
    smallest_error_yet = 1e10
    del_x_by_v_s = 1e10 * np.ones(2)
    learning_rate = 1 * np.ones(2)
    lr_adjustment = 0.1
    reverts = 0
    reset = False
    t = 0

    # eigentliche Simulationsschleife
    with tqdm(
        total=len(sim.t_pics),
        desc="Skyr_Sim",
        unit="pic",
        # unit_scale=sim.steps_per_pic,
    ) as pbar:
        # every timestep (picture) in the simulation
        for index_t, t in enumerate(sim.t_pics):

            if not output.ctrl_c_counter == 0:
                break

            set_skyr_threashold = sim.every__pic_set_skyr * skyr_counter

            # Addiere ein Skyrmion falls noetig
            if index_t >= set_skyr_threashold and skyr_counter < sim.final_skyr_No:
                logging.info(f"Adding Skyrmion No. {skyr_counter + 1} at {t:.2g}ns")
                if sim.sim_type == "skyrmion_creation" or sim.sim_type == "antiferromagnet_simulation":
                    spin.skyr = GPU.spins_evolved.copy()
                    spin.skyr[:, :, 2] *= -1

                # kalkuliere das neue Spinfield
                GPU.spins_evolved = spin.set_skyr(GPU.spins_evolved, spin.skyr, sim.skyr_set_x, sim.skyr_set_y)

                temp_last_skyr_spinfield = GPU.spins_evolved.copy()
                last_skyr_spinfield = GPU.spins_evolved.copy()

                # kopiere das neue Spinfield auf die GPU
                cuda.memcpy_htod(GPU.spins_id, GPU.spins_evolved)

                # setze den Counter einen hoch
                skyr_counter += 1

                # halte das Skyrmion fest und lasse das System relaxen
                relaxed_spins = spin.relax_but_hold_skyr(GPU.spins_evolved, numerical_steps, skyr_counter)

                # make border black
                custom_relaxed_spins_z = relaxed_spins[:, :, 2].copy()

                # Replace 0s with NaN
                custom_relaxed_spins_z[custom_relaxed_spins_z == 0] = np.nan

                # save the relaxed spinfield with the new skyrmion
                pic_dir = f"{sample_dir}/spinfield_t_{t - sim.t_pics[0] + sim.dt:011.6f}.png"
                output.save_image(custom_relaxed_spins_z, pic_dir)

                # SAVE THE NPY FILE IF WANTED
                if sim.save_npys:
                    npy_output_dir = f"{sample_dir}/data/Spins_at_t_{t - sim.t_pics[0] + sim.dt:011.6f}.npy"
                    np.save(npy_output_dir, relaxed_spins.astype(np.float16))

                cuda.Context.synchronize()

            # iterate one timestep of numeric calculations
            for _ in range(int(sim.steps_per_pic / sim.steps_per_avg)):

                # add z component to average calculator
                avgStep(GPU.spins_id, GPU.avgTex_id, block=(GPU.block_dim_x, GPU.block_dim_y, 1), grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1))

                # do n numerical steps
                for _ in range(sim.steps_per_avg):
                    GPU.perform_numerical_steps(
                        numerical_steps,
                    )
                cuda.Context.synchronize()

            # pull spinfield from GPU to spins_evolved
            cuda.memcpy_dtoh(GPU.spins_evolved, GPU.spins_id)

            # setzte die Progessbar auf den aktuellen Stand
            pbar.set_description(f"Sim at {t:.4g}ns")
            pbar.update(1)

            # make border black
            custom_view_spins_z = GPU.spins_evolved[:, :, 2].copy()

            # Replace 0s with NaN
            custom_view_spins_z[custom_view_spins_z == 0] = np.nan

            # SAVE THE Z IMAGE AT t=t1
            pic_dir = f"{sample_dir}/spinfield_t_{t:011.6f}.png"
            output.save_image(custom_view_spins_z, pic_dir)

            # SAVE THE NPY FILE IF WANTED
            if sim.save_npys:
                npy_output_dir = f"{sample_dir}/data/Spins_at_t_{t:011.6f}.npy"
                np.save(npy_output_dir, GPU.spins_evolved.astype(np.float16))

            # calculate topological charge with kernel
            q_topo(GPU.spins_id, GPU.mask_id, GPU.q_topo_id, block=(GPU.block_dim_x, GPU.block_dim_y, 1), grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1))

            # fetch the results from the GPU
            cuda.memcpy_dtoh(q_temp, GPU.q_topo_id)

            # sum up the results
            q_sum = np.sum(q_temp) / (2 * np.pi)

            # helping array of mz values minus original mz values to find the skyrmions center and radius
            tracker_array = GPU.spins_evolved[:, :, 2] - relaxed_init_spins[:, :, 2]
            tracker_array[tracker_array > -0.01] = 0

            output.stat_tracker[index_t]["time"] = t
            q = output.stat_tracker[index_t]["topological_charge"] = q_sum - q_init

            # track radius and ww of skyrmion
            if round(q, 2) != 0:
                if sim.final_skyr_No <= 1:
                    center = spin.find_skyr_center(tracker_array)

                    # cuda.Context.synchronize()
                    output.stat_tracker[index_t]["x0"] = center[0]
                    output.stat_tracker[index_t]["y0"] = center[1]

                    if sim.track_radius:
                        output.stat_tracker[index_t]["r1"], output.stat_tracker[index_t]["w1"] = spin.find_skyr_params(
                            tracker_array + 1, tuple(center)
                        )
                    else:
                        output.stat_tracker[index_t]["r1"] = None
                        output.stat_tracker[index_t]["w1"] = None
                else:
                    # Find skyrmion centers and save their coordinates
                    centers = np.array(spin.find_skyr_center(tracker_array))  # Use the method you define for finding centers

                    if centers.shape[0] >= sim.final_skyr_No:
                        logging.warning(f"Too many skyrmion centers found at {t:011.6f} ns, breaking loop")
                        break
                    for i, center in enumerate(centers):
                        output.stat_tracker[index_t][f"x{i}"] = center[0]
                        output.stat_tracker[index_t][f"y{i}"] = center[1]
            elif sim.final_skyr_No <= 1:
                output.stat_tracker[index_t]["r1"] = None
                output.stat_tracker[index_t]["w1"] = None

            if sim.v_s_active:
                if sim.v_s_to_wall:
                    if output.stat_tracker[index_t]["x0"] > sim.x_threashold and abs(sim.v_s[0, 0, 0]) > 0:
                        logging.warning(f"v_s set to 0 at {t:011.6f} ns")

                        # set v_s to 0
                        sim.v_s = np.zeros((sim.x_size, sim.y_size, 2)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                        logging.warning(f"Skyrmion is at the wall at {t:011.6f} ns")

                if sim.v_s_centering:
                    if 0.5 < abs(q) < 1.5:
                        x_0 = output.stat_tracker[index_t]["x0"]
                        y_0 = output.stat_tracker[index_t]["y0"]
                        del_x_by_v_s = np.array([x_0 - sim.x_size / 2, y_0 - sim.y_size / 2])

                        normalizing_factor = 2000

                        # v_s to the center
                        v_s = (-del_x_by_v_s / np.array([sim.x_size, sim.y_size])) * normalizing_factor * sim.v_s_factor * v_s_fac * -0.05  # in m/s
                        logging.info(f"v_s aktuell: {v_s}")

                        # reshape to add dimensions, tile to repeat the array
                        sim.v_s = np.tile(v_s.reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                    else:
                        sim.v_s = np.zeros((sim.x_size, sim.y_size, 2)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)

                if sim.v_s_positioning:

                    cuda.Context.synchronize()

                    # get the current x and y coordinates of the skyrmion
                    x_0 = output.stat_tracker[index_t]["x0"]
                    y_0 = output.stat_tracker[index_t]["y0"]

                    # get the movement of the skyrmion in the last timestep
                    try:
                        x_min_1 = output.stat_tracker[index_t - 1]["x0"]
                        y_min_1 = output.stat_tracker[index_t - 1]["y0"]
                        movement = np.array([x_0 - x_min_1, y_0 - y_min_1])
                    except (IndexError, KeyError) as e:
                        logging.debug(f"Could not retrieve previous position: {e}")
                        movement = np.array([0.0, 0.0])  # Default value if there is an error

                    # get v_s responsible for current movement
                    prev_v_s_x = output.stat_tracker[index_t - 1]["v_s_x"]
                    prev_v_s_y = output.stat_tracker[index_t - 1]["v_s_y"]

                    # get the error of the skyrmion position to the set position
                    error = np.array([x_0 - sim.skyr_set_x, y_0 - sim.skyr_set_y])
                    error_value = value(error)
                    output.stat_tracker[index_t]["error"] = error

                    # set the error limit
                    local_max_error = sim.max_error * min(float(value([prev_v_s_x, prev_v_s_y])) ** 0.3 * spin.r_skyr**2, 1)

                    # FIRST STEP: gather the distance that the skyrion has moved in time t by just drifting without current
                    if index_t == 0:

                        # drift distance
                        delta_r_native = np.array([x_0 - sim.skyr_set_x, y_0 - sim.skyr_set_y])
                        logging.info(f"delta_r_native: {delta_r_native}")
                        output.stat_tracker[index_t]["v_s_x"] = 0
                        output.stat_tracker[index_t]["v_s_y"] = 0

                        # set v_s to -100 to test relation to movement
                        v_s_x = -100
                        v_s_y = 0
                        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                        cuda.Context.synchronize()

                        # 2D v_s spacial info
                        v_s_strength = value([v_s_x, v_s_y])
                        v_s_angle = np.degrees(np.arctan2(v_s_y, v_s_x))
                        logging.info(f"v_s_strength, v_s_angle: {v_s_strength, v_s_angle}")

                        # reset the spinfield to relaxed_init_spins
                        cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins.copy())
                        cuda.Context.synchronize()

                        # deduce one from skyr_conter to set skyr again
                        skyr_counter -= 1

                    # index 1: with v_s from index 0: save the amount of movement compared to the v_s_factor
                    elif index_t == 1:

                        # movement factor
                        del_x_by_v_s = (np.array([x_0 - sim.skyr_set_x, y_0 - sim.skyr_set_y]) - delta_r_native) / v_s_x
                        logging.info(f"del_x_by_v_s_10: {del_x_by_v_s}")

                        # set v_s to 0 for first try
                        v_s_x = 0
                        v_s_y = 0

                        # make array of field_size_x and field_size_y out of v_s_x and v_s_y
                        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                        # 2D spacial info
                        v_s_strength = value([v_s_x, v_s_y])
                        v_s_angle = np.arctan2(v_s_y, v_s_x)
                        logging.info(f"v_s_strength, v_s_angle: {v_s_strength, v_s_angle}")

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                        cuda.Context.synchronize()

                        # reset the spinfield to relaxed_init_spins
                        GPU.spins_evolved = relaxed_init_spins.copy()
                        cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins.copy())
                        cuda.Context.synchronize()

                        # deduce one from skyr_conter to set skyr again
                        skyr_counter -= 1

                        # set the next position
                        sim.skyr_set_x = sim.distances[0]

                    # ITERATION LOOP
                    elif error_value > local_max_error:

                        # if skyrmion is deleted at wall
                        if not 0.8 < np.abs(q) < 1.2:
                            logging.warning("Skyrmion is entering the wall")

                            # logging.warning(f"Skyrmion last position seems wrong, replaced by the one before")
                            last_skyr_spinfield = temp_last_skyr_spinfield.copy()

                            # reset the learning rate cycle
                            t_one_pos = 0

                            # count the consecutive skyr eliminations in one position
                            skyr_elims += 1

                            # more then 3 eliminations at one position before reaching end
                            if skyr_elims > 3:

                                # potential next step size
                                next_step_size = (sim.distances[index_now] - sim.distances[index_now - 1]) / 10

                                if next_step_size < 0.01:
                                    logging.warning(f"{sim.skyr_set_x} is the furthest that the skyrmion is not stable anymore")
                                    output.stat_tracker[start_v_s_x_y_deletion_index:]["v_s_x"] = 0
                                    output.stat_tracker[start_v_s_x_y_deletion_index:]["v_s_y"] = 0
                                    break
                                else:
                                    logging.warning(f"Skyrmion is destroyed at {sim.skyr_set_x}, increasing the position density")

                                    # ---- get the step distance right now ----
                                    index_now = np.where(sim.distances == sim.skyr_set_x)[0][0]

                                    # set new_positions to be of higher density then before
                                    old_distances = sim.distances[:index_now]
                                    start = old_distances[-1]
                                    stop = sim.distances[-1]
                                    new_distances = np.arange(start + next_step_size, stop, next_step_size)

                                    # concatenate the old distances with the new distances
                                    sim.distances = np.concatenate((old_distances, new_distances))

                                    # set the skyr_set_x to the new position
                                    sim.skyr_set_x = int(sim.distances[index_now])

                                    logging.warning(f"skyr_set_x now: {sim.skyr_set_x}")
                                    logging.warning(f"new distances: {new_distances}")

                        # error is smaller than the smallest error yet and smaller than the max error * 100
                        if error_value < smallest_error_yet:
                            # set this as new best error --> load in as spinfield
                            logging.info(f"new best error: {error_value} setting this as starting spinfield")
                            temp_last_skyr_spinfield = last_skyr_spinfield.copy()
                            last_skyr_spinfield = GPU.spins_evolved.copy()
                            smallest_error_yet = error_value
                            logging.info(f"resetting error_streak_counter and cyclic learning rate")
                            # error_streak_counter = 0
                            t_one_pos = 0

                        # LASTLY THERE WAS A STREAK
                        if error_streak_counter >= 1:
                            learning_rate = np.array([0.1, 0.1])
                            t_one_pos = 0
                            logging.warning("Adjusting learning rate: 0.1")

                        # v_s has non 0 component(s)
                        elif np.any(v_s != 0):
                            learning_rate = spin.calculate_learning_rate(t_one_pos)
                            # learning_rate = np.array([0.8, 0.8])

                        # CALCULATE NEW V_S
                        v_s_x = prev_v_s_x - (error[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                        v_s_y = prev_v_s_y - (error[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment  #! del_x_by_v_s[0] war vorher

                        logging.info(
                            f"at t= {t:.6g} v_s_x, v_s_y, error[0], error[1], learning_rate[0]: {v_s_x, v_s_y, error[0], error[1], learning_rate[0]}"
                        )

                        # if the skyrmion has just been eliminated at the edge
                        if t_one_pos == 0 and skyr_elims > 0:
                            v_s_x /= 2

                        # log the new v_s
                        output.stat_tracker[index_t]["v_s_x"] = v_s_x
                        output.stat_tracker[index_t]["v_s_y"] = v_s_y

                        # make array of field_size_x and field_size_y out of v_s_x and v_s_y
                        v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                        cuda.Context.synchronize()

                        # if too many pictures have passed revert to the last_skyr_spinfield before this one
                        if t_one_pos > sim.No_sim_img / 20:
                            if reverts < 2:
                                logging.warning(f"Skyrmion last position seems wrong, replaced by the one before")
                                last_skyr_spinfield = temp_last_skyr_spinfield.copy()
                                # adjust the learning rate more slowly
                                lr_adjustment *= 0.3
                                t_one_pos = 0
                                reverts += 1
                            else:
                                logging.warning(f"Skyrmion last position seems wrong AGAIN, loop broken")
                                break

                        # reset the spinfield and place skyrmion at location x, y
                        cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
                        cuda.Context.synchronize()

                        # increment t_one_pos, reset the error_streak_counter
                        error_streak_counter = 0
                        t_one_pos += 1

                    # WHEN ERROR IS IN RANGE
                    else:

                        # FINAL POSITION IS NOT REACHED
                        if not error_streak_counter >= sim.cons_reach_threashold:
                            logging.warning(f"{error_streak_counter + 1} reaches at X={sim.skyr_set_x} with (vsx, vsy): ({v_s_x}, {v_s_y})")

                            # error is smaller than the smallest error yet and smaller than the max error * 10
                            if error_value < smallest_error_yet and error_value < local_max_error * 10:
                                # set this as new best error --> load in as spinfield
                                logging.info(f"new best error: {error_value} setting this as starting spinfield")
                                temp_last_skyr_spinfield = last_skyr_spinfield.copy()
                                last_skyr_spinfield = GPU.spins_evolved.copy()
                                smallest_error_yet = error_value
                                logging.info(f"resetting error_streak_counter and cyclic learning rate")
                                # error_streak_counter = 0
                                t_one_pos = 0

                            # NOT 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED
                            if reset:
                                # big learning rate
                                learning_rate = np.array([0.1, 0.1])

                                # Rely on error to calculate new v_s
                                v_s_x = prev_v_s_x - (error[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                                v_s_y = prev_v_s_y - (error[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment

                                # make array from v_s_x and v_s_y
                                v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                                # copy v_s array to GPU
                                GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                                tex.set_array(GPU.cuda_v_s)

                                # reset the spinfield and place skyrmion at location x, y
                                cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
                                cuda.Context.synchronize()
                            else:
                                # small learning rate
                                learning_rate = np.array([0.1, 0.1])

                                # Rely on movement to calculate new v_s
                                v_s_x = prev_v_s_x - (movement[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                                v_s_y = prev_v_s_y - (movement[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment

                                # make array of field_size_x and field_size_y out of v_s_x and v_s_y
                                v_s = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                            # track the new v_s
                            output.stat_tracker[index_t]["v_s_x"] = v_s_x
                            output.stat_tracker[index_t]["v_s_y"] = v_s_y

                            # reset the spinfield and place skyrmion at location x, y
                            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)

                            # increment the consecutive_reaches
                            error_streak_counter += 1
                            t_one_pos += 1

                        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the first time
                        elif error_streak_counter >= sim.cons_reach_threashold and reset:
                            logging.warning(f"POTENTIAL V_S REACHED")

                            # reset the spinfield and place skyrmion at location x, y
                            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
                            cuda.Context.synchronize()

                            # set counters
                            reset = False
                            error_streak_counter = 0
                            t_one_pos = 0

                        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the second time
                        elif error_streak_counter >= sim.cons_reach_threashold and not reset:

                            # angle of v_s
                            theta_deg = np.degrees(np.arctan(v_s_y / v_s_x))
                            logging.warning(f"Skyrmion stays at X={sim.skyr_set_x} with (vsx, vsy): ({v_s_x}, {v_s_y})")
                            logging.warning(f"angle at {t:011.6f} ns: {theta_deg}")
                            logging.warning(f"error at {t:011.6f} ns: {output.stat_tracker[index_t]['error']}")
                            logging.warning(f"max_error: {local_max_error}")

                            # CALCULATE FINAL V_S
                            v_s_x_last_n = output.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_x"]
                            v_s_x_avg = np.average(v_s_x_last_n)
                            v_s_y_last_n = output.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_y"]
                            v_s_y_avg = np.average(v_s_y_last_n)
                            r_last_n = output.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["r1"]
                            r_avg = np.average(r_last_n)
                            logging.info(f"vsx_avg: {v_s_x_avg}")
                            logging.info(f"vsy_avg: {v_s_y_avg}")
                            logging.info(f"last 5 vsx: {v_s_x_last_n}")
                            logging.info(f"last 5 vsy: {v_s_y_last_n}")

                            # track the final v_s and r
                            output.stat_tracker[index_t]["v_s_x"] = v_s_x_avg
                            output.stat_tracker[index_t]["v_s_y"] = v_s_y_avg
                            output.stat_tracker[index_t]["r1"] = r_avg

                            # reset the values of output.stat_tracker before index_t
                            output.stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_x"] = 0
                            output.stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_y"] = 0

                            # set counters
                            reset = True
                            error_streak_counter = 0
                            skyr_elims = 0
                            lr_adjustment = 1
                            t_one_pos = 0

                            # NEW POSITION AVAILABLE
                            if sim.skyr_set_x < sim.distances[-1]:

                                # get the new position
                                index_now = np.where(sim.distances == sim.skyr_set_x)[0][0]
                                sim.skyr_set_x = int(sim.distances[index_now + 1].item())
                                logging.warning(f"position {index_now + 1} of {len(sim.distances)} reached")
                                logging.warning(f"NEW X: {sim.skyr_set_x}")

                                # set v_s to 0
                                spin.update_current(v_s_sample_factor=0, bottom_angle=0)

                                # copy v_s array to GPU
                                GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                                tex.set_array(GPU.cuda_v_s)
                                cuda.Context.synchronize()

                                # afterwards set the deletion index to the current index + 1
                                start_v_s_x_y_deletion_index = index_t + 1

                                # do not change vs simply track vsx and vsy
                                output.stat_tracker[index_t]["v_s_x"] = v_s_x
                                output.stat_tracker[index_t]["v_s_y"] = v_s_y

                                # No eliminations yet
                                if skyr_elims == 0:

                                    # reset the spinfield and place skyrmion at location x, y
                                    skyr_counter -= 1
                                    cuda.memcpy_htod(GPU.spins_id, relaxed_init_spins)
                                    cuda.Context.synchronize()

                                # reset the error_counter
                                smallest_error_yet = 1000

                            else:
                                logging.warning("Skyrmion is at the wall")
                                break

            (postfix_dict, left_c, right_c) = output.update_postfix_dict(postfix_dict, index_t, skyr_counter, output.count_open_fds())

            if sim.final_skyr_No > 1:
                output.stat_tracker[index_t]["left_count"] = left_c
                output.stat_tracker[index_t]["right_count"] = right_c

            pbar.set_postfix(postfix_dict)

            circular_spinfield_buffer.append(GPU.spins_evolved.copy())
            if index_t > 0 and sim.check_variance:
                variance = np.max(np.var(np.array(circular_spinfield_buffer, dtype=np.float32), axis=0))
                if variance < sim.critical_variance and skyr_counter == sim.final_skyr_No:
                    logging.warning("Spins are not moving anymore")
                    break
            if index_t > 0 and sim.check_skyrmion_presence and sim.final_skyr_No == 1:
                radius = output.stat_tracker[index_t]["r1"]
                ww = output.stat_tracker[index_t]["w1"]
                if 0.01 < radius < 1000 and 0.01 < ww < 1000:
                    no_skyr_counter = 0
                else:
                    if no_skyr_counter > 5:
                        logging.warning("no sensible Skyrmion is present anymore")
                        no_skyr_counter = 0
                        break
                    else:
                        no_skyr_counter += 1

    # Berechnung der topologischen ladung am Ende der Simulation
    q_topo(GPU.spins_id, GPU.mask_id, GPU.q_topo_id, block=(GPU.block_dim_x, GPU.block_dim_y, 1), grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1))

    # saving the last spins in x, y and z dir
    final_pic_base_dir = f"{sample_dir}/direction_spinfield_end_t_{t:011.6f}.png"
    output.save_images_x_y_z(GPU.spins_evolved, final_pic_base_dir)

    # fetch the results from the GPU
    cuda.memcpy_dtoh(q_temp, GPU.q_topo_id)

    # sum up the results
    q_end = np.sum(q_temp) / (2 * np.pi)

    # ziehen der Spinmatrix auf die CPU
    cuda.memcpy_dtoh(GPU.spins_evolved, GPU.spins_id)

    # Speicherung der Spins am Ende der Simulation als npy Datei
    if sim.save_npy_end:
        file = f"{sample_dir}/Spins_at_end.npy"
        np.save(file, GPU.spins_evolved)

    # Saving the location tracker and the topological charge tracker
    np.save(f"{sample_dir}/traj_q.npy", output.stat_tracker)

    # ziehen von zeitlich gemittelten Spins auf die CPU in das neue Array avgImg
    avgImg = np.empty_like(GPU.avgTex)
    cuda.memcpy_dtoh(avgImg, GPU.avgTex_id)

    # Normierung von avgImg
    avgImg /= np.max(np.abs(avgImg))

    # Speichern der zeitlich gemittelten Spins
    avg_pic_dir = f"{sample_dir}/avg_z_component.png"
    output.save_image(avgImg, avg_pic_dir)

    if sim.model_type == "atomistic":
        for process in output.plot_processes:
            process.join()

    GPU.free_GPU_memory()

    return q_init, q_end


def main(sim_type="x_current"):
    """
    Main function for the skyrmion simulation.

    This function performs the skyrmion simulation for multiple samples. It creates the necessary folder structure,
    applies bottom angles and v_s factors, saves images and npy files, executes calculations, updates masks and currents,
    transfers variables to the GPU, simulates the skyrmions, creates videos and plots, and resets the q_location_tracker.


    Parameters:
        None

    Returns:
        None
    """

    signal.signal(signal.SIGINT, output.signal_handler)

    # Laden der Variablen mittels der __init__ method der Konstantenklasse und Simulationsklasse
    cst()
    sim(sim_type)
    spin()
    GPU()
    output()

    for sample in range(sim.samples):
        logging.info(f"SAMPLE: {sample + 1}/{sim.samples}\n")

        # select the relevant bottom angle from the angles array
        bottom_angle = sim.bottom_angles[math.floor((sample) / sim.v_s_factors.shape[0])]
        logging.info(f"bottom_angle: {bottom_angle}") if sim.apply_bottom_angle else None

        # select the relevant v_s_factor from the v_s_factors array
        v_s_sample_factor = sim.v_s_factors[(sample) % sim.v_s_factors.shape[0]]
        logging.info(f"v_s_sample_factor: {v_s_sample_factor}\n")

        # Folder creation for the current sample
        sample_dir = f"{output.dest}/sample_{sample+1}_{bottom_angle}_deg_{v_s_sample_factor}_v_s_fac"
        os.makedirs(sample_dir)

        # create the folder for npy files if necessary
        os.makedirs(f"{sample_dir}/data") if sim.save_npys else None

        # save skyrmion image
        skyr_pic_dir = f"{sample_dir}/skyrmion_direction.png"
        output.save_images_x_y_z(spin.skyr, skyr_pic_dir)

        # create npy file with v_s_factor and bottom_angle
        npy_info_dir = f"{sample_dir}/v_s_fac_and_wall_angle.npy"
        np.save(npy_info_dir, np.array([v_s_sample_factor, bottom_angle]))

        new_mask_dir = f"{sample_dir}/racetrack.png"

        # if sample has a new bottom angle
        if sample % sim.v_s_factors.shape[0] == 0:

            # calc new racetrack mask (and current distro if necessary)
            if sim.apply_bottom_angle:
                logging.warning("NEW BOTTOM ANGLE DETECTED AND APPLIED")

                # calculate the new racetrack mask
                new_mask = spin.racetrack_bottom_angle(sim.pivot_point, bottom_angle)

                # save the new racetrack mask
                plt.imsave(new_mask_dir, new_mask.T[::-1, :], cmap="gray", vmin=0, vmax=1)

                if sim.model_type == "atomistic":
                    new_atom_mask_dir = f"{sample_dir}/atomistic_racetrack.png"
                    colormap = "gray"
                    logging.warning(f"drawing hexagonal mask to {new_atom_mask_dir}")
                    output.draw_hexagonal_spinfield(new_mask, colormap, new_atom_mask_dir, vmin=0)

            # copy the racetrack mask from the previous sample
            else:
                shutil.copy(sim.mask_dir, new_mask_dir)
                if sim.model_type == "atomistic":
                    atom_mask_dir = f"{sample_dir}/atomistic_racetrack.png"
                    colormap = "gray"
                    logging.warning(f"drawing hexagonal mask to {atom_mask_dir}")
                    output.draw_hexagonal_spinfield(sim.mask, colormap, atom_mask_dir, vmin=0)

            # execute the current calculation for the specific sample if necessary
            if sim.v_s_dynamic and sim.v_s_active:
                logging.info(f"CURRENT CALCULATION")

                # executing the Neumann problem solving for the CURRENT CALCULATION
                potential, current = cc.solve_neumann_problem(mask_file_dir=new_mask_dir, steps=sim.cc_steps, model=sim.model_type)

                # save the current to sample dest
                logging.info(f"current calculation done, saving npy\n")
                np.save(f"{sample_dir}/current.npy", current)

                # make a current and potential plot
                logging.info(f"saving npy done, making current and potential plot\n")
                output.current_and_potential_plot(current, potential, f"{sample_dir}/current_and_potential.png", model=sim.model_type)

            else:
                sim.v_s = spin.set_constant_v_s(v_s_sample_factor, bottom_angle)

        else:
            try:
                # copy the racetrack mask from the previous sample
                prev_bottom_angle = sim.bottom_angles[math.floor((sample - 1) / sim.v_s_factors.shape[0])]
                prev_v_s_sample_factor = sim.v_s_factors[(sample - 1) % sim.v_s_factors.shape[0]]
                racetrack_source_dir = f"{output.dest}/sample_{sample}_{prev_bottom_angle}_deg_{prev_v_s_sample_factor}_v_s_fac/racetrack.png"
                shutil.copy(racetrack_source_dir, new_mask_dir)

                if sim.v_s_dynamic and sim.v_s_active:

                    # copy the current distribution from the previous sample
                    current_source_dir = f"{output.dest}/sample_{sample}_{prev_bottom_angle}_deg_{prev_v_s_sample_factor}_v_s_fac/current.npy"
                    current_dest_dir = f"{sample_dir}/current.npy"
                    shutil.copy(current_source_dir, current_dest_dir)
            except:
                try:
                    prev_prev_bottom_angle = sim.bottom_angles[math.floor((sample - 2) / sim.v_s_factors.shape[0])]
                    prev_prev_v_s_sample_factor = sim.v_s_factors[(sample - 2) % sim.v_s_factors.shape[0]]
                    racetrack_source_dir = (
                        f"{output.dest}/sample_{sample - 1}_{prev_prev_bottom_angle}_deg_{prev_prev_v_s_sample_factor}_v_s_fac/racetrack.png"
                    )
                    shutil.copy(racetrack_source_dir, new_mask_dir)

                    if sim.v_s_dynamic and sim.v_s_active:

                        # copy the current distribution from the previous sample
                        current_source_dir = (
                            f"{output.dest}/sample_{sample - 1}_{prev_prev_bottom_angle}_deg_{prev_prev_v_s_sample_factor}_v_s_fac/current.npy"
                        )
                        current_dest_dir = f"{sample_dir}/current.npy"
                        shutil.copy(current_source_dir, current_dest_dir)
                except:
                    logging.error("No 2 previous samples found, no racetrack mask copied")

        # update the mask and current
        spin.update_current_and_mask(bottom_angle, v_s_sample_factor, mask_dir=f"{sample_dir}/racetrack.png", j_dir=f"{sample_dir}/current.npy")

        # Simulation der Skyrmionen
        q_init, q_end = simulate(sample, bottom_angle, v_s_sample_factor)

        # logging.info the topological charge at the beginning and at the end
        logging.info(f"q_init: {q_init}")
        logging.info(f"q_end: {q_end}")
        logging.info(f"{sample+1}/{sim.samples} samples simulated")

        # create a video from the pics
        try:
            logging.info(f"CREATING VIDEO FOR SAMPLE {sample+1}")

            if sim.model_type == "continuum":
                os.system(
                    f"ffmpeg -framerate 5 -pattern_type glob -i '{sample_dir}/spinfield*.png' "
                    f"-vf 'scale=-1:1080:flags=neighbor' -vcodec libx264 -crf 18 -pix_fmt yuv420p {sample_dir}/simulation.mp4"
                )

            elif sim.model_type == "atomistic":
                # Full video
                os.system(
                    f"ffmpeg -framerate 5 -pattern_type glob -i '{sample_dir}/spinfield*.png' "
                    f"-vf 'scale=trunc(oh*a/2)*2:1080:flags=neighbor,setsar=1' -vcodec libx264 -crf 18 -pix_fmt yuv420p {sample_dir}/simulation.mp4"
                )

                # Zoomed in video
                outputAspectRatio = 16 / 9
                zoom_factor = 3
                os.system(
                    f"ffmpeg -framerate 5 -pattern_type glob -i '{sample_dir}/spinfield*.png' "
                    f"-vf 'crop=iw/{zoom_factor}*({outputAspectRatio}/(iw/ih)):ih/{zoom_factor},scale=1920:1080:flags=neighbor,setsar=1' "
                    f"-vcodec libx264 -crf 18 -pix_fmt yuv420p {sample_dir}/simulation_zoomed_{zoom_factor}x.mp4"
                )

            if not sim.save_pics:
                logging.info("deleting images")

                for filename in glob.glob(f"{sample_dir}/**/spinfield_t_*.png", recursive=True):
                    os.remove(filename)

        except:
            logging.warning("ffmpeg module not loaded, no video created")

            if not sim.save_pics:
                logging.warning("no images deleted eventhough save_pics is False")

        try:
            if sim.sim_type == "wall_retention" or sim.sim_type == "wall_retention_new" or sim.sim_type == "wall_retention_reverse_beta":
                if sim.final_skyr_No <= 1:
                    logging.info(f"CREATING WALL RETENTION PLOT FOR SAMPLE {sample+1}")
                    make_wall_retention_plot(fetch_dir=sample_dir, dest_dir=sample_dir, dest_file=f"plot_wall_retention_{sample + 1}.png")

            if sim.sim_type == "wall_ret_test_close" or sim.sim_type == "wall_ret_test_far":
                logging.info(f"CREATING WALL current vs distance plot {sample+1}")
                current_vs_distance_plot(fetch_dir=sample_dir, dest_dir=sample_dir, dest_file=f"current_vs_distance_{sample + 1}.png")

            if sim.final_skyr_No > 1:
                logging.info(f"CREATING INPUT OUTPUT PLOT FOR SAMPLE {sample+1}")
                pop_times = create_input_output_plot(fetchpath=sample_dir, destpath=sample_dir, dest_file=f"plot_input_output_{sim.sim_type}.png")

                # save pop times to npy file
                np.save(f"{sample_dir}/pop_times.npy", pop_times)

            elif sim.sim_type == "skyrmion_creation":
                logging.info(f"CREATING Q, r vs time plot {sample+1}")
                create_q_r_vs_time_plot(fetch_dir=sample_dir, dest_dir=sample_dir, dest_file=f"plot_q_r_w_vs_time_{sim.sim_type}.png")

            else:
                logging.info(f"CREATING SIMPLE TRAJECTORY TRACE PLOT FOR SAMPLE {sample+1}")
                trajectory_trace(fetch_folder_name=sample_dir, dest_dir=sample_dir, dest_file=f"plot_trajectory_{sample + 1}.png")
        except Exception as e:
            logging.warning(f"One of the plot creations failed: {e}")

        # reset the q_location_tracker
        output.reset_q_loc_track()

    # delete the current_temp_folder if one is found
    if os.path.exists(f"current_temp/"):
        shutil.rmtree(f"current_temp")


if __name__ == "__main__":

    args = arg_parser()

    main(sim_type=args.sim_type)
