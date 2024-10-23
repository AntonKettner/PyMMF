# native imports
import logging  # enabling display of logging.info messages
import math

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden
from PIL import Image  # Das csv - Paket wird zum Speichern der CSV - Datei benutzt
from scipy.optimize import curve_fit

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# local imports
from constants import cst


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
    - max_error (float)             : Maximum error (wall_rep_test_new)
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
    model_type              = "atomistic"
    calculation_method      = "heun"
    boundary                = "open"
    apply_bottom_angle      = False
    bottom_angles           = np.array([0])
    v_s_factors             = np.array([25])
    pivot_point             = (250, 100)
    final_skyr_No           = 1
    t_max                   = 1
    t_relax_skyr            = 0
    t_relax_no_skyr         = 0.3
    t_circ_buffer           = 0.01
    No_sim_img              = 20
    cc_steps                = 600000
    t_last_skyr_frac        = 1
    save_pics               = True
    save_npys               = False
    save_npy_end            = True
    track_radius            = True
    check_variance          = True
    check_skyrmion_presence = True
    critical_variance       = 1e-6
    steps_per_avg           = 1
    learning_rate           = np.array([1, 1])
    smallest_error_yet      = 1000
    cons_reach_threashold   = 10

    # ----------------------------------------------------------------------------Attributes: j and skyr params ----------------------------------------------------------------------------

    # PATHS
    j_dir    = "current_temp/current.npy"
    skyr_dir = f"needed_files/skyr_{cst.skyr_name_ext}_{model_type}.npy"

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
    def __init__(cls, sim_type="x_current"):
        """
        INFO(max timesteps for each calculation method)
        - "euler": Can be used with a max time step of 0.0000033.
        - "rk4"  : Can be used with a max time step of 0.000051.
        - "heun" : Can be used with a max time step of 0.000018.
        """

        # SIMULATION TYPES: "skyrmion_creation" or "wall_repulsion" or "wall_rep_test_close" or "wall_rep_test_far" or
        #                   "angled_vs_on_edge" or "x_current" or "x_current_SkH_test" or "pinning_tests" or "ReLU" or "ReLU_larger_beta"
        cls.sim_type = sim_type

        if cls.calculation_method == "euler":
            cls.dt = 0.0000033 / 1.8
        elif cls.calculation_method == "rk4":
            cls.dt = 0.000051 / 1.8
        elif cls.calculation_method == "heun":
            cls.dt = 0.000018

        # DEFAULT VELOCITY FIELD V_S PROPERTIES
        cls.v_s_dynamic     = False
        cls.v_s_active      = True
        cls.v_s_to_wall     = False
        cls.v_s_positioning = False
        cls.v_s_factor      = 200

        # only needed for wall_rep_test_new
        cls.distances = np.array([])

        if cls.sim_type == "wall_repulsion":
            # Output dir name
            cls.fig = "Thesis_Fig_9"

            # sim_vars
            # cls.t_max      = 20
            cls.t_max      = 2
            # cls.No_sim_img = 1000
            cls.No_sim_img = 100
            cst.beta       = cst.alpha
            cst.betas      = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_100100atomic.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x    = math.floor((cst.r_skyr * 1e-9 / cst.a * 1.6))
            cls.skyr_set_y    = cls.y_size / 2  # 200 bei big

            # v_s properties
            cls.v_s_active      = False
            cls.v_s_dynamic     = False
            cls.v_s_centering   = False
            cls.check_variance  = False
            cls.v_s_to_wall     = False
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
            cls.skyr_set_x    = 450
            cls.skyr_set_y    = 160

            # v_s properties
            cls.bottom_angles      = np.linspace(3, 9, 49, endpoint=True) * -1
            cls.bottom_angles      = np.linspace(3, 9, 4, endpoint=True) * -1
            cls.v_s_factors        = np.linspace(0.5, 30.5, 31, endpoint=True) * -1
            cls.v_s_factors        = np.linspace(0.5, 30.5, 4, endpoint=True) * -1
            cls.samples            = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
            cst.betas              = np.ones((cls.samples)) * cst.alpha
            cst.B_fields           = np.ones((cls.samples)) * cst.B_ext
            cls.v_s_to_wall        = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "skyrmion_creation":
            cls.fig = "Thesis_Fig_8"

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # vary r_skyr a little bit
            cls.t_max      = 0.3
            cls.No_sim_img = 50

            # set the skyrmion parameters
            cls.final_skyr_No           = 1
            cls.r_skyr                  = 2 / cst.a * 1e-9 / 10  # r = 1.5 nm --> /10 because of the algorithm also working with already set skyrmions
            cls.check_skyrmion_presence = False
            cls.skyr_set_x              = math.floor((cls.x_size) / 2)
            cls.skyr_set_y              = math.floor((cls.y_size) / 2)

            # v_s properties
            cls.v_s_factors        = np.array([0.5])
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = True
            cls.v_s_to_wall        = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "x_current":
            cls.fig = cls.sim_type
            # sim_vars
            cls.t_max      = 0.3
            cst.beta       = cst.alpha
            cst.betas      = np.array([cst.beta], dtype=np.float32)
            cls.No_sim_img = 10

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x    = cls.x_size / 2
            cls.skyr_set_y    = cls.y_size / 2

            # v_s properties
            cls.v_s_factors        = np.array([1])
            cls.v_s_to_wall        = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.apply_bottom_angle = False

        elif cls.sim_type == "x_current_SkH_test":

            cls.fig = "Thesis_Fig_11"
            # sim_vars
            cls.t_max      = 40
            cst.betas      = np.array([0.5, 1, 2]) * cst.alpha
            # cst.beta       = cst.alpha
            cls.No_sim_img = 200

            # set the map
            cls.mask_dir = "needed_files/Mask_track_beta_vs_alpha.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            # cls.skyr_set_x    = cls.x_size / 2
            cls.skyr_set_x    = 50
            cls.skyr_set_y    = cls.y_size / 2

            cls.v_s_factors        = np.ones((3)) / cst.a / 1e9  # -> should be 10 v_s
            cls.samples            = cls.v_s_factors.shape[0]
            cls.v_s_to_wall        = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.apply_bottom_angle = False
            cst.B_fields           = np.ones((cls.samples)) * cst.B_ext

        elif cls.sim_type == "wall_rep_test_close":
            # Output dir name
            cls.fig        = "Thesis_Fig_10_close_2"
            cls.t_max      = 200
            cls.No_sim_img = 5000
            cst.beta       = cst.alpha
            cst.betas      = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x    = cls.x_size / 2
            cls.skyr_set_y    = cls.y_size / 2

            # v_s properties
            cls.v_s_factors        = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.v_s_positioning    = True
            cls.check_variance     = False

            # set distances
            dist_start    = int(cls.x_size - 4 * cst.r_skyr * (1e-9 / cst.a))
            dist_end      = int(cls.x_size - 1.5 * cst.r_skyr * (1e-9 / cst.a))
            cls.distances = np.arange(dist_start, dist_end)
            cls.distances = np.unique(cls.distances)
            # cls.distances = np.floor(np.linspace(dist_start, dist_end, 4))
            cls.max_error             = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples               = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        elif cls.sim_type == "wall_rep_test_far":
            # Output dir name
            cls.fig        = "Thesis_Fig_10_far"
            cls.t_max      = 800
            cls.No_sim_img = 5000
            cst.beta       = cst.alpha
            cst.betas      = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x    = cls.x_size / 2
            cls.skyr_set_y    = cls.y_size / 2

            # v_s properties
            cls.v_s_factors        = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.v_s_positioning    = True
            cls.check_variance     = False

            # set distances
            dist_start                = int(cls.x_size - 7 * cst.r_skyr * (1e-9 / cst.a))
            dist_end                  = int(cls.x_size - 4 * cst.r_skyr * (1e-9 / cst.a))
            cls.distances             = np.arange(dist_start, dist_end)
            cls.distances             = np.unique(cls.distances)
            cls.distances             = np.floor(np.linspace(dist_start, dist_end, 4))
            cls.max_error             = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples               = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        elif cls.sim_type == "pinning_tests":
            # CHOOSE ONE OF THOSE 4 options
            # cls.fig = "Thesis_Fig_13_1"
            # cls.mask_dir = "needed_files/Mask_track_narrowing_through.png"

            cls.fig      = "Thesis_Fig_13_2"
            cls.mask_dir = "needed_files/Mask_track_narrowing_stuck.png"

            # cls.fig = "Thesis_Fig_14_1"
            # cls.mask_dir = "needed_files/Mask_track_corner_through.png"

            # cls.fig = "Thesis_Fig_14_2"
            # cls.mask_dir = "needed_files/Mask_track_corner_stuck.png"

            cls.t_max              = 30
            cls.t_max              = 3
            cls.No_sim_img         = 200
            cls.No_sim_img         = 20
            cst.beta               = cst.alpha
            cst.betas              = np.array([cst.beta], dtype=np.float32)
            cls.skyr_set_x         = 20
            cls.skyr_set_y         = 80
            cls.bottom_angles      = np.array([0])
            cls.v_s_factors        = np.array([2])
            cls.samples            = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = True
            cls.v_s_centering      = False

        elif cls.sim_type == "ReLU":
            # FINAL RESULTS:
            cls.fig                = "Thesis_Fig_15"
            cls.mask_dir           = "needed_files/Mask_final_ReLU_simplification_bigger_11.png"
            cls.final_skyr_No      = 15
            cls.t_max              = 100
            cls.No_sim_img         = 50
            cls.skyr_set_x         = 15
            cls.skyr_set_y         = 140
            cls.v_s_factors        = np.array([1])
            cls.samples            = cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = True
            cls.v_s_centering      = False
            cls.v_s_to_wall        = False
            cst.beta               = cst.alpha / 2

        elif cls.sim_type == "ReLU_larger_beta":
            # FINAL RESULTS: 
            cls.fig                = "Thesis_Fig_19"
            cls.mask_dir           = "needed_files/Mask_final_ReLU_high_beta_modular.png"
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            cls.final_skyr_No      = 50
            cls.t_max              = 200
            cls.No_sim_img         = 1000
            cst.beta               = cst.alpha * 2
            cst.betas              = np.array([cst.beta], dtype=np.float32)
            cls.skyr_set_x         = 15
            cls.skyr_set_y         = 140
            cls.v_s_factors        = np.array([1])
            cls.samples            = cls.v_s_factors.shape[0]
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = True
            cls.v_s_centering      = False
            cls.v_s_to_wall        = False

        elif cls.sim_type == "wall_rep_test":
            # Output dir name
            cls.fig        = "wall_rep_test_middle"
            cls.t_max      = 40
            cls.No_sim_img = 1000
            cst.beta       = cst.alpha
            cst.betas      = np.array([cst.beta], dtype=np.float32)

            # set the map
            cls.mask_dir = "needed_files/Mask_track_free.png"

            # load the mask and get the size of the mask
            cls.x_size, cls.y_size = cls.load_mask(cls.mask_dir)

            # set skyrmion vars
            cls.final_skyr_No = 1
            cls.skyr_set_x    = cls.x_size / 2
            cls.skyr_set_y    = cls.y_size / 2

            # v_s properties
            cls.v_s_factors        = np.array([0])
            cls.apply_bottom_angle = False
            cls.v_s_active         = True
            cls.v_s_dynamic        = False
            cls.v_s_centering      = False
            cls.v_s_positioning    = True
            cls.check_variance     = False

            # set distances
            cls.distances             = np.array([int(cls.x_size - 3 * cst.r_skyr * (1e-9 / cst.a))])
            cls.max_error             = 0.00005
            cls.cons_reach_threashold = 10
            cls.samples               = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]

        else:
            raise ValueError("Invalid simulation type.")

        # calculated params from base params
        cls.len_circ_buffer = min(max(int(cls.t_circ_buffer * cls.No_sim_img / cls.t_max), 5), 50)
        cls.time_per_img    = cls.t_max / cls.No_sim_img
        cls.samples         = cls.bottom_angles.shape[0] * cls.v_s_factors.shape[0]
        cls.len_circ_buffer = min(max(int(cls.t_circ_buffer * cls.No_sim_img / cls.t_max), 5), 50)
        cls.time_per_img    = cls.t_max / cls.No_sim_img
        cls.t_pics          = np.linspace(0, cls.t_max, cls.No_sim_img + 1, endpoint=True)[1:]

        # Berechnung der Anzahl an Bildern, nach denen je ein Skyrmion gesetzt wird
        if not cls.final_skyr_No == 0:
            cls.every__pic_set_skyr = np.floor(cls.No_sim_img * cls.t_last_skyr_frac / cls.final_skyr_No)
        else:
            cls.every__pic_set_skyr = cls.No_sim_img + 1

        # some additional parameters where dt is needed
        cls.total_steps                    = int(cls.t_max / cls.dt)
        cls.steps_per_pic                  = int(cls.total_steps / len(cls.t_pics))
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

        # initializes the skyrmion parameters
        cls.r_skyr, cls.w_skyr = sim.find_skyr_params(cls.skyr[:, :, 2], (cls.skyr.shape[0] / 2, cls.skyr.shape[1] / 2))

    @classmethod
    def load_mask(cls, dir):
        cls.mask = np.ascontiguousarray(
            np.array(np.array(Image.open(dir), dtype=bool)[:, :, 0]).T[:, ::-1]
        )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis

        return cls.mask.shape[0], cls.mask.shape[1]

    @classmethod
    def set_constant_v_s(cls, v_s_sample_factor, angle=0):
        x_old   = -0.05 * sim.v_s_factor * v_s_sample_factor
        x_new   = np.cos(np.radians(angle)) * x_old
        y_new   = np.sin(np.radians(angle)) * x_old
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
        sim.j_dir    = j_dir
        sim.mask_dir = mask_dir

        # load mask
        sim.load_mask(mask_dir)

        # dynamic v_s
        if sim.v_s_dynamic and sim.v_s_active:

            try:
                sim.j = np.load(sim.j_dir)  # [a*nm/ps]
                logging.info("current_temp file found")
                logging.warning(f"j_shape: {sim.j.shape}")
                sim.v_s = np.ascontiguousarray(
                    (
                        sim.j
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
        sim.j_dir = j_dir

        # for dynamic v_s
        if sim.v_s_dynamic and sim.v_s_active:

            try:
                sim.j = np.load(sim.j_dir)  # [a*nm/ps]
                logging.info("current_temp file found")
                logging.warning(f"j_shape: {sim.j.shape}")
                sim.v_s = np.ascontiguousarray(
                    (
                        sim.j
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
        total_step_fn       = lambda spag: (int(cls.steps_per_pic / spag)) * (spag + 1) * len(cls.t_pics)
        revised_total_steps = total_step_fn(sa_guess)

        # Adjust the steps_per_avg_guess upwards until the actual total steps meet or exceed the required steps_total
        while revised_total_steps < cls.total_steps:
            sa_guess            += 1
            revised_total_steps  = total_step_fn(sa_guess)

        return sa_guess, revised_total_steps

    @staticmethod
    def spa_guess(x):
        # Adjust b and c based on observations.
        a     = x / 2000
        d     = x / 2000
        b     = 1e-6
        c     = -5
        value = a * np.tanh(b * x + c) + d
        return max(1, int(value))

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
        bins        = np.logspace(np.log10(1), np.log10(np.max((r))), 400)
        bin_indices = np.digitize(r, bins)
        binned_mz   = [np.mean(m_z[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(1, len(bins))]
        # remove nan values from bins and binned_mz
        bins      = bins[:-1][~np.isnan(binned_mz)]
        binned_mz = np.array(binned_mz)[~np.isnan(binned_mz)]

        # Initial guesses for R and w
        p0 = [5, 1]

        # Fit using curve_fit
        popt, _     = curve_fit(sim.skyr_profile, bins, binned_mz, p0=p0)
        skyr_params = np.array(popt) * 1e9 * cst.a  # Convert back to lattice units

        return float(skyr_params[0]), float(skyr_params[1])  # Return the fitted R value

    @staticmethod
    def skyr_profile(r, R, w):
        """
        Describes the radial profile of a skyrmion with radius R and wall width w.
        same as cos(2*arctan((sinh(R/w)/sinh(r/w))))
        as 2*arctan((sinh(R/w)/sinh(r/w))) is the angle from the center of the skyrmion to the point r
        """
        return np.tanh((r - R) / w)

    @staticmethod
    def get_kernel_functions(mod):
        """
        Retrieves the kernel functions for the skyrmion simulation based on the specified calculation method.
        Args:
            mod: The module containing the kernel functions.
        Returns:
            tuple: A tuple containing:
                - simSteps: A list of kernel functions for the simulation steps, or a single kernel function if using the Euler method.
                - avgStep : The kernel function for averaging steps.
                - q_topo  : The kernel function for calculating the topological charge.
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
        q_topo  = mod.get_function("CalQTopo")

        return simSteps, avgStep, q_topo


if __name__ == "__main__":
    print("This Code is not meant to be run directly, but imported from main.py.")
