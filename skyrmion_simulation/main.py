# PyMMF by Anton Kettner, 2024
# INFO:
# - The code is split into 5 different scripts: main, simulation, spin_operations, constants and output
# - other than the main script, the other scripts are not meant to be run on their own
# - they are imported and functions are called from the respective classes.
# - sim class                       --> the basic simulation parameters and METHODS OF THE SIMULATION
# - spin class                      --> operations on the spin field, e.g. initialization, relaxation, setting skyrmions etc.
# - cst class                       --> PHYSICAL CONSTANTS AND PARAMETERS
# - op class                        --> CONSTANTS: output location, METHODS: status bar stats, conversion to mp4, save images

# Standard library imports
import logging  # enabling display of logging.info messages

# import subprocess
import os
import signal
import sys
import glob
import shutil
import math
import collections
import argparse as ap

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden
from numpy.linalg import norm as value
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Navigate one level up from the current script's directory
parent_directory = os.path.abspath(os.path.join(__file__, "..", ".."))

# Add the parent directory to sys.path to access modules from there
sys.path.insert(0, parent_directory)

# Local application/class imports
from current_calculation import current_calculation as cc
from analysis.input_output import create_input_output_plot
from analysis.trajectory_trace_new_wall_retention import make_wall_retention_plot
from analysis.trajectory_trace_simple import trajectory_trace
from analysis.q_r_vs_time_plot import create_q_r_vs_time_plot
from analysis.angle_distance_plot import current_vs_distance_plot
from constants import cst
from simulation import sim
from spin_operations import spin
from gpu import GPU
from output import op

# logging and plt backend config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
matplotlib.use("Agg")


def compile_kernel(GPU, sim, cst):
    """
    CREATED WITH THE HELP OF COPILOT
    Compiles a CUDA kernel for skyrmion simulation with the given external field.
        ARGS:
    current_ext_field (float): The current external magnetic field value.
        RETURNS:
    tuple: A tuple containing the compiled CUDA module and the texture reference for the spin field.
        RAISES:
    FileNotFoundError: If the kernel file cannot be found.
    IOError          : If there is an error reading the kernel file.
        NOTES:
        - The function reads the CUDA kernel from a file based on the current model type and calculation method.
        - The function allocates memory on the GPU and transfers necessary data arrays to the GPU.
        - The spin field is set up as a texture reference for use in the CUDA kernel.
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
    NN_vec_id          = mod.get_global("NN_vec")[0]
    NN_pos_even_row_id = mod.get_global("NN_pos_even_row")[0]
    NN_pos_odd_row_id  = mod.get_global("NN_pos_odd_row")[0]
    DM_vec_id          = mod.get_global("DM_vec")[0]
    cuda.memcpy_htod(NN_vec_id, cst.NN_vecs)
    cuda.memcpy_htod(NN_pos_even_row_id, cst.NN_pos_even_row.astype(np.int32))
    cuda.memcpy_htod(NN_pos_odd_row_id, cst.NN_pos_odd_row.astype(np.int32))
    cuda.memcpy_htod(DM_vec_id, cst.DM_vecs)

    return mod, texref


def arg_parser():
    """
    CREATED WITH THE HELP OF COPILOT
    Parses command-line arguments for the simulation script.

    This function sets up an argument parser to handle the command-line arguments
    required to run different types of simulations. It defines a list of acceptable
    simulation types and ensures that the provided simulation type is one of the
    acceptable options.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Available simulation types:
        - "skyrmion_creation"
        - "wall_retention"
        - "wall_ret_test"
        - "wall_ret_test_close"
        - "wall_ret_test_far"
        - "angled_vs_on_edge"
        - "x_current"
        - "x_current_SkH_test"
        - "pinning_tests"
        - "ReLU"
        - "ReLU_larger_beta"
    """

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


def run_simulation(sim_no, angle, v_s_fac):
    """
    CREATED WITH THE HELP OF COPILOT
    Simulates the spin field evolution over time using CUDA.

    Args:
        sim_no (int)   : The number of the simulation.
        angle (float)  : The angle parameter for the simulation.
        v_s_fac (float): The velocity scaling factor.

    Returns:
        tuple: A tuple containing the initial and final topological charges (q_init, q_end).
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
    mod, tex = compile_kernel(GPU, sim, cst)

    # get the fitting name(s) of the kernel function(s)
    numerical_steps, avgStep, q_topo = sim.get_kernel_functions(mod)

    # picture before the relaxation
    sample_dir = f"{op.dest}/sample_{sim_no+1}_{angle}_deg_{v_s_fac}_v_s_fac"
    pic_dir    = f"{sample_dir}/z_before_relaxation.png"
    op.save_image(spins[:, :, 2], pic_dir)

    # relax the spinfield without a skyrmion
    relaxed_init_spins = spin.relax_but_hold_skyr(np.copy(spins), numerical_steps, skyr_counter)
    pic_dir            = f"{sample_dir}/relaxed_z_before_skyr.png"
    op.save_image(relaxed_init_spins[:, :, 2], pic_dir)

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
            "Q"     : np.float32(0),
            "r"     : None,
            "w"     : None,
        }  # mit "Max fetched_field": spin.ff_max[-1], \  falls noetig

    else:
        postfix_dict = {
            "L"     : np.float32(0),
            "R"     : np.float32(0),
            "No set": skyr_counter,
            "Q"     : np.float32(0),
        }  # mit "Max fetched_field": spin.ff_max[-1], \  falls noetig

    if sim.model_type == "atomistic":
        postfix_dict["sub_no"] = np.float32(0)

    if sim.sim_type == "wall_ret_test_close" or sim.sim_type == "wall_ret_test_far":
        postfix_dict["error"]        = np.empty(2, dtype=np.float32)
        start_v_s_x_y_deletion_index = 0
        error_streak_counter         = 0
        t_one_pos                    = 0
        skyr_elims                   = 0
        reverts                      = 0
        lr_adjustment                = 1
        smallest_error_yet           = 1000
        reset                        = True
        learning_rate                = np.array([1, 1])

    no_skyr_counter = 0

    circular_spinfield_buffer = collections.deque(maxlen=sim.len_circ_buffer)
    circular_spinfield_buffer.append(relaxed_init_spins)

    # write the relaxed init spins to evolved spins
    GPU.spins_evolved = relaxed_init_spins.copy()

    # error avoidance for positioning part:
    v_s_x                        = 0
    v_s_y                        = 0
    delta_r_native               = 0
    temp_last_skyr_spinfield     = last_skyr_spinfield = GPU.spins_evolved.copy()
    skyr_elims                   = 0
    index_now                    = 0
    start_v_s_x_y_deletion_index = 0
    error_streak_counter         = 0
    v_s                          = np.array([])
    t_one_pos                    = 0
    smallest_error_yet           = 1e10
    del_x_by_v_s                 = 1e10 * np.ones(2)
    learning_rate                = 1 * np.ones(2)
    lr_adjustment                = 0.1
    reverts                      = 0
    reset                        = False
    t                            = 0

    # eigentliche Simulationsschleife
    with tqdm(
        total = len(sim.t_pics),
        desc  = "Skyr_Sim",
        unit  = "pic",
        # unit_scale=sim.steps_per_pic,
    ) as pbar:
        # every timestep (picture) in the simulation
        for index_t, t in enumerate(sim.t_pics):

            if not op.ctrl_c_counter == 0:
                break

            set_skyr_threashold = sim.every__pic_set_skyr * skyr_counter

            # Addiere ein Skyrmion falls noetig
            if index_t >= set_skyr_threashold and skyr_counter < sim.final_skyr_No:
                logging.info(f"Adding Skyrmion No. {skyr_counter + 1} at {t:.2g}ns")
                if sim.sim_type == "skyrmion_creation" or sim.sim_type == "antiferromagnet_simulation":
                    sim.skyr           = GPU.spins_evolved.copy()
                    sim.skyr[:, :, 2] *= -1

                # kalkuliere das neue Spinfield
                GPU.spins_evolved = spin.set_skyr(GPU.spins_evolved, sim.skyr, sim.skyr_set_x, sim.skyr_set_y)

                temp_last_skyr_spinfield = GPU.spins_evolved.copy()
                last_skyr_spinfield      = GPU.spins_evolved.copy()

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
                op.save_image(custom_relaxed_spins_z, pic_dir)

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
            op.save_image(custom_view_spins_z, pic_dir)

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
            tracker_array                        = GPU.spins_evolved[:, :, 2] - relaxed_init_spins[:, :, 2]
            tracker_array[tracker_array > -0.01] = 0
            op.stat_tracker[index_t]["time"]     = t
            q                                    = op.stat_tracker[index_t]["topological_charge"] = q_sum - q_init

            # track radius and ww of skyrmion
            if round(q, 2) != 0:
                if sim.final_skyr_No <= 1:
                    center = spin.find_skyr_center(tracker_array)

                    # cuda.Context.synchronize()
                    op.stat_tracker[index_t]["x0"] = center[0]
                    op.stat_tracker[index_t]["y0"] = center[1]

                    if sim.track_radius:
                        op.stat_tracker[index_t]["r1"], op.stat_tracker[index_t]["w1"] = sim.find_skyr_params(tracker_array + 1, tuple(center))
                    else:
                        op.stat_tracker[index_t]["r1"] = None
                        op.stat_tracker[index_t]["w1"] = None
                else:
                    # Find skyrmion centers and save their coordinates
                    centers = np.array(spin.find_skyr_center(tracker_array))  # Use the method you define for finding centers

                    if centers.shape[0] >= sim.final_skyr_No:
                        logging.warning(f"Too many skyrmion centers found at {t:011.6f} ns, breaking loop")
                        break
                    for i, center in enumerate(centers):
                        op.stat_tracker[index_t][f"x{i}"] = center[0]
                        op.stat_tracker[index_t][f"y{i}"] = center[1]
            elif sim.final_skyr_No <= 1:
                op.stat_tracker[index_t]["r1"] = None
                op.stat_tracker[index_t]["w1"] = None

            if sim.v_s_active:
                if sim.v_s_centering:
                    if 0.5 < abs(q) < 1.5:
                        x_0          = op.stat_tracker[index_t]["x0"]
                        y_0          = op.stat_tracker[index_t]["y0"]
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
                    x_0 = op.stat_tracker[index_t]["x0"]
                    y_0 = op.stat_tracker[index_t]["y0"]

                    # get the movement of the skyrmion in the last timestep
                    try:
                        x_min_1  = op.stat_tracker[index_t - 1]["x0"]
                        y_min_1  = op.stat_tracker[index_t - 1]["y0"]
                        movement = np.array([x_0 - x_min_1, y_0 - y_min_1])
                    except (IndexError, KeyError) as e:
                        logging.debug(f"Could not retrieve previous position: {e}")
                        movement = np.array([0.0, 0.0])  # Default value if there is an error

                    # get v_s responsible for current movement
                    prev_v_s_x = op.stat_tracker[index_t - 1]["v_s_x"]
                    prev_v_s_y = op.stat_tracker[index_t - 1]["v_s_y"]

                    # get the error of the skyrmion position to the set position
                    error                             = np.array([x_0 - sim.skyr_set_x, y_0 - sim.skyr_set_y])
                    error_value                       = value(error)
                    op.stat_tracker[index_t]["error"] = error

                    # set the error limit
                    local_max_error = sim.max_error * min(float(value([prev_v_s_x, prev_v_s_y])) ** 0.3 * sim.r_skyr**2, 1)

                    # FIRST STEP: gather the distance that the skyrion has moved in time t by just drifting without current
                    if index_t == 0:

                        # drift distance
                        delta_r_native                    = np.array([x_0 - sim.skyr_set_x, y_0 - sim.skyr_set_y])
                        op.stat_tracker[index_t]["v_s_x"] = 0
                        op.stat_tracker[index_t]["v_s_y"] = 0
                        logging.info(f"delta_r_native: {delta_r_native}")

                        # set v_s to -100 to test relation to movement
                        v_s_x = -100
                        v_s_y = 0
                        v_s   = np.tile(np.array([v_s_x, v_s_y]).reshape(1, 1, 2), (sim.x_size, sim.y_size, 1)).astype(np.float32)

                        # copy v_s array to GPU
                        GPU.cuda_v_s = cuda.np_to_array(v_s, order="C")
                        tex.set_array(GPU.cuda_v_s)
                        cuda.Context.synchronize()

                        # 2D v_s spacial info
                        v_s_strength = value([v_s_x, v_s_y])
                        v_s_angle    = np.degrees(np.arctan2(v_s_y, v_s_x))
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
                        v_s_angle    = np.arctan2(v_s_y, v_s_x)
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
                        sim.skyr_set_x = float(sim.distances[0])

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
                                    op.stat_tracker[start_v_s_x_y_deletion_index:]["v_s_x"] = 0
                                    op.stat_tracker[start_v_s_x_y_deletion_index:]["v_s_y"] = 0
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
                                    sim.skyr_set_x = float(sim.distances[index_now])

                                    logging.warning(f"skyr_set_x now: {sim.skyr_set_x}")
                                    logging.warning(f"new distances: {new_distances}")

                        # error is smaller than the smallest error yet and smaller than the max error * 100
                        if error_value < smallest_error_yet:
                            # set this as new best error --> load in as spinfield
                            logging.info(f"new best error: {error_value} setting this as starting spinfield")
                            temp_last_skyr_spinfield = last_skyr_spinfield.copy()
                            last_skyr_spinfield      = GPU.spins_evolved.copy()
                            smallest_error_yet       = error_value
                            
                            logging.info(f"resetting cyclic learning rate")
                            t_one_pos = 0

                        # LASTLY THERE WAS A STREAK
                        if error_streak_counter >= 1:
                            learning_rate = np.array([0.1, 0.1])
                            t_one_pos     = 0
                            logging.warning("Adjusting learning rate: 0.1")

                        # v_s has non 0 component(s)
                        elif np.any(v_s != 0):
                            learning_rate = spin.calculate_learning_rate(t_one_pos)

                        # CALCULATE NEW V_S
                        v_s_x = prev_v_s_x - (error[0]) / del_x_by_v_s[0] * learning_rate[0] * lr_adjustment
                        v_s_y = prev_v_s_y - (error[1]) / del_x_by_v_s[0] * learning_rate[1] * lr_adjustment

                        logging.info(
                            f"at t= {t:.6g} v_s_x, v_s_y, error[0], error[1], learning_rate[0]: {v_s_x, v_s_y, error[0], error[1], learning_rate[0]}"
                        )

                        # if the skyrmion has just been eliminated at the edge
                        if t_one_pos == 0 and skyr_elims > 0:
                            v_s_x /= 2

                        # log the new v_s
                        op.stat_tracker[index_t]["v_s_x"] = v_s_x
                        op.stat_tracker[index_t]["v_s_y"] = v_s_y

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
                                t_one_pos      = 0
                                reverts       += 1
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
                                last_skyr_spinfield      = GPU.spins_evolved.copy()
                                smallest_error_yet       = error_value

                                logging.info(f"resetting cyclic learning rate")
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
                            op.stat_tracker[index_t]["v_s_x"] = v_s_x
                            op.stat_tracker[index_t]["v_s_y"] = v_s_y

                            # reset the spinfield and place skyrmion at location x, y
                            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)

                            # increment the consecutive_reaches
                            error_streak_counter += 1
                            t_one_pos            += 1

                        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the first time
                        elif error_streak_counter >= sim.cons_reach_threashold and reset:
                            logging.warning(f"POTENTIAL V_S REACHED")

                            # reset the spinfield and place skyrmion at location x, y
                            cuda.memcpy_htod(GPU.spins_id, last_skyr_spinfield)
                            cuda.Context.synchronize()

                            # set counters
                            reset                = False
                            error_streak_counter = 0
                            t_one_pos            = 0

                        # 10 CONSECUTIVE REACHES OF ERROR HAVE HAPPENED the second time
                        elif error_streak_counter >= sim.cons_reach_threashold and not reset:

                            # angle of v_s
                            theta_deg = np.degrees(np.arctan(v_s_y / v_s_x))
                            logging.warning(f"Skyrmion stays at X={sim.skyr_set_x} with (vsx, vsy): ({v_s_x}, {v_s_y})")
                            logging.warning(f"angle at {t:011.6f} ns: {theta_deg}")
                            logging.warning(f"error at {t:011.6f} ns: {op.stat_tracker[index_t]['error']}")
                            logging.warning(f"max_error: {local_max_error}")

                            # CALCULATE FINAL V_S
                            v_s_x_last_n = op.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_x"]
                            v_s_x_avg    = np.average(v_s_x_last_n)
                            v_s_y_last_n = op.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["v_s_y"]
                            v_s_y_avg    = np.average(v_s_y_last_n)
                            r_last_n     = op.stat_tracker[index_t - sim.cons_reach_threashold - 1 : index_t]["r1"]
                            r_avg        = np.average(r_last_n)
                            logging.info(f"vsx_avg: {v_s_x_avg}")
                            logging.info(f"vsy_avg: {v_s_y_avg}")
                            logging.info(f"last 5 vsx: {v_s_x_last_n}")
                            logging.info(f"last 5 vsy: {v_s_y_last_n}")

                            # track the final v_s and r
                            op.stat_tracker[index_t]["v_s_x"] = v_s_x_avg
                            op.stat_tracker[index_t]["v_s_y"] = v_s_y_avg
                            op.stat_tracker[index_t]["r1"]    = r_avg

                            # reset the values of op.stat_tracker before index_t
                            op.stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_x"] = 0
                            op.stat_tracker[start_v_s_x_y_deletion_index:index_t]["v_s_y"] = 0

                            # set counters
                            reset                = True
                            error_streak_counter = 0
                            skyr_elims           = 0
                            lr_adjustment        = 1
                            t_one_pos            = 0

                            # NEW POSITION AVAILABLE
                            if sim.skyr_set_x < sim.distances[-1]:

                                # get the new position
                                index_now      = np.where(sim.distances == sim.skyr_set_x)[0][0]
                                sim.skyr_set_x = sim.distances[index_now + 1].item()
                                logging.warning(f"position {index_now + 1} of {len(sim.distances)} reached")
                                logging.warning(f"NEW X: {sim.skyr_set_x}")

                                # set v_s to 0
                                sim.update_current(v_s_sample_factor=0, bottom_angle=0)

                                # copy v_s array to GPU
                                GPU.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")
                                tex.set_array(GPU.cuda_v_s)
                                cuda.Context.synchronize()

                                # afterwards set the deletion index to the current index + 1
                                start_v_s_x_y_deletion_index = index_t + 1

                                # do not change vs simply track vsx and vsy
                                op.stat_tracker[index_t]["v_s_x"] = v_s_x
                                op.stat_tracker[index_t]["v_s_y"] = v_s_y

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

            (postfix_dict, left_c, right_c) = op.update_postfix_dict(postfix_dict, index_t, skyr_counter, op.count_open_fds())

            if sim.final_skyr_No > 1:
                op.stat_tracker[index_t]["left_count"] = left_c
                op.stat_tracker[index_t]["right_count"] = right_c

            pbar.set_postfix(postfix_dict)

            circular_spinfield_buffer.append(GPU.spins_evolved.copy())
            if index_t > 0 and sim.check_variance:
                variance = np.max(np.var(np.array(circular_spinfield_buffer, dtype=np.float32), axis=0))
                if variance < sim.critical_variance and skyr_counter == sim.final_skyr_No:
                    logging.warning("Spins are not moving anymore")
                    break
            if index_t > 0 and sim.check_skyrmion_presence and sim.final_skyr_No == 1:
                radius = op.stat_tracker[index_t]["r1"]
                ww = op.stat_tracker[index_t]["w1"]
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
    op.save_images_x_y_z(GPU.spins_evolved, final_pic_base_dir)

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
    np.save(f"{sample_dir}/traj_q.npy", op.stat_tracker)

    # ziehen von zeitlich gemittelten Spins auf die CPU in das neue Array avgImg
    avgImg = np.empty_like(GPU.avgTex)
    cuda.memcpy_dtoh(avgImg, GPU.avgTex_id)

    # Normierung von avgImg
    avgImg /= np.max(np.abs(avgImg))

    # Speichern der zeitlich gemittelten Spins
    avg_pic_dir = f"{sample_dir}/avg_z_component.png"
    op.save_image(avgImg, avg_pic_dir)

    if sim.model_type == "atomistic":
        for process in op.plot_processes:
            process.join()

    GPU.free_GPU_memory()

    return q_init, q_end


def main(sim_type="x_current"):
    """
    CREATED WITH THE HELP OF COPILOT
    Main function for the skyrmion simulation.

    This function performs the skyrmion simulation for multiple samples. It creates the necessary folder structure,
    applies bottom angles and v_s factors, saves images and npy files, executes calculations, updates masks and currents,
    transfers variables to the GPU, simulates the skyrmions, creates videos and plots, and resets the q_location_tracker.

    Parameter:
    sim_type (str): The type of simulation to run. Default is "x_current".

    """

    signal.signal(signal.SIGINT, op.signal_handler)

    # Laden der Variablen mittels der __init__ method der Konstantenklasse und Simulationsklasse
    cst()
    sim(sim_type)
    spin()
    GPU()
    op()

    for sample in range(sim.samples):
        logging.info(f"SAMPLE: {sample + 1}/{sim.samples}\n")

        # select the relevant bottom angle from the angles array
        bottom_angle = sim.bottom_angles[math.floor((sample) / sim.v_s_factors.shape[0])]
        logging.info(f"bottom_angle: {bottom_angle}") if sim.apply_bottom_angle else None

        # select the relevant v_s_factor from the v_s_factors array
        v_s_sample_factor = sim.v_s_factors[(sample) % sim.v_s_factors.shape[0]]
        logging.info(f"v_s_sample_factor: {v_s_sample_factor}\n")

        # Folder creation for the current sample
        sample_dir = f"{op.dest}/sample_{sample+1}_{bottom_angle}_deg_{v_s_sample_factor}_v_s_fac"
        os.makedirs(sample_dir)

        # create the folder for npy files if necessary
        os.makedirs(f"{sample_dir}/data") if sim.save_npys else None

        # save skyrmion image
        skyr_pic_dir = f"{sample_dir}/skyrmion_direction.png"
        op.save_images_x_y_z(sim.skyr, skyr_pic_dir)

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
                    colormap          = "gray"
                    logging.warning(f"drawing hexagonal mask to {new_atom_mask_dir}")
                    op.draw_hexagonal_spinfield(new_mask, colormap, new_atom_mask_dir, vmin=0)

            # copy the racetrack mask from the previous sample
            else:
                shutil.copy(sim.mask_dir, new_mask_dir)
                if sim.model_type == "atomistic":
                    atom_mask_dir = f"{sample_dir}/atomistic_racetrack.png"
                    colormap      = "gray"
                    logging.warning(f"drawing hexagonal mask to {atom_mask_dir}")
                    op.draw_hexagonal_spinfield(sim.mask, colormap, atom_mask_dir, vmin=0)

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
                op.current_and_potential_plot(current, potential, f"{sample_dir}/current_and_potential.png", model=sim.model_type)

            else:
                sim.v_s = sim.set_constant_v_s(v_s_sample_factor, bottom_angle)

        else:
            try:
                # copy the racetrack mask from the previous sample
                prev_bottom_angle      = sim.bottom_angles[math.floor((sample - 1) / sim.v_s_factors.shape[0])]
                prev_v_s_sample_factor = sim.v_s_factors[(sample - 1) % sim.v_s_factors.shape[0]]
                racetrack_source_dir   = f"{op.dest}/sample_{sample}_{prev_bottom_angle}_deg_{prev_v_s_sample_factor}_v_s_fac/racetrack.png"
                shutil.copy(racetrack_source_dir, new_mask_dir)

                if sim.v_s_dynamic and sim.v_s_active:

                    # copy the current distribution from the previous sample
                    current_source_dir = f"{op.dest}/sample_{sample}_{prev_bottom_angle}_deg_{prev_v_s_sample_factor}_v_s_fac/current.npy"
                    current_dest_dir   = f"{sample_dir}/current.npy"
                    shutil.copy(current_source_dir, current_dest_dir)
            except:
                try:
                    prev_prev_bottom_angle      = sim.bottom_angles[math.floor((sample - 2) / sim.v_s_factors.shape[0])]
                    prev_prev_v_s_sample_factor = sim.v_s_factors[(sample - 2) % sim.v_s_factors.shape[0]]
                    racetrack_source_dir        = (
                        f"{op.dest}/sample_{sample - 1}_{prev_prev_bottom_angle}_deg_{prev_prev_v_s_sample_factor}_v_s_fac/racetrack.png"
                    )
                    shutil.copy(racetrack_source_dir, new_mask_dir)

                    if sim.v_s_dynamic and sim.v_s_active:

                        # copy the current distribution from the previous sample
                        current_source_dir = (
                            f"{op.dest}/sample_{sample - 1}_{prev_prev_bottom_angle}_deg_{prev_prev_v_s_sample_factor}_v_s_fac/current.npy"
                        )
                        current_dest_dir = f"{sample_dir}/current.npy"
                        shutil.copy(current_source_dir, current_dest_dir)
                except:
                    logging.error("No 2 previous samples found, no racetrack mask copied")

        # update the mask and current
        sim.update_current_and_mask(bottom_angle, v_s_sample_factor, mask_dir=f"{sample_dir}/racetrack.png", j_dir=f"{sample_dir}/current.npy")

        # Simulation der Skyrmionen
        q_init, q_end = run_simulation(sample, bottom_angle, v_s_sample_factor)

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
        op.reset_q_loc_track()

    # delete the current_temp_folder if one is found
    if os.path.exists(f"current_temp/"):
        shutil.rmtree(f"current_temp")


if __name__ == "__main__":

    args = arg_parser()

    main(sim_type=args.sim_type)
