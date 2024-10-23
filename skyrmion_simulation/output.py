# native imports
import logging
import gc
import os
import sys
import shutil
import time
import math
import multiprocessing

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial import cKDTree

# local imports
from constants import cst
from simulation import sim
from gpu import GPU

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# if os.environ.get("DISPLAY", "") == "":
#     logging.info("No display found. Using non-interactive Agg backend.")
matplotlib.use("Agg")
# else:
#     logging.info("Display found. Using interactive Qt5Agg backend.")
#     matplotlib.use("Qt5Agg")

# Navigate two levels up from the current script's directory
parent_directory = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Add the parent directory to sys.path to access modules from there
sys.path.insert(0, parent_directory)


class op:
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
            cls.atomistic_upscaling_factor = 5
            cls.upscaling_indices, cls.locations = cls.calculate_upscaling_array(sim.x_size, sim.y_size)

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
        logging.info(f"skyr radius: {sim.r_skyr:.4g} [nm]")
        try:
            logging.info(f"skyr wall width: {sim.r_skyr:.4g} [nm]")
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
        # if sim.v_s_to_wall:
        #     logging.info(f"v_s_to_wall threshold: {sim.x_threashold}\n")

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
            f"CALCULATING UPSCALING ARRAY FOR ATOMISTIC UPSCALING FACTOR {op.atomistic_upscaling_factor} and FIELD SIZE {field_size_x} x {field_size_y}\n"
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
                (op.atomistic_upscaling_factor * field_size_x, math.ceil(field_size_y * op.atomistic_upscaling_factor * np.sqrt(3) / 2))
            )
            / op.atomistic_upscaling_factor
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
            field_size_x * op.atomistic_upscaling_factor, math.ceil(field_size_y * op.atomistic_upscaling_factor * np.sqrt(3) / 2)
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
            op.save_image(spinfield_vectors[:, :, cst.coords[coord]], current_pic_name)

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
                cmap=op.custom_seismic,
                vmin=-1,
                vmax=1,
            )

        if sim.model_type == "atomistic":
            plot_process = multiprocessing.Process(
                target=op.draw_hexagonal_spinfield,
                args=(
                    spinfield_x_or_y_or_z,
                    op.custom_seismic,
                    pic_name,
                ),
            )

            plot_process.start()
            op.plot_processes.append(plot_process)

            for process in list(op.plot_processes):
                if not process.is_alive():
                    process.join()
                    op.plot_processes.remove(process)

            while len([p for p in op.plot_processes if p.is_alive()]) >= op.max_parallel_processes:
                time.sleep(0.1)  # Wait for 0.1 second

    @staticmethod
    def draw_hexagonal_spinfield(orig_array, colormap, pic_dir, vmin=-1, vmax=1):

        # logging.warning(f"upscaling_indices shape: {op.upscaling_indices.shape}")
        # logging.warning(f"orig_array shape: {orig_array.shape}")
        # Calculate the valid indices for orig_array

        if orig_array.shape == (sim.x_size, sim.y_size):
            valid_x_index = op.upscaling_indices[..., 0] % orig_array.shape[0]
            valid_y_index = op.upscaling_indices[..., 1] % orig_array.shape[1]
        else:
            temp_upscaling_indices, locations = op.calculate_upscaling_array(orig_array.shape[0], orig_array.shape[1])
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

            hex_array = op.draw_hexagonal_spinfield(potential, "hot", dest_dir)

            op.draw_hex_with_plt(potential, current, hex_array)

    @staticmethod
    def draw_hex_with_plt(potential, current, hex_array_upscaled):

        scalefactor = hex_array_upscaled.shape[0] / potential.shape[0]

        locations_x = op.locations[:, :, 0].flatten()
        locations_y = op.locations[:, :, 1].flatten()
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
        # logging.info(f"t: {op.stat_tracker[index_t]['time']:.2f} ns; index_t: {index_t}; q: {op.stat_tracker[index_t]['topological_charge']:.2f}, q_prev: {op.stat_tracker[index_t-1]['topological_charge']:.2f}")
        postfix_dict["Q"] = round(float(op.stat_tracker[index_t]["topological_charge"]))
        postfix_dict["No set"] = skyrs_set

        # Count skyrmions on the left and right
        left_count, right_count = 0, 0
        # Update for single or multiple skyrmions
        if sim.final_skyr_No <= 1:
            postfix_dict["(x, y)"] = (
                round(float(op.stat_tracker[index_t]["x0"]), 4),
                round(float(op.stat_tracker[index_t]["y0"]), 4),
            )
            # print("updating this shit")
            # print(op.stat_tracker[index_t]["r1"])
            postfix_dict["r"] = f'{op.stat_tracker[index_t]["r1"]:.2f} nm'
            postfix_dict["w"] = f'{op.stat_tracker[index_t]["w1"]:.2f} nm'
            # print(f"tracker ist da{op.q_location_tracker[index_t]['r1']}")
        else:
            # Define the boundary for left and right side
            boundary_x = sim.x_size / 2 + 70

            for i in range(min(abs(postfix_dict["Q"]), sim.final_skyr_No)):
                x_coord = op.stat_tracker[index_t][f"x{i}"]
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
            postfix_dict["error"] = op.stat_tracker[index_t]["error"]

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


if __name__ == "__main__":
    print("This Code is not meant to be run directly, but imported from main.py.")