# Standard library imports

import logging  # enabling display of logging.info messages
import os
import shutil
import math

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import numpy as np

# import cupy as cp
from PIL import Image
from tqdm import tqdm

# import matplotlib

# matplotlib.use("TkAgg")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from scipy.spatial import cKDTree


global calculation_steps
calculation_steps = 6000000
global sim_model_for_main
sim_model_for_main = "atomistic"  # "atomistic" or "continuum"

def is_temp(path):
    return "temp" in os.path.basename(path).lower()


def draw_hex_with_plt(locations, potential, current, hex_array_upscaled, dest_file_name, dest_dir, image_dir):

    scalefactor = hex_array_upscaled.shape[0] / potential.shape[0]

    # scalefactor -> then it does not fit with racetrack shown below. could need debugging for edge cases of n
    n = 4

    # locations_x = locations[:, :, 0].flatten()
    # locations_y = locations[:, :, 1].flatten()
    all_locations_x = locations[:, :, 0]
    all_locations_y = locations[:, :, 1]
    locations_x = locations[::n, ::n, 0].flatten()
    locations_y = locations[::n, ::n, 1].flatten()
    pic_offset = 1 / (scalefactor * 2)

    # define the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the extent of the imshow plot to match the range of your locations
    extent = np.array(
        [all_locations_x.min() - pic_offset, all_locations_x.max() + 0.5 - pic_offset, all_locations_y.min() - pic_offset, all_locations_y.max() + 1 - pic_offset]
    )

    
    seismic = matplotlib.colormaps["seismic"]

    custom_seismic = colors.ListedColormap(seismic(np.linspace(0, 1, 256)))
    custom_seismic.set_bad(color="black")

    # set values in hex_array_upscaled that are in mask to nan
    # max_pot = np.max(hex_array_upscaled)
    # as value is set to 0 if not inside mask
    hex_array_upscaled[hex_array_upscaled == 0] = np.nan

    logging.warning(f"hex_array_upscaled[10,10]: {hex_array_upscaled[10,10]}\n")

    # if image_dir is None:
    #     # activate if potential as background is needed
    plt.imshow(hex_array_upscaled.T[::-1, :], cmap=custom_seismic, extent=list(extent))
    # else:
    #     # activate if image as background is needed
    #     img = Image.open(image_dir)
    #     plt.imshow(img, extent=list(extent))

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_title("Hexagonal Spinfield")

    ax.set_aspect("equal")

    # current_x = current[:, :, 0].flatten()
    # current_y = current[:, :, 1].flatten()
    current_x = current[::n, ::n, 0].flatten()
    current_y = current[::n, ::n, 1].flatten()

    # Create a mask where both current_x and current_y are not zero
    mask_not_zero = ~np.logical_and(current_x == 0, current_y == 0)

    # Use the mask_not_zero to filter the arrays
    locations_x_masked = locations_x[mask_not_zero]
    locations_y_masked = locations_y[mask_not_zero]
    current_x_masked = current_x[mask_not_zero]
    current_y_masked = current_y[mask_not_zero]

    # Overlay the current distribution with 0.5 opacity
    Q = ax.quiver(
        locations_x_masked,
        locations_y_masked,
        current_x_masked,
        current_y_masked,
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

    # logging.warning(f"Q.scale: {Q.scale}\n")
    # logging.warning(f"Q.width: {Q.width}\n")

    # Q = ax.quiver(
    #     locations_x_masked,
    #     locations_y_masked,
    #     current_x_masked,
    #     current_y_masked,
    #     pivot="middle",
    #     alpha=0.7,
    #     headwidth=3,
    #     headlength=5,
    #     scale=Q.scale * 3,
    #     scale_units="xy",
    #     width=Q.width * 1.2,
    # )

    # # for stuck comps:
    # Q = ax.quiver(
    #     locations_x_masked,
    #     locations_y_masked,
    #     current_x_masked,
    #     current_y_masked,
    #     pivot="middle",
    #     alpha=0.7,
    #     headwidth=2,
    #     headlength=4.5,
    #     scale=Q.scale * 2.5,
    #     scale_units="xy",
    #     width=Q.width * 1.1,
    # )
    # for stuck comps try -> every second arrow and larger:
    Q = ax.quiver(
        locations_x_masked,
        locations_y_masked,
        current_x_masked,
        current_y_masked,
        pivot="middle",
        alpha=0.7,
        headwidth=3,
        headlength=5,
        scale=Q.scale * 0.8,  # the bigger the shorter the arrows
        scale_units="xy",
        width=Q.width * 0.4,    # the bigger the thicker the arrows
        color = "yellow"
    )

    # for narrowings
    # if "narrowing" in dest_file_name:
    #     xmin = 161.11565209222337
    #     xmax = 229.81876068486244

    #     ymin = 53.69308410191169
    #     ymax = 96.80637087057181

    # # for corners
    # elif "corner" in dest_file_name:
    #     xmin = 150
    #     xmax = 220

    #     ymin = 75
    #     ymax = 120

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(dest_dir, dest_file_name), dpi=800, transparent=True)
    plt.show()


def calculate_upscaling_array(x_size, y_size, upscaling_factor=15):
    """
    Determine the upscaling locations that each pixel in the original hex array will be upscaled to."""

    # check if file in temp exists with the same mask and calc_steps as metadata
    temp_dir = f"{dirs.upscaling_temp_dir}/temp_upscaling_array_factor_{upscaling_factor}_{x_size}x{y_size}.npz"
    if os.path.exists(temp_dir):
        logging.info(f"UPSCALING ARRAY FOUND IN TEMP DIRECTORY\n")
        with np.load(temp_dir) as data:
            return data["upscaled_indices"], data["orig_locations"]

    logging.info(f"CALCULATING UPSCALING ARRAY FOR ATOMISTIC UPSCALING FACTOR {upscaling_factor} and FIELD SIZE {x_size} x {y_size}\n")

    # Create arrays for i and j indices
    i, j = np.indices((x_size, y_size))

    # Calculate x and y values
    x = i + 0.5 * ((j + 1) % 2)
    y = j * np.sqrt(3) / 2

    # Combine x and y values into a single array
    orig_locations = np.dstack([x, y])

    # logging.warning(f"ORIG LOCATIONS SHAPE {orig_locations.shape}\n")
    # logging.warning(f"ORIG LOCATIONS at 0,0 {orig_locations[0,0]}\n")
    # logging.warning(f"ORIG LOCATIONS at 0,1 {orig_locations[0,1]}\n")
    # logging.warning(f"ORIG LOCATIONS at 0,2 {orig_locations[0,2]}\n")
    # logging.warning(f"ORIG LOCATIONS at 0,3 {orig_locations[0,3]}\n")

    # Create arrays for i and j indices
    i, j = np.indices((upscaling_factor * x_size, math.ceil(y_size * upscaling_factor * np.sqrt(3) / 2))) / upscaling_factor

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
    indices = indices.reshape(x_size * upscaling_factor, math.ceil(y_size * upscaling_factor * np.sqrt(3) / 2))

    # Use the indices to fill the upscaled indices array
    upscaled_indices = np.stack(np.unravel_index(indices, (x_size, y_size)), axis=-1)

    logging.info(f"UPSCALING ARRAY SHAPE {upscaled_indices.shape}\n")

    np.savez(
        temp_dir,
        upscaled_indices=upscaled_indices,
        orig_locations=orig_locations,
    )

    logging.info(f"size of upscaling_temp array: {os.path.getsize(temp_dir)}\n")

    return upscaled_indices, orig_locations


def make_hexagonal_spinfield(orig_array):

    # logging.warning(f"upscaling_indices shape: {output.upscaling_indices.shape}")
    # logging.warning(f"orig_array shape: {orig_array.shape}")
    # Calculate the valid indices for orig_array

    upscaling_indices, locations = calculate_upscaling_array(orig_array.shape[0], orig_array.shape[1])
    valid_x_index = upscaling_indices[..., 0] % orig_array.shape[0]
    valid_y_index = upscaling_indices[..., 1] % orig_array.shape[1]

    # Use the valid indices to index orig_array
    upscaled_hex_array = orig_array[valid_x_index, valid_y_index]
    # plt.imsave(
    #     pic_name,
    #     upscaled_hex_array.T[::-1, :],
    #     cmap=colormap,
    #     vmin=-1,
    #     vmax=1,
    # )
    # plt.close()
    return upscaled_hex_array, locations


def current_and_potential_plot(current, potential, dest_dir, model, dest_file_name, image_dir):

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

        file_dest_dir = os.join(dest_dir, dest_file_name)

        plt.savefig(file_dest_dir, dpi=600)

        # plt.show()

    if model == "atomistic":

        if image_dir is None:
            hex_array, locations = make_hexagonal_spinfield(potential)
        else:
            # load the image as the potential:
            image = np.load(image_dir)[:,:,2] #--> only z value necessary
            logging.warning(f"image[10,10]: {image[10,10]}\n")
            logging.warning(f"image[100,100]: {image[100,100]}\n")
            hex_array, locations = make_hexagonal_spinfield(image)

        draw_hex_with_plt(locations, potential, current, hex_array, dest_file_name, dest_dir, image_dir)


class Maths:
    @staticmethod
    def project(a, b):
        if np.all(b == 0):
            return b
        else:
            return (np.dot(a, b) / np.linalg.norm(b) ** 2) * b


class dirs:

    upscaling_temp_dir = "OUTPUT/upscaling_temp"
    neumann_temp_dir = "OUTPUT/neumann_temp"
    if not os.path.exists(upscaling_temp_dir):
        os.makedirs(upscaling_temp_dir)
    if not os.path.exists(neumann_temp_dir):
        os.makedirs(neumann_temp_dir)

    dest_dir = "OUTPUT/current_calc"

    mask_file_dirs = []
    image_npy_dirs = []
    dest_file_names = []

    mask_file_dirs.append("needed_files/Mask_final_ReLU_high_beta_modular.png")
    image_npy_dirs.append(None)
    dest_file_names.append("current_test.png")


    # 
    # =========================================simple small beta for ReLU function =========================================
    # for i in range(8):
        # if i == 0:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_0.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0000.000018.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 1:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_2.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0002.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 2:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_4.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0004.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 3:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_6.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0006.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 4:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_8.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0008.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 5:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_10.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0010.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 6:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_12.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0012.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 7:
        #     mask_file_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_small_beta_end.png"
        #     image_npy_dir = "OUTPUT/ROMMING_SIMPLE_ReLU_for_THESIS_one_skyr_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0056.800000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)



    # =========================================for ReLU function (B_ext_var enabled) simple small beta=========================================
    # for i in range(8):
    #     if i == 0:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_0_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0000.000018.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 1:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_5_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0005.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 2:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_11_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0011.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 3:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_12_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0012.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 4:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_13_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0013.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 5:
    #         mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #         dest_file_name = "temp_simple_ReLU_small_beta_79_new.png"
    #         image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0079.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     # if i == 6:
    #     #     mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #     #     dest_file_name = "temp_simple_ReLU_small_beta_12_new.png"
    #     #     image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0012.000000.npy"
    #     #     image_npy_dirs.append(image_npy_dir)
    #     #     mask_file_dirs.append(mask_file_dir)
    #     #     dest_file_names.append(dest_file_name)
    #     # if i == 7:
    #     #     mask_file_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/racetrack.png"
    #     #     dest_file_name = "temp_simple_ReLU_small_beta_end_new.png"
    #     #     image_npy_dir = "OUTPUT/ROMMING_traj_FINAL_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac/data/Spins_at_t_0056.800000.npy"
    #     #     image_npy_dirs.append(image_npy_dir)
    #     #     mask_file_dirs.append(mask_file_dir)
    #     #     dest_file_names.append(dest_file_name)


    # =========================================for ReLU function (B_ext_var enabled) simple big beta=========================================

    # mask_file_dir = "OUTPUT/simulation/sample/racetrack.png"
    # for i in range(8):
    #     if i == 0:
    #         dest_file_name = "temp_simple_ReLU_big_beta_0_new.png"
    #         image_npy_dir = "OUTPUT/simulation/sample/data/Spins_at_t_0000.000014.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 1:
    #         dest_file_name = "temp_simple_ReLU_big_beta_4_new.png"
    #         image_npy_dir = "OUTPUT/simulation/sample/data/Spins_at_t_0004.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 2:
    #         dest_file_name = "temp_simple_ReLU_big_beta_5_new.png"
    #         image_npy_dir = "OUTPUT/simulation/sample/data/Spins_at_t_0005.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 3:
    #         dest_file_name = "temp_simple_ReLU_big_beta_6_new.png"
    #         image_npy_dir = "OUTPUT/simulation/sample/data/Spins_at_t_0006.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
    #     if i == 4:
    #         dest_file_name = "temp_simple_ReLU_big_beta_60_new.png"
    #         image_npy_dir = "OUTPUT/simulation/sample/data/Spins_at_t_0060.000000.npy"
    #         image_npy_dirs.append(image_npy_dir)
    #         mask_file_dirs.append(mask_file_dir)
    #         dest_file_names.append(dest_file_name)
        # if i == 5:
        #     mask_file_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/sample_1_0_deg_0.9_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_big_beta_79_new.png"
        #     image_npy_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/data/Spins_at_t_0079.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 6:
        #     mask_file_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/sample_1_0_deg_0.9_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_big_beta_12_new.png"
        #     image_npy_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/data/Spins_at_t_0012.000000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 7:
        #     mask_file_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/sample_1_0_deg_0.9_v_s_fac/racetrack.png"
        #     dest_file_name = "temp_simple_ReLU_big_beta_end_new.png"
        #     image_npy_dir = "OUTPUT/ROMMING_big_beta_traj_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0/sample_1_0_deg_0.9_v_s_fac/data/Spins_at_t_0056.800000.npy"
        #     image_npy_dirs.append(image_npy_dir)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)




    # for the pinning picture
    # for i in range(4):
        # if i == 0:
        #     mask_file_dir = "needed_files/Mask_track_corner_stuck.png"
        #     dest_file_name = "temp_full_hex_current_corner_stuck.png"
        #     image_dirs.append(None)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 1:
        #     mask_file_dir = "needed_files/Mask_track_corner_through.png"
        #     dest_file_name = "temp_full_hex_current_corner_through.png"
        #     image_dirs.append(None)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 2:
        #     mask_file_dir = "needed_files/Mask_track_narrowing_through.png"
        #     dest_file_name = "temp_full_hex_current_narrowing_through.png"
        #     image_dirs.append(None)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)
        # if i == 3:
        #     mask_file_dir = "needed_files/Mask_track_narrowing_stuck.png"
        #     dest_file_name = "temp_full_hex_current_stuck_narrowing.png"
        #     image_dirs.append(None)
        #     mask_file_dirs.append(mask_file_dir)
        #     dest_file_names.append(dest_file_name)


class csts:

    @classmethod
    def __init__(cls, model="continuum"):
        """
        calculates the  NN-vecs.

        Args:
            rotate_anticlock (bool, optional): If True, rotates the field collection
                anticlockwise by 90 degrees. Defaults to False.
        """

        # Rotation Matrix to create the NN vecs and exts
        cls.rot_matrix_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        # one nearest neighbor vector
        NN_vec_1 = np.array([1, 0, 0])
        NN_atom_ext_vec_1 = np.array([0, np.sqrt(3), 0])
        NN_cont_ext_vec_1 = np.array([1, 1, 0])

        if model == "atomistic":
            cls.No_NNs_max = 6
            cls.hex_image_scalefactor = 4
        if model == "continuum":
            cls.No_NNs_max = 4

        rotation_by_angle = lambda angle: np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # Nearest neighbor vectors
        cls.NN_vecs = np.empty((cls.No_NNs_max, 3), float)

        # to fit the second neighbor vectors of the atomistic model
        cls.NN_vec_exts = np.empty((cls.No_NNs_max, 3), float)

        # angle between the NNs
        NN_angle = 2 * np.pi / cls.No_NNs_max

        # calculate the NN vecs and exts
        for i in range(cls.No_NNs_max):
            cls.NN_vecs[i, :] = np.around((rotation_by_angle(i * NN_angle) @ NN_vec_1), decimals=5)
            if model == "atomistic":
                cls.NN_vec_exts[i, :] = np.around((rotation_by_angle(i * NN_angle) @ NN_atom_ext_vec_1), decimals=5)
                # reorganize the NN_vecs_ext to the last entry of the first axis is moved to the first position
            if model == "continuum":
                cls.NN_vec_exts[i, :] = np.around((rotation_by_angle(i * NN_angle) @ NN_cont_ext_vec_1), decimals=5)

        if model == "atomistic":
            cls.NN_vec_exts = np.roll(cls.NN_vec_exts, 1, axis=0)

        logging.info(f"csts CLASS INITIALIZED")
        logging.info(f"NN_vecs: \n {cls.NN_vecs}\n")
        logging.info(f"NN_vec_exts: \n {cls.NN_vec_exts}\n")

    def rotate_vec_90(vec):
        return np.ascontiguousarray(np.dot(csts.rot_matrix_90, vec))

    @staticmethod
    def find_NN_indices(mask, model="continuum"):

        # set the relative positions of the nearest neighbors and the extended nearest neighbors
        if model == "atomistic":
            NN_pos_even_row = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, -1, 0]])
            NN_pos_even_ext = np.array([[1, 1, 0], [0, 2, 0], [-2, 1, 0], [-2, -1, 0], [0, -2, 0], [1, -1, 0]])
            NN_pos_odd_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]])
            NN_pos_odd_ext = np.array([[2, 1, 0], [0, 2, 0], [-1, 1, 0], [-1, -1, 0], [0, -2, 0], [2, -1, 0]])
        if model == "continuum":
            NN_pos_even_row = NN_pos_odd_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
            NN_pos_even_ext = NN_pos_odd_ext = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])

        # NN counter for each point
        No_NNs = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

        # NN indices for each point
        NN_indices = np.ones((mask.shape[0], mask.shape[1], csts.No_NNs_max), dtype=int) * -1

        # mask for NN indices --> True if NN exists, False if not
        NN_indice_mask = np.zeros((mask.shape[0], mask.shape[1], csts.No_NNs_max), dtype=bool)

        # replace vecs for each point
        replace_vecs_field = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=float)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                NNs = 0
                if mask[i, j]:

                    # select the NN_positions depending on the row
                    if j % 2 == 0:
                        NN_pos = NN_pos_even_row
                    else:
                        NN_pos = NN_pos_odd_row

                    # count the number of NNs and set the NN index if it exists
                    for k in range(csts.No_NNs_max):
                        if mask[i + NN_pos[k, 0], j + NN_pos[k, 1]]:
                            NNs += 1
                            NN_indices[i, j, k] = (i + NN_pos[k, 0]) * mask.shape[1] + (j + NN_pos[k, 1])
                            NN_indice_mask[i, j, k] = True

                    # if point is in contact with border of mask
                    if not NNs == csts.No_NNs_max:

                        # calculate the replacement vector and add it to the field of replacement vectors
                        replace_vec = np.array([0, 0, 0])
                        for k in range(csts.No_NNs_max):
                            if not NN_indices[i, j, k] == -1:
                                replace_vec = replace_vec + csts.NN_vecs[k, :]
                        replace_vecs_field[i, j, :] = replace_vec = np.around(replace_vec, 3).copy()

                        # find the index of the neighbor that replacement vector points to
                        idx_replace_vec_found = False
                        for idx, vec in enumerate(np.around(csts.NN_vecs, 3)):
                            if np.array_equal(replace_vec, vec) or np.array_equal(replace_vec, 2 * vec):
                                idx_replace_vec_found = True
                                for k in range(csts.No_NNs_max):
                                    if NN_indices[i, j, k] == -1:
                                        NN_indices[i, j, k] = (i + NN_pos[idx, 0]) * mask.shape[1] + (j + NN_pos[idx, 1])
                                break

                        # if nesessary, search in the substitute locations
                        if not idx_replace_vec_found:
                            for idx, vec in enumerate(np.around(csts.NN_vec_exts, 3)):
                                if np.array_equal(replace_vec, vec):
                                    if j % 2 == 0:
                                        NN_pos_ext = NN_pos_even_ext
                                    else:
                                        NN_pos_ext = NN_pos_odd_ext
                                    for k in range(csts.No_NNs_max):
                                        if NN_indices[i, j, k] == -1:
                                            NN_indices[i, j, k] = (i + NN_pos_ext[idx, 0]) * mask.shape[1] + (j + NN_pos_ext[idx, 1])
                                    break
                # add the number of NNs to the field
                No_NNs[i, j] = NNs

        return (
            NN_indices.astype(np.int32),
            NN_pos_even_row.astype(np.int32),
            NN_pos_odd_row.astype(np.int32),
            No_NNs.astype(np.int32),
            replace_vecs_field.astype(np.float32),
            NN_indice_mask.astype(bool),
        )


def solve_neumann_problem(mask_file_dir, steps, model, vs=-0.05, neumann_temp_dir=dirs.neumann_temp_dir):

    # loading csts and sim init
    csts(model)
    # Initialisierung des Potentials
    # bestimmung von j-default anhand von vs
    j_default = vs / 7.801e-13

    # read mask from file and send it to the GPU
    mask = np.ascontiguousarray(np.array(np.array(Image.open(mask_file_dir), dtype=bool)[:, :, 0]).T[:, ::-1])

    # setting the field size
    field_size_x = mask.shape[0]
    field_size_y = mask.shape[1]

    # check if file in temp exists with the same mask and calc_steps as metadata
    for i in range(100):
        temp_dir = f"{neumann_temp_dir}/temp_{field_size_x}x{field_size_y}_{i}.npz"
        if os.path.exists(temp_dir):
            with np.load(temp_dir) as data:
                if np.array_equal(data["mask"], mask) and np.array_equal(data["steps"], steps):
                    logging.warning(f"temp file found with the same calculation steps -> no calculation done, taken from temp")
                    return data["phi"], data["j"]
        if i == 99:
            logging.warning(f"no temp file found with same mask and calculation steps as metadata, calculating new data")
    

    logging.info(f"mask to gpu")
    mask_id = cuda.mem_alloc(mask.nbytes)
    cuda.memcpy_htod(mask_id, mask)

    # setting NN indices:
    NN_indices, NN_pos_even_row, NN_pos_odd_row, No_NNs, replace_vecs, NN_indice_mask = csts.find_NN_indices(mask, model)
    NN_indices_id = cuda.np_to_array(NN_indices, order="C")
    print(NN_indices.shape)

    print(f"field_size_x: {field_size_x}")
    print(f"field_size_y: {field_size_y}")

    # Manuelles Setzen von input_left und output_right
    # input_left = np.column_stack((np.linspace(0, 99, 100), np.full((100), 3))).astype(int)
    # output_right = np.column_stack((np.linspace(0, 99, 100), np.full((100), 196))).astype(int)

    # =========================================same for atomistic and continuum: load the input and Output=========================================

    # Automatisches finden der input_left und output_right
    input_start_idx = None
    for i in range(field_size_x):  # swapped x and y
        for j in range(field_size_y):  # swapped x and y
            if mask[i, j]:  # swapped indices
                input_start_idx = i  # swapped x and y
                break
        if input_start_idx is not None:
            break

    input_left = np.empty((0, 2), dtype=int)
    if input_start_idx is not None:
        for i in range(field_size_y):  # swapped x and y
            if mask[input_start_idx, i]:  # swapped indices
                input_left = np.row_stack((input_left, np.array((input_start_idx, i))))  # swapped indices

    output_start_idx = None
    for i in range(field_size_x):  # swapped x and y
        for j in range(field_size_y):  # swapped x and y
            if mask[field_size_x - 1 - i, j]:  # swapped indices
                output_start_idx = field_size_x - 1 - i  # swapped x and y
                break
        if output_start_idx is not None:
            break

    output_right = np.empty((0, 2), dtype=int)
    if output_start_idx is not None:
        for i in range(field_size_y):  # swapped x and y
            if mask[output_start_idx, i]:  # swapped indices
                output_right = np.row_stack((output_right, np.array((output_start_idx, i))))  # swapped indices

    # logging.info(f"input (start, end): ({input_start_idx}, {input_left[-1, 0]})")
    # logging.info(f"output (start, end): ({output_start_idx}, {output_right[-1, 0]})\n")
    logging.info(f"input first (startx, starty): ({input_left[0,0]}, {input_left[0, 1]})")
    logging.info(f"input last (endx, endy): ({input_left[-1,0]}, {input_left[-1, 1]})")
    logging.info(f"output first (startx, starty): ({output_right[0,0]}, {output_right[0, 1]})\n")
    logging.info(f"output last (endx, endy): ({output_right[-1,0]}, {output_right[-1, 1]})\n")

    # flatten input_left and output_right
    input_left_flattened = np.empty((len(input_left[:, 0])), dtype=int)
    output_right_flattened = np.empty((len(output_right[:, 0])), dtype=int)
    for i in range(len(input_left[:, 0])):
        input_left_flattened[i] = input_left[i, 0] * field_size_y + input_left[i, 1]
    for i in range(len(output_right[:, 0])):
        output_right_flattened[i] = output_right[i, 0] * field_size_y + output_right[i, 1]

    # Initialisierung der Stromdichte als guess
    guess_pot = np.zeros((field_size_x, field_size_y))
    for i in range(field_size_x):
        for j in range(field_size_y):
            if mask[i, j]:
                guess_pot[i, j] = -j_default * i + 1 * (np.sign(j_default)) * (j_default * field_size_x) / 3

    # senden von guess_pot an GPU
    logging.info(f"Sending guess_pot to gpu\n")
    guess_pot = guess_pot.astype(np.float32)
    guess_pot_id = cuda.mem_alloc(guess_pot.nbytes)
    cuda.memcpy_htod(guess_pot_id, guess_pot)

    # factor for faster convergence --> to play around with
    omega_rel = 1.100

    # load the kernel into a string
    kernel_dir = "kernels/cc_kernel.c"
    with open(kernel_dir, "r") as file:
        map_creation_kernel = file.read()

    # add the constants to a string --> add the kernel to the constants
    constString = (
        f"__constant__ int field_size_y = {field_size_y};"
        f"__constant__ int field_size_x = {field_size_x};"
        f"__constant__ float omega_rel = {omega_rel};"
        f"__constant__ int input_left[{input_left_flattened.size}];"
        f"__constant__ int output_right[{output_right_flattened.size}];"
        f"__constant__ float j_default = {j_default};"
        f"__constant__ int input_size = {input_left_flattened.size};"
        f"__constant__ int output_size = {output_right_flattened.size};"
        f"__constant__ int No_NNs = {csts.No_NNs_max};"
        f"__constant__ float NN_vec[{csts.NN_vecs.size}];\n"
        f"__constant__ int NN_pos_even_row[{NN_pos_even_row.size}];\n"
        f"__constant__ int NN_pos_odd_row[{NN_pos_odd_row.size}];\n"
    )

    # load the kernel into the module and send the NN_indices to the GPU
    mod = SourceModule(constString + map_creation_kernel)
    texref = mod.get_texref("NNs")
    texref.set_array(NN_indices_id)
    cuda.mem_alloc(NN_indices.nbytes)

    # send the NN array and positions to the GPU
    NN_vec_id = mod.get_global("NN_vec")[0]
    NN_pos_even_row_id = mod.get_global("NN_pos_even_row")[0]
    NN_pos_odd_row_id = mod.get_global("NN_pos_odd_row")[0]
    cuda.memcpy_htod(NN_vec_id, csts.NN_vecs.astype(np.float32))
    cuda.memcpy_htod(NN_pos_even_row_id, NN_pos_even_row.astype(np.int32))
    cuda.memcpy_htod(NN_pos_odd_row_id, NN_pos_odd_row.astype(np.int32))

    # fetch the main function
    calc_step = mod.get_function("calc_step")

    # As mod is only defined right before ...
    input_left_device = mod.get_global("input_left")[0]  # only zeroth element as get_global returns a tuple
    output_right_device = mod.get_global("output_right")[0]  # only zeroth element as get_global returns a tuple

    # Copy the arrays where to input the current to the device
    logging.info(f"input_left and output_right to gpu\n")
    cuda.memcpy_htod(input_left_device, input_left_flattened.astype(np.int32))
    cuda.memcpy_htod(output_right_device, output_right_flattened.astype(np.int32))

    # set the grid and block size
    threadsperblock = (16, 16, 1)  # This is a common choice; adjust as needed
    blockspergrid_x = int(np.ceil(mask.shape[1] / threadsperblock[1] * 1))
    blockspergrid_y = int(np.ceil(mask.shape[0] / threadsperblock[0] * 1))
    blockspergrid = (blockspergrid_x, blockspergrid_y, 1)

    logging.info(f"blockspergrid: {blockspergrid}")
    logging.info(f"threadsperblock: {threadsperblock}")
    logging.info(f"total_threads: ({blockspergrid_x * threadsperblock[0]}, {blockspergrid_y * threadsperblock[1]})\n")

    # This is where the magic happens
    with tqdm(total=steps, desc="Potential calculation", unit=f"...steps") as pbar:
        for _ in range(steps):
            # calc_step(guess_pot_id, mask_id, block=(8,8,1), grid=(20,40,1))
            calc_step(guess_pot_id, mask_id, block=threadsperblock, grid=blockspergrid)
            pbar.update(1)

    logging.info("potential calculation finished, fetching data")
    phi = np.empty_like(guess_pot)
    cuda.memcpy_dtoh(phi, guess_pot_id)

    # set potential around spinfield to higher then max potential for plotting later
    phi[~mask] = 1.2 * np.max(phi)

    logging.info("data fetched, calculating current density")
    j = np.zeros((field_size_x, field_size_y, 2)).astype(np.float32)

    # calculate the current density based on the potential (negative gradient)
    if model == "continuum":
        j[1:-1, 1:-1, 0] = -0.5 * (phi[2:, 1:-1] - phi[:-2, 1:-1])
        j[1:-1, 1:-1, 1] = -0.5 * (phi[1:-1, 2:] - phi[1:-1, :-2])
    if model == "atomistic":
        # calculate the current at each location manually:
        for i in range(field_size_x):
            for l in range(field_size_y):
                if mask[i, l]:
                    div = np.array([0, 0, 0])
                    for k in range(csts.No_NNs_max):
                        if NN_indice_mask[i, l, k]:
                            direction = csts.NN_vecs[k, :]
                            neigh_loc = NN_indices[i, l, k]
                            neigh_loc_y = neigh_loc % field_size_y
                            neigh_loc_x = int((neigh_loc - neigh_loc_y) / field_size_y)
                            Phi_neigh = phi[neigh_loc_x, neigh_loc_y]
                            div_directional = (Phi_neigh - phi[i, l]) * direction
                            div = div + div_directional
                    j[i, l] = -div[:-1]

    # Ausnahmen für Stromdichte an Rändern, Ecken und Input
    for i in range(field_size_x):
        for k in range(field_size_y):

            # Falls der Punkt input oder output ist, ist j = j_default
            if k + i * field_size_y in input_left_flattened or k + i * field_size_y in output_right_flattened:
                j[i, k, 0] = 0
                j[i, k, 1] = j_default

            # otherwise, if the point is not in the mask, set j to zero
            elif mask[i, k]:
                if No_NNs[i, k] < csts.No_NNs_max:
                    proj_vec = csts.rotate_vec_90(vec=replace_vecs[i, k, :])
                    j[i, k] = Maths.project(j[i, k, :], proj_vec[:-1])

    # löschen von j an den Stellen, wo mask False ist
    j[~mask] = np.array((0, 0))

    # pycuda.autoinit.context.pop()

    # save phi and j in temp npz file together with mask and calculation steps to compare to

    for i in range(100):
        temp_dir = f"{neumann_temp_dir}/temp_{field_size_x}x{field_size_y}_{i}.npz"
        if not os.path.exists(temp_dir):
            break
        if i == 99:
            raise ValueError("Too many temp files")
    
    logging.warning(f"saving temp file to {temp_dir}\n")
    np.savez_compressed(
        temp_dir,
        phi=phi,
        j=j,
        mask=mask,
        steps=steps,
    )

    return phi, j


def main():

    dirs()

    dest_dir = dirs.dest_dir

    for i in range(len(dirs.mask_file_dirs)):
        mask_file_dir = dirs.mask_file_dirs[i]
        image_dir = dirs.image_npy_dirs[i]
        dest_file_name = dirs.dest_file_names[i]

        potential_data, current_data = solve_neumann_problem(mask_file_dir, calculation_steps, sim_model_for_main)
        logging.info(f"j calculated, writing current to {dest_dir}\n")

        if os.path.exists(dest_dir):
            # Iterate over the contents of dest_dir
            for item in os.listdir(dest_dir):
                item_path = os.path.join(dest_dir, item)
                if not is_temp(item_path):
                    # Delete files and directories that don't contain "temp"
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        else:
            os.makedirs(dest_dir)

        np.save(
            f"{dest_dir}/current.npy",
            current_data,
        )

        np.save(
            f"{dest_dir}/potential.npy",
            potential_data,
        )

        # logging.warning(f"potential_data: {potential_data}\n")
        # logging.warning(f"potential_data.shape: {potential_data.shape}\n")
        # logging.warning(f"potential_data[10,10] datatype: {potential_data[10,10].dtype}\n")
        # logging.warning(f"info potential_data[10,10]: {np.info(potential_data[10, 10])}\n")
        # logging.warning(f"info potential_data[10,100]: {np.info(potential_data[10, 100])}\n")
        # logging.warning(f"info potential_data[10,10]: {potential_data[10, 10]}\n")
        # logging.warning(f"info potential_data[10,10]: {potential_data[10, 12]}\n")
        # logging.warning(f"info potential_data[10,100]: {potential_data[10, 100]}\n")
        # logging.warning(f"isnan pot[10,10] {np.isnan(potential_data[10,10])}\n")
        # current_data = np.load(f"{sim.dest_dir}/current.npy")
        # potential_data = np.load(f"{sim.dest_dir}/potential.npy")

        # ---------------------------------------------------------------Data Plot-------------------------------------------------------------------

        current_and_potential_plot(current_data, potential_data, dest_dir, model=sim_model_for_main, dest_file_name=dest_file_name, image_dir=image_dir)


if __name__ == "__main__":
    main()

    # # Create some data
    # x = [1, 2, 3, 4, 5]
    # y = [1, 4, 9, 16, 25]

    # # Plot the data
    # plt.plot(x, y)

    # # Show the plot
    # plt.show()
