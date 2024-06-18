# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function

import json
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import matplotlib.ticker as tck


def linear_fit(x, y):
    """
    Fit x and y data with a linear model and return the slope, intercept, and fitted data.

    Parameters:
    x (array-like): The x-coordinates of the data.
    y (array-like): The y-coordinates of the data.

    Returns:
    tuple: slope (a), intercept (b), fitted data (y_fit)
    """
    x = np.array(x)[:]  # Ensure x is a numpy array
    y = np.array(y)[:]  # Ensure y is a numpy array too, for consistency

    # Perform linear fit
    a, b = np.polyfit(x, y, 1)

    # Generate the fitted data
    y_fit = a * x + b

    return a, b, y_fit


def format_func(value, tick_number):
    return "{:.1g}".format(value)


def angled_line_and_r(matrix, x, y, x_plus=1, y_plus=1, red=0, max_r=200):
    # the default is 45 deg
    values = []

    # Get matrix dimensions
    rows, cols = matrix.shape

    while 0 <= x < rows and 0 <= y < cols:
        values.append(matrix[x, y] - red)
        x += x_plus
        y += y_plus

    step_length = np.sqrt(x_plus**2 + y_plus**2)

    # cut off at max_r
    if len(values) * step_length > max_r:
        values = values[: int(np.floor(max_r / step_length))]

    r = np.arange(len(values)) * step_length

    return np.array(values), r


def avg(matrix, n):
    # Ensure matrix is a numpy array
    matrix = np.asarray(matrix)

    # Check if matrix is 1D
    if len(matrix.shape) != 1:
        raise ValueError("The input matrix should be a 1D numpy array.")

    # Create a uniform kernel of size 2*n + 1
    kernel_size = 2 * n + 1
    kernel = np.ones(kernel_size) / kernel_size

    # Convolve data with the kernel
    averaged_data = np.convolve(matrix, kernel, mode="same")

    # Handle edge cases
    for i in range(n):
        averaged_data[i] = matrix[: i + n + 1].mean()
        averaged_data[-(i + 1)] = matrix[-(i + n + 1) :].mean()

    return averaged_data


def find_skyr_center(m_z, idx):
    """
    Determine the center of the skyrmion using a weighted center of mass approach.

    Parameters:
    m_z (2D numpy array): The mz component of the magnetization.

    Returns:
    tuple: The coordinates (y_center, x_center) of the skyrmion center.
    """
    x_indices, y_indices = np.indices(m_z.shape)
    density = -(m_z)  # goes from 0 to 2

    print(f"density_max: {np.max(density)}") if idx == 0 else None
    print(f"density_min: {np.min(density)}") if idx == 0 else None

    sig = density  # **20

    # density[density < 1] = 0

    x_center = np.sum(x_indices * sig) / np.sum(sig)
    y_center = np.sum(y_indices * sig) / np.sum(sig)

    return x_center, y_center


def create_input_output_plot(fetchpath, destpath, dest_file):
    # fetch the imagepath from the fetch folder: spinfield_t_0001.805000.png

    image_path_1_pattern = os.path.join(fetchpath, "**", "spinfield_t_0001.805000.png")

    image_path_1 = glob.glob(image_path_1_pattern, recursive=True)

    logging.info(f"image_path: {image_path_1}")

    file = "traj_q.npy"

    pattern_file = os.path.join(fetchpath, "**", file)

    traj_q_file = glob.glob(pattern_file, recursive=True)

    amount_of_traj_q_files = len(traj_q_file)

    # error catching
    if amount_of_traj_q_files != 1:
        logging.error("There should be exactly one traj_q.npy file.")
        if amount_of_traj_q_files == 0:
            logging.error("No traj_q.npy file was found.")
        elif amount_of_traj_q_files > 1:
            logging.error(f"{amount_of_traj_q_files} traj_q.npy files were found.")
        exit()

    traj_q = np.load(traj_q_file[0])

    # logging.info(f"traj_q: {traj_q}")

    logging.info(f"traj_q data: {traj_q.dtype.names}")

    # # Determine the number of fields
    # num_fields = len(traj_q.dtype.names)

    # # Create a new array where each record is a row
    # regular_array = np.empty((len(traj_q), num_fields))

    # # Copy data from each field
    # for i, name in enumerate(traj_q.dtype.names):
    #     regular_array[:, i] = traj_q[name]

    q = traj_q["topological_charge"]
    l = traj_q["left_count"]
    r = traj_q["right_count"]

    # löschen von file in dest_path, falls es existiert
    if os.path.exists(f"{destpath}/{dest_file}"):
        os.remove(f"{destpath}/{dest_file}")

    # ansonsten erstellung von dest_path falls es nicht existiert
    if not os.path.exists(f"{destpath}"):
        os.makedirs(f"{destpath}")

    # ---------------------------------------------------------------Input Output plot-------------------------------------------------------------------

    plt.figure()

    # clear all values of q and r, where r is the same value as the previous one
    q_filtered = q.copy()
    for i in range(len(r) - 1):
        if r[i] == r[i + 1] and not r[i] == 0:
            q_filtered[i] = np.nan
            r[i] = np.nan

    # clear all values of q and r, where r is the same value as the previous one
    q_filtered = q_filtered[~np.isnan(q_filtered)]
    r = r[~np.isnan(r)]

    # clear last value of q and r, where r is the same value as the previous one
    q_filtered = q_filtered[:-1]
    r = r[:-1]
    plt.style.use("dark_background")

    plt.plot(abs(q_filtered), r, "r-", label="Output")
    plt.plot(abs(q), l, "wo", label="Input-Output")

    plt.title("Input - Output")
    plt.legend(loc="upper left")
    plt.xlabel("Q")
    plt.ylabel("Output")
    plt.tight_layout()
    plt.savefig(f"{destpath}/{dest_file}", dpi=800)
    plt.close()


def main():
    mode = "first_results_replica"
    boundary = "open"  # "ferro", "open"
    vs = 9  # 15, 8
    B_ext = 1.5  # 1.5, 1.15

    dest_file = f"input_output_{B_ext}_{boundary}_{vs}.png"

    fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/ap_r/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/results_{mode}_test_multitrack_{boundary}_boundary_rk4_{B_ext}_B_ext_{vs}_vs"

    dest_folder = f"/afs/physnet.uni-hamburg.de/users/ap_r/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/input_output"

    create_input_output_plot(fetch_folder_name, dest_folder, dest_file)


if __name__ == "__main__":
    main()
