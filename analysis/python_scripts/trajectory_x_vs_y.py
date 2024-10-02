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

    sig = density**20

    # density[density < 1] = 0

    x_center = np.sum(x_indices * sig) / np.sum(sig)
    y_center = np.sum(y_indices * sig) / np.sum(sig)

    return x_center, y_center


def main():
    fetch_folder_name = "/results_wall_retention_new_angled_2_ferro_boundary_rk4_1.15_B_ext_seems_correct"

    file = "traj_q.npy"

    pattern = os.path.join(fetch_folder_name, "**", file)

    traj_q_file = glob.glob(pattern, recursive=True)

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

    x_min_list = traj_q[:, 2]
    y_min_list = traj_q[:, 3]
    t = traj_q[:, 0]

    dest_folder = "trajectory_new_ferro_alex_1.15"

    # löschen von dest_folder, falls es existiert
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    os.makedirs(f"{dest_folder}")

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    plt.figure()
    a, b, fit = linear_fit(x_min_list, y_min_list)
    plt.plot(t, x_min_list, "ro", label="x")

    plt.plot(
        t,
        y_min_list,
        "g-",
        label=f"y",
    )
    plt.title("Graph of Trajectory (x vs y) over time")
    plt.legend(loc="upper right")
    plt.xlabel("t [5.1e-6 ns]")
    plt.ylabel("x, y [0.3 nm]")
    # plt.xlim([240, 580])
    # plt.ylim([60, 400])
    # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_folder}/trajectory_x,y_vs_t.png", dpi=800)
    plt.close()

    # x_y_array = np.array([x_min_list, y_min_list])

    # np.save(f"{dest_folder}/trajectory.npy", x_y_array)

    # ---------------------------------------------------------------insert photo snippets of the skyrmion at respective places-------------------------------------------------------------------

    # ---------------------------------------------------------------minima over time plot-------------------------------------------------------------------

    # x = np.arange(spinfield_right_min.shape[0])

    # plt.figure()
    # plt.plot(x[1:], spinfield_left_min[1:], "r-")
    # plt.plot(x[1:], spinfield_right_min[1:], "g-")
    # plt.title("comparison of right and left Energy minima distance from center of Skyr")
    # plt.xlabel("t")
    # plt.ylabel("y [0.4 nm]")
    # # plt.xlim([240, 580])
    # plt.ylim([0, 40])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_folder}/Minima_distances.png", dpi=500)
    # plt.close()

    # ---------------------------------------------------------------integral plot-------------------------------------------------------------------

    # x_0 = np.arange(integral_0.shape[0])
    # x_90 = np.arange(integral_90.shape[0])
    # x_180 = np.arange(integral_180.shape[0])
    # x_270 = np.arange(integral_270.shape[0])

    # print(integral_90)
    # print(integral_270)

    # plt.figure()
    # plt.plot(x_0, avg(integral_0, 20), color="blue", label=0)
    # plt.plot(x_90, avg(integral_90, 20), color="red", label=90)
    # plt.plot(x_180, avg(integral_180, 20), color="orange", label=180)
    # plt.plot(x_270, avg(integral_270, 20), color="brown", label=270)
    # plt.title("integrals over time")
    # plt.xlabel("t")
    # plt.ylabel("E_tot bis r = 150")

    # # add standard legend
    # plt.legend(loc="upper right")

    # # plt.xlim([240, 580])
    # # plt.ylim([0, 40])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_folder}/integrals_over time.png", dpi=500)
    # plt.close()

    # ---------------------------------------------------------------video from pngs with ffmpeg-------------------------------------------------------------------

    # os.system(
    #     f"ffmpeg -framerate 5 -pattern_type glob -i '{dest_folder}/*.png' -vcodec mpeg4 -y {dest_folder}/movie.mp4"
    # )


if __name__ == "__main__":
    main()
