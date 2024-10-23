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

# local imports
from common_functions import linear_fit

def main():
    fetch_folder_name = "results_wall_retention_2_rk4_1.5_B_ext"

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

    dest_folder = "trajectory_2_alex_1.15"

    # löschen von dest_folder, falls es existiert
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    os.makedirs(f"{dest_folder}")

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    plt.figure()
    a, b, fit = linear_fit(x_min_list, y_min_list)
    plt.plot(x_min_list, y_min_list, "ro", label="Trajectory")

    plt.plot(
        x_min_list,
        fit,
        "g-",
        label=f"Linear Fit,\na = {float(a):.5g},\nb = {float(b):.5g}",
    )
    print(f"a: {a}")
    plt.title("Graph of Trajectory (center of skyr)")
    plt.legend(loc="upper right")
    plt.xlabel("x [0.3 nm]")
    plt.ylabel("y [0.3 nm]")
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_folder}/trajectory_1_15T.png", dpi=800)
    plt.close()


if __name__ == "__main__":
    main()
