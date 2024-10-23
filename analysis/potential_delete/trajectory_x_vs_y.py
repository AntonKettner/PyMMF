# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
import shutil
import matplotlib.pyplot as plt


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
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_folder}/trajectory_x,y_vs_t.png", dpi=800)
    plt.close()


if __name__ == "__main__":
    main()
