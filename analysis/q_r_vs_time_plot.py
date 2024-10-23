# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def create_q_r_vs_time_plot(fetch_dir, dest_dir, dest_file="q_r_vs_time.png"):
    matplotlib.use("Agg")
    # fetch the traj_q file from the fetch folder
    filename = "traj_q.npy"

    pattern_file = os.path.join(fetch_dir, "**", filename)

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

    # fetch the spinfield from the fetch folder
    racetrack_name = "racetrack.png"
    racetrack_pattern = os.path.join(fetch_dir, "**", racetrack_name)
    racetrack_files = glob.glob(racetrack_pattern, recursive=True)
    amount_of_race_track_files = len(glob.glob(racetrack_pattern, recursive=True))
    if amount_of_race_track_files != 1:
        logging.error(f"There should be exactly one {racetrack_name} file.")
        if amount_of_race_track_files == 0:
            logging.error(f"No {racetrack_name} file was found.")
        elif amount_of_race_track_files > 1:
            logging.error(f"{amount_of_race_track_files} {racetrack_name} files were found.")
        exit()
    racetrack = np.ascontiguousarray(
        np.array(np.array(Image.open(racetrack_files[0]), dtype=bool)[:, :, 0]).T[:, ::-1]
    )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis


    # ---------------------------------------------------------------Trajectory, q, r triple plot-------------------------------------------------------------------

    # remove entries from the back of traj_q, where traj_q["time"] is 0 and remove the first entry
    traj_q = traj_q[traj_q["time"] != 0][1:]

    # # Extract data fields
    w = traj_q["w1"]
    t = traj_q["time"]
    q = traj_q["topological_charge"]
    r = traj_q["r1"]

    # print(q)

    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    plt.rcParams['font.family'] = 'serif'


    # Create the plot
    fig, ax1 = plt.subplots()

    fig.patch.set_facecolor("black")  # Setting figure background to black
    ax1.set_facecolor("black")  # Setting axes background to black

    # Set plot limits and labels
    # ax1.set_xlim([0, 410])
    ax1.set_xlabel("t [ns]")
    ax1.tick_params(colors="white", which="both")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax1.title.set_color("white")

    # Create and set properties of the first y-axis (Q)
    (Q,) = ax1.plot(t[2:], q[2:], "b-", label="Q", linewidth=0.8)
    ax1.set_ylim([np.min(q[2:]), np.min(q[2:]) + (np.max(q[2:]) - np.min(q[2:])) * 3])
    ax1.set_ylabel("Q", color="b")
    ax1.tick_params("y", colors="b")

    # Create and set properties of the second y-axis (radius)
    ax2 = ax1.twinx()
    (Radius,) = ax2.plot(t, r, "g-", label="Radius", linewidth=0.8)
    # log all the limits of the radius:
    logging.info(f"min(r): {np.min(r)}")
    logging.info(f"max(r): {np.max(r)}")
    ax2.set_ylim([np.min(r) - (np.max(r) - np.min(r)), np.max(r) + (np.max(r) - np.min(r))])
    ax2.set_ylabel("Radius[nm]", color="g")
    ax2.tick_params("y", colors="g")
    ax2.spines["right"].set_color("g")

    # Create and set properties of the third y-axis (wall width)
    ax3 = ax1.twinx()
    # Offset the third axis
    ax3.spines["right"].set_position(("outward", 50))

    # Assuming w is your data for wall width
    (WallWidth,) = ax3.plot(t, w, "r-", label="Wall Width", linewidth=0.8)
    ax3.set_ylim([np.min(w) - 2 * (np.max(w) - np.min(w)), np.max(w)])
    ax3.set_ylabel("Wall Width [nm]", color="r")
    ax3.tick_params("y", colors="r")
    ax3.spines["right"].set_color("r")
    ax3.spines["bottom"].set_color("white")  # x-axis
    ax3.spines["left"].set_color("blue")  # Left y-axis

    # Update the legend to include Wall Width
    legend = ax1.legend(
        [Q, Radius, WallWidth],
        ["Q", "Radius", "Wall Width"],
        loc="upper right",
        facecolor="black",
    )

    plt.setp(legend.get_texts(), color="w")

    # plt.ylim([60, 400])
    # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    ax1.set_title(rf"Graph of Radius, Wall width and Topological Charge against time")
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_dir}/{dest_file}", dpi=800)
    print(f"end_q: {q[-1]}, end_r: {r[-1]}, end_w: {w[-1]}")
    plt.close()


def main():
    # set these vars
    boundary = "open"  # "ferro", "open"
    vs = 0  # 15, 8
    B_ext = 1.5  # 1.5, 1.15

    # fetch_folder_name = f"OUTPUT/results_x_current_dynamic_center_start_{boundary}_boundary_heun_{B_ext}_B_ext_{vs}_vs"

    # boundary = "open"  # "ferro", "open"
    # vs = 8  # 15, 8
    # B_ext = 1.5  # 1.5, 1.15

    fetch_folder_name = f"OUTPUT/ROMMING_FINAL_SIM_x_offset_test_atomistic_skyrmion_creation_2.5_r_open_heun_1.5_5_0/"

    dest_folder = f"OUTPUT/trajectories"
    dest_folder = fetch_folder_name

    create_q_r_vs_time_plot(fetch_folder_name, dest_folder)

    # ---------------------------------------------------------------Trajectory, q, r triple plot-------------------------------------------------------------------


if __name__ == "__main__":
    main()
