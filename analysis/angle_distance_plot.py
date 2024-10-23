# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# local imports

if __name__ == "__main__":
    from common_functions import setup_plt_with_tex
elif __name__ == "analysis.angle_distance_plot":
    from analysis.common_functions import setup_plt_with_tex

def constant_fit(x, y):
    """
    Fit x and y data with a constant model and return the constant.

    Parameters:
    x (array-like): The x-coordinates of the data.
    y (array-like): The y-coordinates of the data.

    Returns:
    float: constant (c)
    """
    x = np.array(x)[:]  # Ensure x is a numpy array
    y = np.array(y)[:]  # Ensure y is a numpy array too, for consistency

    # Perform linear fit
    c = np.polyfit(x, y, 0)

    return c[0]


def fetch_traj_q_file(fetch_dir, fetch_file="traj_q.npy"):
    # fetch the traj_q file from the fetch folder
    filename = fetch_file

    pattern_file = os.path.join(fetch_dir, "**", filename)

    traj_q_file = glob.glob(pattern_file, recursive=True)

    amount_of_traj_q_files = len(traj_q_file)

    # error catching
    if amount_of_traj_q_files != 1:
        logging.info(f"current working directory: {os.getcwd()}")
        logging.error(f"There should be exactly one traj_q.npy file at {fetch_dir}.")
        if amount_of_traj_q_files == 0:
            logging.error("No traj_q.npy file was found.")
        elif amount_of_traj_q_files > 1:
            logging.error(f"{amount_of_traj_q_files} traj_q.npy files were found.")
        exit()

    return np.load(traj_q_file[0])


def current_vs_distance_plot(fetch_dir, dest_dir, dest_file="angle_distance_overhaul_new.png", fetch_file="traj_q.npy"):
    matplotlib.use("Agg")
    traj_q = fetch_traj_q_file(fetch_dir, fetch_file)

    # fetch the spinfield from the fetch folder
    racetrack_name = "racetrack.png"
    racetrack_pattern = os.path.join(fetch_dir, "**", racetrack_name)
    racetrack_files = glob.glob(racetrack_pattern, recursive=True)
    amount_of_race_track_files = len(glob.glob(racetrack_pattern, recursive=True))
    if amount_of_race_track_files != 1:
        logging.info(f"current working directory: {os.getcwd()}")
        logging.error(f"There should be exactly one {racetrack_name} file at {fetch_dir}.")
        if amount_of_race_track_files == 0:
            logging.error(f"No {racetrack_name} file was found.")
        elif amount_of_race_track_files > 1:
            logging.error(f"{amount_of_race_track_files} {racetrack_name} files were found.")
        exit()
    racetrack = np.ascontiguousarray(
        np.array(np.array(Image.open(racetrack_files[0]), dtype=bool)[:, :, 0]).T[:, ::-1]
    )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis

    # get the x and y dim
    field_size_x = racetrack.shape[0]
    field_size_y = racetrack.shape[1]
    logging.info(f"field_size_x, field_size_y: {field_size_x}, {field_size_y}")

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    # Find the indices for the images that should be fetched -> equal spacing from max x

    x = traj_q["x0"]
    r = traj_q["r1"]

    # Extract data fields
    y = traj_q["y0"]
    vsx = traj_q["v_s_x"]
    vsy = traj_q["v_s_y"]
    logging.info(f"x: {x}")
    logging.info(f"r: {r}")
    logging.info(f"last 100 vsx: {vsx[100:]}")
    logging.info(f"last 100 vsy: {vsy[100:]}")

    # filter all elements of traj_q where vsx is 0
    valid_indices = np.where(np.logical_and(vsx != 0, True))[0][2:]  # [0]  # Get the array from the tuple x > 372
    # valid_indices = valid_indices[3:]
    # if valid_indices.size > 14:
    #     valid_indices = np.concatenate((valid_indices[:-12], [valid_indices[-1]]))  # Remove the last 13 indices but keep the last one

    x = x[valid_indices]
    vsx = vsx[valid_indices]
    vsy = vsy[valid_indices]
    r = r[valid_indices]
    a = 0.27    # 0.27 for mine, 0.3 for martinez
    vsx *= -1 * a  # weil Berechnung in (a)/ns und hier a = 3e-10m
    vsy *= -1 * a  # weil Berechnung in (a)/ns und hier a = 3e-10m
    # r = np.append(r,
    # x = np.append(x, np.array([388.00003, 388.99976, 389.99976]))
    # vsx = np.append(vsx, np.array([18.376328, 21.231339, 22.72457]))
    # vsy = np.append(vsy, np.array([146.95436, 169.66826, 178.41371]))

    # reformulate x as distance to the wall
    Delta_x = (field_size_x - x) * a

    # Calculate the angle of the velocity vector
    angle = np.degrees(np.arctan(vsy / vsx))

    # # Fit a line to the angle data
    # slope, intercept = np.polyfit(Delta_x, angle, 1)  # 1 is the degree of the polynomial
    # fit_line = slope * Delta_x + intercept

    logging.info(f"x: {x}")
    logging.info(f"angle: {angle}")
    logging.info(f"Delta_x: {Delta_x}")
    logging.info(f"r: {r}")
    logging.info(f"vsx: {vsx}")
    logging.info(f"vsy: {vsy}")


    # q = traj_q["topological_charge"]
    # r = traj_q["r1"]

    setup_plt_with_tex()

    alpha = 1

    fig, ax1 = plt.subplots()

    # Create a second y-axis and plot the angle and its fit on it
    ax2 = ax1.twinx()
    (vsx_line,) = ax2.plot(Delta_x, np.sqrt(vsx**2 + vsy**2), label=r"$|\vec{\bm{v}}_{\mathrm{s}}|$", color="#00bbff", zorder=1)
    # (vsy_line,) = ax2.plot(Delta_x, vsy, "--", label=r"$v_{s,y}$", color="blue", zorder=1)
    vsx_line.set_alpha(alpha)
    # vsy_line.set_alpha(alpha)
    ax2.spines["right"].set_alpha(alpha)
    ax2.spines["top"].set_alpha(0)
    ax2.yaxis.label.set_alpha(alpha)

    # Set transparency for ax2 tick labels
    for tick in ax2.get_yticklabels():
        tick.set_alpha(alpha)

    for tick in ax2.yaxis.get_ticklines():
        tick.set_alpha(alpha)

    ax2.set_ylabel(r"Spincurrent strength $|\vec{\bm{v}}_{\mathrm{s}}|$ [nm/ns]", color="#00bbff", alpha=alpha, fontsize=12)
    ax2.tick_params("y", colors="#00bbff")

    # Create a third y-axis and plot r on it
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    (r_line,) = ax3.plot(Delta_x, r, label=r"$r_{\mathrm{\raisebox{-0.2ex}{\scriptsize Sk}}}$", color="red", zorder=2, markersize=3, markeredgewidth=0)
    r_line.set_alpha(alpha)
    ax3.set_ylim(5, 18)
    ax3.set_ylim(0.5, 2)
    ax3.set_ylabel(r"Skyrmion radius $r_{\mathrm{\raisebox{-0.025ex}{\scriptsize Sk}}}$ [nm]", color="red", alpha=alpha, fontsize=12)
    ax3.tick_params("y", colors="red")
    ax3.spines["right"].set_alpha(alpha)
    ax3.spines["top"].set_alpha(0)
    ax3.yaxis.label.set_alpha(alpha)

    # Set transparency for ax3 tick labels
    for tick in ax3.get_yticklabels():
        tick.set_alpha(alpha)

    for tick in ax3.yaxis.get_ticklines():
        tick.set_alpha(alpha)

    # Plot vsx and vsy on the first y-axis
    ax1.plot(Delta_x, angle, "o", label=r"$\theta_{\vec{\bm{F}}}$", color="#00ff02", zorder=3, markersize=3)


    # =========================================ONLY FOR COMBINED GRAPHS:=========================================
    # # make a linear fit of the values of the angle up to from Delta_x = 7 to inf
    # fit_indices = np.where(Delta_x > 7)[0]
    # c = constant_fit(Delta_x[fit_indices], angle[fit_indices])

    # logging.info(rf"$\theta_{{\infty}}={c:.4g}$")
    # logging.info(f"max angle: {np.max(angle)}")

    # # Plot the fit
    # ax1.axhline(y=c, linestyle="--", color="#00ff02", label=r"$\theta_{\vec{\bm{F}}}^{\infty}$", zorder=3)

    # ax1.axvline(x=7, linestyle="--", color="black", zorder=0, alpha=0.1)


    ax1.set_ylabel(r"Edge repulsion angle $\theta_{\vec{\bm{F}}}$ [deg]", color="#00ff02", fontsize=12)
    ax1.set_xlabel(r"Distance skyrmion center to edge $\Delta$ [nm]", fontsize=12)
    # ax1.set_ylim(77, 85)  # for martinez
    ax1.set_ylim(80, 85)
    ax1.tick_params("y", colors="#00ff02")
    ax1.spines["top"].set_alpha(0)
    ax1.spines["right"].set_alpha(0)

    ax1.set_zorder(max(ax2.get_zorder(), ax3.get_zorder()) + 1)  # Put ax1 on top
    ax1.set_frame_on(False)  # but show the other plots

    # Get the lines and labels from all axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    # # Create the legend with the lines and labels from all axes for right edge
    # legend = plt.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left", fontsize=10)

    # Create the legend with the lines and labels from all axes for right edge
    legend = plt.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper right", fontsize=10)

    for text in legend.get_texts():
        text.set_va('baseline')

    # Make the legend background transparent
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('black')  # Set the rim color to black

    # font_title = {"family": "CMU Serif", "size": 15}
    # plt.title(f"Skyrmion Drift due to Edge", fontdict=font_title)

    # # für retention rechts
    # plt.gca().invert_xaxis()

    fig.tight_layout()
    # Save the plot
    logging.info(f"Saving plot to {dest_dir}/{dest_file}")
    plt.savefig(f"{dest_dir}/{dest_file}", dpi=800, transparent=True)
    plt.close()
    # plt.xlabel("x [0.3 nm]")
    # plt.ylabel("vsx/vsy [0.3 m/s]")

    # Plot the angle on the first y-axis

    # plt.style.use("dark_background")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    plt.rcParams['font.family'] = 'serif'


def main():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(cwd))

    file_name = "traj_q.npy"
    fetch_folder_name = f"OUTPUT/Thesis_Fig_10_close_wall_ret_test_close"

    dest_folder = f"OUTPUT/trajectories/angle_distance_plot_left_edge"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    current_vs_distance_plot(fetch_folder_name, dest_folder, fetch_file=file_name)

    # # ---------------------------------------------------------------stitch together data from 2 different simulations with threashold -------------------------------------------------------------------

    # fetch_folder_1_name = f"/OUTPUT/your_folder_name"
    # fetch_folder_2_name = f"/OUTPUT/your_second_folder_name"
    # threashold = 372.5
    # dest_traj_q_file = f"OUTPUT/trajectories/traj_q_stitch.npy"

    # traj_q_1 = fetch_traj_q_file(fetch_folder_1_name)
    # keys = traj_q_1.dtype.names
    # for key in keys:
    #     traj_q_1[key] = np.append(traj_q_1[key], traj_q_2[key])

    # traj_q_2 = fetch_traj_q_file(fetch_folder_2_name)
    # # valid_indices_1 = np.where(np.logical_and(vsx != 0, True))[0][1:]

    # ---------------------------------------------------------------Trajectory, q, r triple plot-------------------------------------------------------------------


if __name__ == "__main__":
    main()
