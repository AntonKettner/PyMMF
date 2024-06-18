# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
from PIL import Image
import matplotlib
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


def create_q_r_vs_time_plot(fetch_dir, dest_dir, dest_file="q_r_vs_time.png"):
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

    # print(traj_q.dtype.names)
    # print(traj_q["topological_charge"])

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

    # get the x and y dim
    # field_size_x = racetrack.shape[0]
    # field_size_y = racetrack.shape[1]

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    # # Extract data fields
    # x = traj_q["x0"]
    # y = traj_q["y0"]

    # # make an equal spacing of 5 points between 0 and len(traj_q["time"])
    # indices_pictures = np.linspace(0, len(traj_q["time"]) - 1, 5, dtype=int)

    # # set the locations of the images
    # x_pictures = traj_q["x0"][indices_pictures]
    # y_pictures = traj_q["y0"][indices_pictures]

    # # make list for images --> list of np arrays
    # image = []

    # # fetch the images
    # for index in indices_pictures:
    #     image_path_pattern = os.path.join(
    #         fetch_dir, "**", f"spinfield_t_{traj_q[index]['time']:011.6f}.png"
    #     )

    #     # get the path
    #     image_path = glob.glob(image_path_pattern, recursive=True)

    #     # load the image
    #     img = np.array(Image.open(image_path[0]))

    #     image.append(img)

    # # Define the range for cropping around the skyrmion
    # crop_ranges = (traj_q["r1"][indices_pictures] * 1.3 + 5).astype(np.int32)

    # # crop the images around the skyrmion positions
    # skyrs = []

    # x_converted = (field_size_y - y_pictures).astype(np.int32)
    # y_converted = (x_pictures).astype(np.int32)

    # for i in range(len(image)):
    #     skyrs.append(
    #         image[i][
    #             x_converted[i] - crop_ranges[i] : x_converted[i] + crop_ranges[i],
    #             y_converted[i] - crop_ranges[i] : y_converted[i] + crop_ranges[i],
    #         ]
    #     )

    # plt.figure()

    # plt.style.use("dark_background")

    # for i in range(len(skyrs)):
    #     plt.imshow(
    #         skyrs[i],
    #         extent=[
    #             x_pictures[i] - crop_ranges[i],
    #             x_pictures[i] + crop_ranges[i],
    #             y_pictures[i] - crop_ranges[i],
    #             y_pictures[i] + crop_ranges[i],
    #         ],
    #         aspect="auto",
    #     )

    
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    # plt.rcParams['font.family'] = 'serif'


    # plt.plot(x, y, "ro", label="Trajectory")

    # # for linear fit
    # a, b, fit = linear_fit(x, y)
    # plt.plot(
    #     x,
    #     fit,
    #     "g-",
    #     label=f"Linear Fit,\na = {float(a):.5g},\nb = {float(b):.5g}",
    # )
    # print(f"a: {a}")

    # font_title = {"family": "CMU Serif", "color": "white", "size": 25}

    # plt.title("Graph of Trajectory (center of skyr)", fontdict=font_title)
    # plt.legend(loc="upper left")
    # plt.xlabel("x [0.3 nm]")
    # plt.ylabel("y [0.3 nm]")

    # # calculate plt.ylim and plt.xlim based on max and min of x and y, but make graph rectangular, so same range for x and y
    # x_min = np.min(x)
    # x_max = np.max(x)
    # y_min = np.min(y)
    # y_max = np.max(y)

    # x_range = x_max - x_min
    # y_range = y_max - y_min

    # if x_range > y_range:
    #     new_y_min = y_min - 0.5 * (x_range - y_range)
    #     new_y_max = y_max + 0.5 * (x_range - y_range)
    #     if new_y_min < 0:
    #         new_y_max += abs(new_y_min)
    #         new_y_min = 0
    #     if new_y_max > field_size_y:
    #         new_y_min -= new_y_max - field_size_y
    #         new_y_max = field_size_y
    #     plt.ylim([new_y_min - 10, new_y_max + 10])
    #     plt.xlim([x_min, x_max])
    # else:
    #     plt.ylim([y_min - 10, y_max + 10])
    #     new_x_min = x_min - 0.5 * (y_range - x_range)
    #     new_x_max = x_max + 0.5 * (y_range - x_range)
    #     if new_x_min < 0:
    #         new_x_max += abs(new_x_min)
    #         new_x_min = 0
    #     if new_x_max > field_size_x:
    #         new_x_min -= new_x_max - field_size_x
    #         new_x_max = field_size_x
    #     plt.xlim([new_x_min, new_x_max])

    # # plt.xlim([240, 580])
    # # plt.ylim([60, 400])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_dir}/{dest_file}", dpi=800)
    # plt.close()

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
