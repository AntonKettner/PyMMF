# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from common_functions import linear_fit
elif __name__ == "analysis.trajectory_trace_new_wall_repulsion":
    from analysis.common_functions import linear_fit


def make_wall_repulsion_plot(fetch_dir, dest_dir, dest_file="wall_repulsion_fit.png"):
    # fetch the traj_q file from the fetch folder
    filename = "traj_q.npy"

    pattern_file = os.path.join(fetch_dir, "**", filename)

    traj_q_file = glob.glob(pattern_file, recursive=True)

    amount_of_traj_q_files = len(traj_q_file)

    # error catching
    if amount_of_traj_q_files != 1:
        logging.error("There should be exactly one traj_q.npy file.")
        if amount_of_traj_q_files == 0:
            logging.error(f"No traj_q.npy file was found at {pattern_file}.")
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
            logging.error(
                f"{amount_of_race_track_files} {racetrack_name} files were found."
            )
        exit()
    racetrack = np.ascontiguousarray(
        np.array(np.array(Image.open(racetrack_files[0]), dtype=bool)[:, :, 0]).T[
            :, ::-1
        ]
    )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis

    # get the x and y dim
    field_size_x = racetrack.shape[0]
    field_size_y = racetrack.shape[1]

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    # Find the indices for the images that should be fetched -> equal spacing from max x

    x = traj_q["x0"]
    index_0 = np.argmax(x)

    traj_q = traj_q[index_0:]
    x = traj_q["x0"]

    # Extract data fields
    y = traj_q["y0"]
    # q = traj_q["topological_charge"]
    # r = traj_q["r1"]

    # make an equal spacing of 5 points between 0 and len(traj_q["time"])
    indices_pictures = np.linspace(0, len(traj_q["time"]) - 1, 5, dtype=int)

    # set the locations of the images
    x_pictures = traj_q["x0"][indices_pictures]
    y_pictures = traj_q["y0"][indices_pictures]

    # make list for images --> list of np arrays
    image = []

    # fetch the images
    for index in indices_pictures:
        image_path_pattern = os.path.join(
            fetch_dir, "**", f"spinfield_t_{traj_q[index]['time']:011.6f}.png"
        )

        # get the path
        image_path = glob.glob(image_path_pattern, recursive=True)

        # load the image
        img = np.array(Image.open(image_path[0]))

        image.append(img)

    # Define the range for cropping around the skyrmion
    crop_ranges = (traj_q["r1"][indices_pictures] * 1.8).astype(np.int32)

    # crop the images around the skyrmion positions
    skyrs = []

    x_converted = (field_size_y - y_pictures).astype(np.int32)
    y_converted = (x_pictures).astype(np.int32)

    for i in range(len(image)):
        skyrs.append(
            image[i][
                x_converted[i] - crop_ranges[i] : x_converted[i] + crop_ranges[i],
                y_converted[i] - crop_ranges[i] : y_converted[i] + crop_ranges[i],
            ]
        )

    plt.figure()

    plt.style.use("dark_background")

    for i in range(len(skyrs)):
        plt.imshow(
            skyrs[i],
            extent=[
                x_pictures[i] - crop_ranges[i],
                x_pictures[i] + crop_ranges[i],
                y_pictures[i] - crop_ranges[i],
                y_pictures[i] + crop_ranges[i],
            ],
            aspect="auto",
        )
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    plt.rcParams['font.family'] = 'serif'

    plt.plot(x, y, "ro", label="Trajectory")

    # for linear fit
    a, b, fit = linear_fit(x, y)
    plt.plot(
        x,
        fit,
        "g-",
        label=f"Linear Fit,\na = {float(a):.5g},\nb = {float(b):.5g}",
    )
    print(f"a: {a}")

    # font_title = {"family": "CMU Serif", "color": "white", "size": 25}

    plt.title("Graph of Trajectory (center of skyr)")
    plt.legend(loc="upper left")
    plt.xlabel("x [0.3 nm]")
    plt.ylabel("y [0.3 nm]")

    # calculate plt.ylim and plt.xlim based on max and min of x and y, but make graph rectangular, so same range for x and y
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        new_y_min = y_min - 0.5 * (x_range - y_range)
        new_y_max = y_max + 0.5 * (x_range - y_range)
        if new_y_min < 0:
            new_y_max += abs(new_y_min)
            new_y_min = 0
        if new_y_max > field_size_y:
            new_y_min -= new_y_max - field_size_y
            new_y_max = field_size_y
        plt.ylim([new_y_min - 10, new_y_max + 10])
        plt.xlim([x_min, x_max])
    else:
        plt.ylim([y_min - 10, y_max + 10])
        new_x_min = x_min - 0.5 * (y_range - x_range)
        new_x_max = x_max + 0.5 * (y_range - x_range)
        if new_x_min < 0:
            new_x_max += abs(new_x_min)
            new_x_min = 0
        if new_x_max > field_size_x:
            new_x_min -= new_x_max - field_size_x
            new_x_max = field_size_x
        plt.xlim([new_x_min, new_x_max])

    # plt.xlim([240, 580])
    # plt.ylim([60, 400])
    # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_dir}/{dest_file}", dpi=800)
    plt.close()


def main():
    matplotlib.use("Agg")
    # set these vars
    boundary = "open"  # "ferro", "open"
    vs = 0  # 15, 8
    B_ext = 1.5  # 1.5, 1.15

    # fetch_folder_name = f"OUTPUT/results_x_current_dynamic_center_start_{boundary}_boundary_heun_{B_ext}_B_ext_{vs}_vs"

    # boundary = "open"  # "ferro", "open"
    # vs = 8  # 15, 8
    # B_ext = 1.5  # 1.5, 1.15

    orig_cwd = os.getcwd()
    os.chdir("../../ongoing_work")
    logging.info(f"Current working directory: {os.getcwd()}")
    fetch_folder_name = "OUTPUT/results_wall_retention_new_open_heun_1.5_25_83.8/sample_1_83.8_deg_25_v_s_fac"

    dest_folder = f"../PyMMF/OUTPUT/trajectories"

    make_wall_repulsion_plot(fetch_folder_name, dest_folder)

    # ---------------------------------------------------------------Trajectory, q, r triple plot-------------------------------------------------------------------


if __name__ == "__main__":
    main()
