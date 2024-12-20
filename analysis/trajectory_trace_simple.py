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
    from common_functions import setup_plt
elif __name__ == "analysis.trajectory_trace_simple":
    from analysis.common_functions import setup_plt

def setup_plt():
    matplotlib.use("Agg")

    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    plt.rcParams["font.family"] = "serif"


def trajectory_trace(
    fetch_folder_name, dest_dir, native_v_s_factor=10, v_s_to_j_s_factor=1.61e10, dest_file="trajectory3thesis.png", title="Trajectory of Skyrmion"
):
    # fetch the traj_q file from the fetch folder

    setup_plt()

    fig, ax = plt.subplots()

    # initialize with opposite values to overwrite with first ones
    y_min_global = np.inf
    y_max_global = -np.inf
    x_min_global = np.inf
    x_max_global = -np.inf

    # subfolders in fetch_folder_name
    
    list_subdirs = [dir for dir in os.listdir(fetch_folder_name) if os.path.isdir(os.path.join(fetch_folder_name, dir))]
    if len(list_subdirs) == 0:
        fetch_dirs = [fetch_folder_name]
    else:
        fetch_dirs = [f.path for f in os.scandir(fetch_folder_name) if f.is_dir()]
        
    if isinstance(fetch_dirs, str):
        fetch_dirs = [fetch_dirs]
    
    fetch_dirs.append(fetch_folder_name)

    len_fetch_dirs = len(fetch_dirs)

    logging.info(f"len_fetch_dirs: {len_fetch_dirs}")

    no_traj_q_files = 0

    for index, fetch_dir in enumerate(fetch_dirs):

        logging.info(f"fetch_dir: {fetch_dir}")

        filename = "traj_q.npy"

        pattern_file = os.path.join(fetch_dir, "**", filename)

        traj_q_file = glob.glob(pattern_file, recursive=True)

        amount_of_traj_q_files = len(traj_q_file)

        no_traj_q_files += amount_of_traj_q_files

        if amount_of_traj_q_files == 0:
            logging.info(f"index: {index}")
            logging.warning(f"No traj_q.npy file was found in dir {fetch_dir}.")
            if index == len_fetch_dirs - 1:
                logging.info(f"index: {index}")
                logging.error(f"No '{filename}' file was found in any of the subdirs of {fetch_folder_name}.")
            continue
        if amount_of_traj_q_files > 1:
            logging.error(f"There are {amount_of_traj_q_files} traj_q.npy files in dir {fetch_dir}\n\
                            totalling to {no_traj_q_files} '{filename}' files in all subdirs.")
            # continue

        # # error catching
        # if amount_of_traj_q_files != 1:
        #     logging.error(f"There should be exactly one traj_q.npy file in dir {fetch_dir} and subdirs.")
        #     if amount_of_traj_q_files == 0:
        #         logging.error("No traj_q.npy file was found.")
        #     elif amount_of_traj_q_files > 1:
        #         logging.error(f"{amount_of_traj_q_files} traj_q.npy files were found.")
        #     exit()

        # # fetch the v_s_sample_fac from fetch_dir_name
        # fetch_dir_parts = fetch_dir.split("_")
        # try:
        #     v_s_fac_index = fetch_dir_parts.index("v") - 1
        #     v_s_fac = float(fetch_dir_parts[v_s_fac_index])
        # except:
        #     logging.warning("v not found in fetch_dir_name, taking v_s_fac as 1")
        #     v_s_fac = 1
        try:
            filename = "v_s_fac_and_wall_angle.npy"
            v_s_fac = np.load(os.path.join(fetch_dir, filename))[0]
        except:
            logging.warning("v_s_fac not found in fetch_dir, taking v_s_fac as 1")
            v_s_fac = 1

        j_s = v_s_fac * native_v_s_factor * v_s_to_j_s_factor
        logging.info(f"j_s: {j_s}")

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

        # get the x and y dim
        field_size_x = racetrack.shape[0]
        field_size_y = racetrack.shape[1]

        # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

        q = traj_q["topological_charge"]
        # delete the indices in traj_q where abs(topo charge) is not close to 0 (+- 0.1)
        sim_finished = np.where(np.logical_or(abs(q) < 0.9, abs(q) > 1.1))
        traj_q = np.delete(traj_q, sim_finished)

        # Extract data fields
        x = traj_q["x0"]
        y = traj_q["y0"]
        """
        # # make an equal spacing of 5 points between 0 and len(traj_q["time"])
        # print(x)
        # indices_pictures = np.linspace(0, len(traj_q["time"]) - 1, min(5, len(traj_q["time"])), dtype=int)
        # logging.info(f"Indices for the images: {indices_pictures}")

        # # set the locations of the images
        # x_pictures = traj_q["x0"][indices_pictures]
        # y_pictures = traj_q["y0"][indices_pictures]

        # # make list for images --> list of np arrays
        # image = []

        # # fetch the images
        # for index in indices_pictures:
        #     image_path_pattern = os.path.join(fetch_dir, "**", f"spinfield_t_{traj_q[index]['time']:011.6f}.png")

        #     # get the path
        #     image_path = glob.glob(image_path_pattern, recursive=True)

        #     # load the image
        #     logging.info(f"image_path: {image_path}")
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
        """

        # plt.style.use("dark_background")

        # plt.rcParams["font.family"] = "CMU Serif"
        # plt.rcParams["font.serif"] = "CMU Serif Roman"

        # to scale with the triangular lattice structure
        y_factor = np.sqrt(3) / 2 * 0.27
        x_factor = 0.27
        y *= y_factor
        x *= x_factor

        # calculate plt.ylim and plt.xlim based on max and min of x and y, but make graph rectangular, so same range for x and y
        x_min_local = np.min(x)
        x_max_local = np.max(x)
        y_min_local = np.min(y)
        y_max_local = np.max(y)

        x_min_new_local = x_min_local - 10
        x_max_new_local = x_max_local + 10
        y_min_new_local = y_min_local - 10
        y_max_new_local = y_max_local + 10

        # replace the new value of max/mins if smaller/bigger than the global ones
        if x_min_new_local < x_min_global:
            x_min_global = x_min_new_local
        if x_max_new_local > x_max_global:
            x_max_global = x_max_new_local
        if y_min_new_local < y_min_global:
            y_min_global = y_min_new_local
        if y_max_new_local > y_max_global:
            y_max_global = y_max_new_local

        # choose color based on index / len_fetch_dirs times around the color wheel
        color = plt.get_cmap("hsv")(index / len_fetch_dirs)

        # plot the trajectory of the current skyrmion
        ax.plot(x, y, "o", color=color, markersize=4, label=f"Trajectory" + r" $j_s = %.2f \cdot 10^{10}$" % (j_s / 1e10))

    x_range = x_max_global - x_min_global
    y_range = y_max_global - y_min_global

    aspect_ratio_local = x_range / y_range  #

    # # Add an inset Axes with the desired aspect ratio
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], aspect=aspect_ratio)
    ax.set_box_aspect(1 / aspect_ratio_local)

    # Set the limits with a padding of 10 units
    ax.set_xlim([0, field_size_x * x_factor])
    ax.set_ylim([0, field_size_y * y_factor])

    # # Set the limits with a padding of 10 units
    # ax.set_xlim([x_min_global - 10, x_max_global + 10])
    # ax.set_ylim([y_min_global - 10, y_max_global + 10])

    # font_title = {"family": "CMU Serif", "color": "black", "size": 20}

    plt.title(title)
    plt.legend(loc="upper center")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")

    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{dest_dir}/{dest_file}", dpi=800)
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

    fetch_folder_name = f"../OUTPUT/Thesis_Fig_10_close_2_wall_ret_test_close"

    dest_folder = f"../OUTPUT/trajectories"

    native_v_s_factor = 10  # 200 v_s_factor * 0.05 (s. num methods)

    v_s_to_j_s_factor = 1.61e10

    # j_s = np.array([1, 2, 4, 8, 16]) * 4.025e10

    # print(subfolders)

    # for i, subfolder in enumerate(subfolders):

    # trajectory_trace(subfolders, dest_folder, native_v_s_factor, v_s_to_j_s_factor, dest_file="trajectory.png", title="Trajectories")

    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/PyMMF/OUTPUT/wall_ret_test_middle_wall_ret_test"
    # dest_folder = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/PyMMF/OUTPUT/trajectories"

    # fetch dirs not files in fetch_folder_name
    for entry in os.scandir(fetch_folder_name):
        if entry.is_dir():
            fetch_folder = os.path.join(fetch_folder_name, entry.name)
            logging.info(f"fetch_folder: {fetch_folder}")
            dest_file = f"trajectory_{entry.name}.png"
            try:
                trajectory_trace(fetch_folder_name, dest_folder, native_v_s_factor, v_s_to_j_s_factor, dest_file=dest_file, title="Trajectories")
            except:
                logging.error(f"Error for {entry.name}")

    # ---------------------------------------------------------------Trajectory, q, r triple plot-------------------------------------------------------------------


if __name__ == "__main__":
    main()

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
    #     plt.xlim([x_min - 10, x_max + 10])
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
    #     plt.xlim([new_x_min - 10, new_x_max + 10])

    # Calculate ranges and midpoints for x and y
    # x_mid = (x_max + x_min) / 2
    # y_mid = (y_max + y_min) / 2
    # x_range = x_max - x_min
    # y_range = y_max - y_min

    # Determine the maximum range and calculate the new limits
    # max_range = max(x_range, y_range)

    # # correct for aspect ratio
    # plt.gca().set_aspect(1 / aspect_ratio, adjustable="box")

    # # Set the new limits
    # plt.xlim([x_min_new, x_max_new])
    # plt.ylim([y_min_new, y_max_new])

    # # Adjust the new limits if they're out of bounds
    # x_min_new = x_min_new
    # y_min_new = y_min_new
    # x_max_new = x_max_new
    # y_max_new = y_max_new

    # draw lines

    # log the new limits

    # plt.xlim([240, 580])
    # plt.ylim([60, 400])
    # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
