import numpy as np

import time

from tqdm import tqdm

from PIL import Image

# import matplotlib
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import curve_fit

import os

import glob

from sklearn import svm

import logging  # enabling display of logging.info messages


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



v_s_factor = 10


a = 0.27


phase_diag_temp_dir = "OUTPUT/phase_diag_temp"
if not os.path.exists(phase_diag_temp_dir):
    os.makedirs(phase_diag_temp_dir)



def fetch_traj_q_file(fetch_file, neg_index=1):

    # fetch the traj_q file from the fetch folder
    traj_q_file = glob.glob(fetch_file, recursive=True)

    amount_of_traj_q_files = len(traj_q_file)

    # error catching
    if amount_of_traj_q_files != 1:
        logging.error("There should be exactly one traj_q.npy file.")
        if amount_of_traj_q_files == 0:
            if neg_index == 1:
                logging.error("No traj_q.npy file was found in the LAST folder --> ongoing work there.")
            else:
                logging.error(f"No traj_q.npy file was found at {fetch_file}.")
        elif amount_of_traj_q_files > 1:
            logging.error(f"{amount_of_traj_q_files} traj_q.npy files were found.")
        return None
        # exit()

    return np.load(traj_q_file[0])


def get_subdirs(dir_paths):
    subdir_paths = []
    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths
    for dir_path in dir_paths:
        subdir_paths.extend(glob.iglob(os.path.join(dir_path, '*/')))
    return subdir_paths


def fetch_racetrack(fetch_dir, racetrack_name="racetrack.png"):
    # create search pattern for racetrack file
    racetrack_pattern = os.path.join(fetch_dir, "**", racetrack_name)

    racetrack_files = glob.glob(racetrack_pattern, recursive=True)
    amount_of_race_track_files = len(racetrack_files)
    if amount_of_race_track_files != 1:
        logging.error(f"There should be exactly one {racetrack_name} file.")
        if amount_of_race_track_files == 0:
            logging.error(f"No {racetrack_name} file was found in {fetch_dir}.")
        elif amount_of_race_track_files > 1:
            logging.error(f"{amount_of_race_track_files} {racetrack_name} files were found.")
        exit()
    racetrack = np.ascontiguousarray(
        np.array(np.array(Image.open(racetrack_files[0]), dtype=bool)[:, :, 0]).T[:, ::-1]
    )  # [:,:,0] for rgb to grayscale, .T for swapping x and y axis, [::-1] for flipping y axis
    return racetrack


def filter_points(points, x_range, y_range):
    if x_range[0] is not None:
        points = points[points[:, 0] > x_range[0]]
    if x_range[1] is not None:
        points = points[points[:, 0] < x_range[1]]
    if y_range[0] is not None:
        points = points[points[:, 1] > y_range[0]]
    if y_range[1] is not None:
        points = points[points[:, 1] < y_range[1]]
    return points


def label_wrap_and_filter(datapoints, regions, lims=np.array([[None, None], [None, None]])):
    regionpoints = []
    for region_index, region in enumerate(regions):
        v_s_data = np.array([])
        angle_data = np.array([])
        for r in region:
            # concatenate the data points for velocity and angle
            v_s_data = np.concatenate((v_s_data, datapoints[r + "_v_s"]), axis=0)
            angle_data = np.concatenate((angle_data, datapoints[r + "_angle"]), axis=0)
        regionpoints.append(np.column_stack((v_s_data, angle_data)))
    labels = [1] * len(regionpoints[0]) + [-1] * len(regionpoints[1])
    data = np.vstack((regionpoints[0], regionpoints[1]))
    return data, labels, regionpoints


def setup_plt():

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
    plt.rcParams['font.family'] = 'serif'

def create_plot_with_boundaries(datapoints, isatomicstep, dest_boundary_dir, plotlims):
    """
    entries in datapoints:
    - end_reach_v_s
    - end_reach_angle
    - repelled_back_v_s
    - repelled_back_angle
    - stays_wall_v_s
    - stays_wall_angle
    - destr_wall_v_s
    - destr_wall_angle
    """
    setup_plt()
    plt.figure()
    if isatomicstep:
        # 4 regions -->         top left,       middle left,    top and middle right,   entire bottom
        # represented by -->    repelled_back,  stays_wall,     destr_wall,             end_reach

        separation_areas = [
            [
        ["repelled_back"], 
        ["stays_wall"],
            ],
            [
        ["repelled_back", "stays_wall"],
        ["destr_wall"],
            ],
            [
        ["destr_wall", "stays_wall", "repelled_back"],
        ["end_reach"],
            ],
        ]
    else:
        # 3 regions -->         top left,       top right,    bottom
        # represented by -->    repelled_back,  destr_wall,   end_reach
        
        separation_areas = [
            [
        ["repelled_back", "end_reach"],
        ["destr_wall"],
            ],
            [
        ["repelled_back"],
        ["end_reach"],
            ],
        ]


    boundaries = []

    for index, regions in enumerate(separation_areas):
        region1, region2 = regions
        logging.info(f"region1: {region1}")
        logging.info(f"region2: {region2}")
        regions = [region1, region2]
        


        # # now separate regions with phase lines:
        # # 1. separate repell_back from stays_wall
        # region1 = ["repelled_back"]
        # region2 = ["stays_wall"]
        # regions = [region1, region2]
        data, labels, regionpoints = label_wrap_and_filter(datapoints, regions)
        
        # # define limits for the plot
        # x_lim_low = np.min()
        lims = np.array([[None, None], [None, None]])
        
        # Train SVM
        if index == 0:
            clf = svm.SVC(kernel="rbf", C=1e6)
        elif index == 1:
            clf = svm.SVC(kernel="rbf", C=1e6)
        elif index == 2:
            clf = svm.SVC(kernel="rbf", C=1e6)
        clf.fit(data, labels)
        
        # Visualization
        alpha=0
        # if index == 1:
        #     alpha = 1
        plt.scatter(regionpoints[0][:, 0], regionpoints[0][:, 1], color="r", alpha=alpha, label="Phase 1")
        plt.scatter(regionpoints[1][:, 0], regionpoints[1][:, 1], color="b", alpha=alpha, label="Phase 2")

        # get decision boundary
        ax = plt.gca()
        xlim = lims[0]
        ylim = lims[1]
        xlim_ax = ax.get_xlim()
        ylim_ax = ax.get_ylim()
        xlim = (xlim[0] if xlim[0] is not None else xlim_ax[0], xlim[1] if xlim[1] is not None else xlim_ax[1])
        ylim = (ylim[0] if ylim[0] is not None else ylim_ax[0], ylim[1] if ylim[1] is not None else ylim_ax[1])

        logging.info(f"xlim: {xlim}")
        logging.info(f"ylim: {ylim}")

        # Create grid to show the result and evaluate the model
        gridsize = 500
        xx = np.linspace(xlim[0], xlim[1], gridsize)
        yy = np.linspace(ylim[0], ylim[1], gridsize)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # 3. Extract points from the decision boundary
        alpha = 0
        boundary = ax.contour(XX, YY, Z, levels=[0], colors='k', alpha=alpha, linestyles='-', linewidths=2)
        boundary_points = boundary.collections[0].get_paths()[0].vertices
        """
        # logging.info(f"boundary_points: {boundary_points}")
        # coeffs = np.polyfit(boundary_points[:, 0], boundary_points[:, 1], 2)
        # logging.info(f"coeffs: {coeffs}")

        # # gather data points from the coeffs
        # x_poly = np.linspace(xlim[0], xlim[1], 100)
        # y_poly = coeffs[0] * x_poly**2 + coeffs[1] * x_poly + coeffs[2]

        # # fit a 4th degree polynomial, x^4 having a negative coefficient
        # coeffs = np.polyfit(boundary_points[:, 0], boundary_points[:, 1], 4)
        # logging.info(f"coeffs: {coeffs}")

        # # gather data points from the coeffs
        # x_poly = np.linspace(xlim[0], xlim[1], 100)
        # y_poly = coeffs[0] * x_poly**4 + coeffs[1] * x_poly**3 + coeffs[2] * x_poly**2 + coeffs[3] * x_poly + coeffs[4]

        # # plot the polynomial
        # plt.plot(x_poly, y_poly, color="k", label="Decision boundary x^4 fit")
        """        
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1], "-", color="k", label="Decision boundary points")

        boundaries.append(boundary_points)





    colors = ['red', 'yellow', 'cyan', (0,0.9,0)]  # Add more colors if needed
    alpha = 1
    if isatomicstep:
        plt.fill_between(boundaries[0][:, 0], boundaries[0][:, 1], ylim[1], color=colors[2], alpha=alpha)
        plt.fill_between(boundaries[0][:, 0], 0, boundaries[0][:, 1], color=colors[1], alpha=alpha)
        plt.fill_betweenx(boundaries[1][:, 1], boundaries[1][:, 0], xlim[1], color=colors[0], alpha=alpha)
        plt.fill_between(boundaries[2][:, 0], 0, boundaries[2][:, 1], color=colors[3], alpha=alpha)
    else:
        plt.fill_between(xlim, ylim[0], ylim[1], color=colors[3], alpha=alpha)
        plt.fill_between(boundaries[1][:, 0], boundaries[1][:, 1], ylim[1], color=colors[2], alpha=alpha)
        plt.fill_between(boundaries[0][:, 0], boundaries[0][:, 1], ylim[1], color=colors[0], alpha=alpha)
    # for idx, boundary_points in enumerate(boundaries):
    #     color = colors[idx % len(colors)]  # Cycle through colors
    #     plt.plot(boundary_points[:, 0], boundary_points[:, 1], "-", color=color, label=f"Boundary {idx+1}")

    plt.plot(datapoints["end_reach_v_s"], datapoints["end_reach_angle"], "o", linewidth=4, markersize=4, label="survives, passes", color=(0, 0.4, 0))  # dark green
    plt.plot(datapoints["repelled_back_v_s"], datapoints["repelled_back_angle"], "o", linewidth=4, markersize=4, label="survives, bounces back", color="blue")   # blue
    plt.plot(datapoints["stays_wall_v_s"], datapoints["stays_wall_angle"], "o", linewidth=4, markersize=4, label="survives, gets stuck", color=(1.0, 0.5, 0.0)) # orange
    plt.plot(datapoints["destr_wall_v_s"], datapoints["destr_wall_angle"], "^", linewidth=4, markersize=4, label="destroyed on wall", color=(1.0, 0.5, 0.0))    # orange

    legend = plt.legend(fontsize='small')
    legend.remove()
    logging.info(f"dest_boundary_dir: {dest_boundary_dir}")
    
    ax.set_ylim(plotlims[0])
    ax.set_xlim(plotlims[1])
    plt.xlabel(r"Current strength $|\vec{\bm{v}}_{\mathrm{s}}|$ [nm/ns]", fontsize=14)
    plt.ylabel(r"Current angle $\varphi_{\mathrm{s}}$ [deg]", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(dest_boundary_dir, dpi=800, transparent=True)
    # plt.show()
    plt.close()

    legend_fig = plt.figure()
    
    # Add the legend to the new figure
    legend_fig.legend(handles=legend.legend_handles, labels=[t.get_text() for t in legend.texts], facecolor='none')

    # Save the legend figure
    # replace filename with newfilename
    dest_dir_legend_boundary = os.path.join(os.path.dirname(dest_boundary_dir), "legend_boundary.png")

    legend_fig.savefig(dest_dir_legend_boundary, dpi=800, transparent=True)

    # Don't forget to close the legend figure when you're done with it
    plt.close(legend_fig)


def create_phase_diagram_v_s_factor_vs_v_s_angle(fetch_dir, dest_dir, dest_file, fetch_file="traj_q.npy"):

    setup_plt()



    # traj_q_pattern = os.path.join(fetch_dir, "**", fetch_file)

    # for wall angled, v_s_x = const, v_s_y = 0
    x_threashold_back = 450
    x_threashold_mid = 180

    # for v_s_angled not wall angled
    x_threashold_back = 40
    x_threashold_mid = 400

    q_variance = 0.2
    q_ideal = -1

    # traj_q_dirs = list(glob.iglob(traj_q_pattern, recursive=True))
    traj_q_dirs = get_subdirs(fetch_dir)

    total_files = len(traj_q_dirs)

    # min_v_s_fac = None
    # min_v_s_angle = None
    # max_v_s_fac = None
    # max_v_s_angle = None

    
    # # check if file in temp exists with the same mask and calc_steps as metadata
    # for i in range(100):
    #     temp_dir = f"{neumann_temp_dir}/temp_{field_size_x}x{field_size_y}_{i}.npz"
    #     if os.path.exists(temp_dir):
    #         with np.load(temp_dir) as data:
    #             if np.array_equal(data["mask"], mask) and np.array_equal(data["steps"], steps):
    #                 logging.warning(f"temp file found with the same calculation steps as metadata aswell")
    #                 return data["phi"], data["j"]
    #     if i == 99:
    #         logging.warning(f"no temp file found with same mask and calculation steps as metadata, calculating new data")

    # check if file in temp exists with the same mask and calc_steps as metadata
    for i in range(100):
        temp_dir = f"{phase_diag_temp_dir}/temp_phase_dir_{i}.npz"
        if os.path.exists(temp_dir):
            with np.load(temp_dir) as data:
                if np.array_equal(data["fetch_dir"], fetch_dir):
                    logging.info(f"PHASE DIAG DIR WITH CORRECT DATA FOUND IN TEMP DIRECTORY\n")
                    stays_back_v_s = data["stays_back_v_s"]
                    stays_back_angle = data["stays_back_angle"]
                    stays_wall_v_s = data["stays_wall_v_s"]
                    stays_wall_angle = data["stays_wall_angle"]
                    stays_start_v_s = data["stays_start_v_s"]
                    stays_start_angle = data["stays_start_angle"]
                    stays_slow_v_s = data["stays_slow_v_s"]
                    stays_slow_angle = data["stays_slow_angle"]
                    destr_back_v_s = data["destr_back_v_s"]
                    destr_back_angle = data["destr_back_angle"]
                    destr_wall_v_s = data["destr_wall_v_s"]
                    destr_wall_angle = data["destr_wall_angle"]
                    destr_start_v_s = data["destr_start_v_s"]
                    destr_start_angle = data["destr_start_angle"]
                    temp_dir_found = True
                    break
        if i == 99:
            logging.info(f"no temp file found with same mask and calculation steps as metadata, now searching through traj_qs\n")
            stays_back_v_s = np.array([])
            stays_back_angle = np.array([])
            stays_wall_v_s = np.array([])
            stays_wall_angle = np.array([])
            stays_start_v_s = np.array([])
            stays_start_angle = np.array([])
            stays_slow_v_s = np.array([])
            stays_slow_angle = np.array([])

            destr_back_v_s = np.array([])
            destr_back_angle = np.array([])
            destr_wall_v_s = np.array([])
            destr_wall_angle = np.array([])
            destr_start_v_s = np.array([])
            destr_start_angle = np.array([])
            temp_dir_found = False
                
    # to acquire the stays_wall etc from the fetch_folders
    for traj_q_idx in tqdm(range(total_files)):
        if temp_dir_found:
            break

        traj_q_dir = os.path.join(traj_q_dirs[traj_q_idx], fetch_file)

        traj_q = fetch_traj_q_file(traj_q_dir, total_files - traj_q_idx)

        if traj_q is None:
            logging.info(f"traj_q is None in {traj_q_dir}")
            continue
        
        t = traj_q["time"]
        sim_finished = np.where(t==0)
        traj_q = np.delete(traj_q, sim_finished)

        # # possibility for further condition
        # condition = True
        # valid_indices = np.where(np.logical_and(t != 0, condition))[0]
        # filtered_traj_q = traj_q[~np.all(traj_q == 0, axis=1)]

        # Extract data fields
        q = traj_q["topological_charge"]
        x = traj_q["x0"]
        y = traj_q["y0"]
        t = traj_q["time"]

        # fetch the spinfield from the fetch folder
        racetrack = fetch_racetrack(traj_q_dirs[traj_q_idx])
        

        # get the x and y dim
        field_size_x = racetrack.shape[0]
        field_size_y = racetrack.shape[1]

        # logging.info(f"field_size_x, field_size_y: {field_size_x}, {field_size_y}")

        pass_location_low = field_size_x
        pass_location_high = field_size_x

        # if string contains section "atomic_step"
        if "atomic_step" in traj_q_dir:

            pass_location_low = 180
            pass_location_high = 210


        # x_threashold_back = pass_location

        dirname = os.path.dirname(traj_q_dir)

        v_s_fac_and_wall_angle = np.load(
            os.path.join(dirname, "v_s_fac_and_wall_angle.npy")
        )
        v_s_fac = v_s_fac_and_wall_angle[0] * -1 * v_s_factor * a
        wall_angle = v_s_fac_and_wall_angle[1] * -1

        try:
            q_init = q[0]
        except:
            logging.info(f"traj_q_dir: {traj_q_dir}")
            continue
        q_sum_last = q[-1]
        q_last = q_sum_last
        t_last = t[-1]
        x_last = x[-1]
        y_last = y[-1]
        x_7_back = x[-7]
        y_7_back = y[-7]
        x_before_last = x[-2]
        x_8_back =x[-8]
        end_times = (10, 30, 50)
        
        if len(q) < 7:
            logging.warning(f"q_last: {q_last} probably wrong simulation from the start for v_s_fac {v_s_fac} and wall_angle {wall_angle}")
            logging.info(f"assuming destuction")
            destr_wall_v_s = np.append(destr_wall_v_s, v_s_fac)
            destr_wall_angle = np.append(destr_wall_angle, wall_angle)
            continue

        if False:
            logging.info(f"v_s_fac: {v_s_fac}")
            logging.info(f"wall_angle: {wall_angle}")
            logging.info(f"SAMPLE {traj_q_idx}")
            logging.info(f"t_last: {t_last}")
            logging.info(f"x_last: {x_last}")
            logging.info(f"q_last: {q_last}")

        q_variance = 0.2

        # TODO: CHECK WHY NECESSARY
        # Skyrmion is moving until end
        if t_last not in end_times:
            indices = np.where((q >= -1 - q_variance) & (q <= -1 + q_variance))[0]
            last_index = indices[-1] if len(indices) > 0 else None
            x_last = x[last_index]
            x_before_last = x[last_index - 1]
        # if t_last == 20:
        #     print(file)

        movement_threshold = 0.0001
        slow_movement_threshold = 0.1

        negative_velocity = (x_last - x_before_last < -movement_threshold)
        
        positive_velocity = (x_last - x_before_last > movement_threshold)

        slow_velocity = (abs(x_last - x_before_last) < slow_movement_threshold)
        # bool representing if the skyrmion is moving to the right

        # if v_s_fac < 5:

        survives = q_ideal + q_variance > q_last > q_ideal - q_variance

        if not survives:
            if x_last < x_threashold_back:
                destr_back_v_s = np.append(destr_back_v_s, v_s_fac)
                destr_back_angle = np.append(destr_back_angle, wall_angle)
                # if v_s_fac > 50 and wall_angle > 8:
                #     logging.info(f"v_s_fac, wall_angle, x_last: {v_s_fac}, {wall_angle}, {x_last}")
                if False:
                    logging.info("destroyed after passing")
            elif x_last < x_threashold_mid:
                destr_wall_v_s = np.append(destr_wall_v_s, v_s_fac)
                destr_wall_angle = np.append(destr_wall_angle, wall_angle)
                # if v_s_fac > 58:
                #     logging.info(f"v_s_fac, wall_angle: {v_s_fac}, {wall_angle}")
                #     logging.info(f"x_last: {x_last}")
                if False:
                    logging.info("destroyed on wall")
            else:
                destr_start_v_s = np.append(destr_start_v_s, v_s_fac)
                destr_start_angle = np.append(destr_start_angle, wall_angle)
                if False:
                    logging.info("destroyed at start")
        # if it survives
        else:
            # if it passes the wall and survives
            if x_last < x_threashold_back:
                stays_back_v_s = np.append(stays_back_v_s, v_s_fac)
                stays_back_angle = np.append(stays_back_angle, wall_angle)
                if False:
                    logging.info("survives, passes")
            elif x_last > x_threashold_mid:
                stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
                stays_start_angle = np.append(stays_start_angle, wall_angle)
                if False:
                    logging.info("survives, bounces back")
            else:
                if slow_velocity and pass_location_low < x_last < pass_location_high:
                    # logging.info(f"x_last, v_s_fac, v_s_angle: {x_last}, {v_s_fac}, {wall_angle}")
                    stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
                    stays_wall_angle = np.append(stays_wall_angle, wall_angle)
                elif negative_velocity:
                    # logging.info(f"x_last, pass_location: {x_last}, {pass_location}")
                    # if x_last < pass_location:
                    if "atomic_step" in traj_q_dir:
                        stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
                        stays_wall_angle = np.append(stays_wall_angle, wall_angle)
                    else:
                        stays_slow_v_s = np.append(stays_slow_v_s, v_s_fac)
                        stays_slow_angle = np.append(stays_slow_angle, wall_angle)
                    # else:
                        # stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
                        # stays_wall_angle = np.append(stays_wall_angle, wall_angle)
                    if False:
                        logging.info("survives, passes; t > 10ns")
                elif positive_velocity:
                    stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
                    stays_start_angle = np.append(stays_start_angle, wall_angle)
                    if False:
                        logging.info("survives, bounces back")
                elif t_last not in end_times:
                    # print(t_last)
                    stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
                    stays_wall_angle = np.append(stays_wall_angle, wall_angle)
                    if False:
                        logging.info("survives, gets stuck")
                else:
                    logging.info(f"t_last, x_last, v_s_fac, v_s_angle: {t_last}, {x_last}, {v_s_fac}, {wall_angle}")

                
            # else:
            #     stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
            #     stays_start_angle = np.append(stays_start_angle, wall_angle)
                if False:
                    logging.info("survives, bounces back")
            # elif x_last < x_threashold_mid:
            #     if negative_velocity:
            #         # logging.info(f"x_last, pass_location: {x_last}, {pass_location}")
            #         if x_last < pass_location:
            #             stays_slow_v_s = np.append(stays_slow_v_s, v_s_fac)
            #             stays_slow_angle = np.append(stays_slow_angle, wall_angle)
            #         else:
            #             stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
            #             stays_wall_angle = np.append(stays_wall_angle, wall_angle)
            #         if False:
            #             logging.info("survives, passes; t > 10ns")
            #     elif positive_velocity:
            #         stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
            #         stays_start_angle = np.append(stays_start_angle, wall_angle)
            #         if False:
            #             logging.info("survives, bounces back")
            #     elif t_last not in end_times:
            #         # print(t_last)
            #         stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
            #         stays_wall_angle = np.append(stays_wall_angle, wall_angle)
            #         if False:
            #             logging.info("survives, gets stuck")
            # else:
            #     stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
            #     stays_start_angle = np.append(stays_start_angle, wall_angle)
            #     if False:
            #         logging.info("survives, bounces back")

    
    if not temp_dir_found:
        for i in range(100):
            pot_temp_dir = f"{phase_diag_temp_dir}/temp_phase_dir_{i}.npz"
            if not os.path.exists(pot_temp_dir):
                temp_dir = pot_temp_dir
                break
            if i == 99:
                logging.info(f"Too many temp files, writing not possible, waiting 5 sec then deleting oldest temp file")
                time.sleep(5)
                # get name for oldest file in pot_temp_dir and delete it
                oldest_file = min(glob.iglob(f"{phase_diag_temp_dir}/temp_phase_dir_*.npz"), key=os.path.getctime)
                os.remove(oldest_file)
                for i in range(100):
                    pot_temp_dir = f"{phase_diag_temp_dir}/temp_phase_dir_{i}.npz"
                    if not os.path.exists(pot_temp_dir):
                        temp_dir = pot_temp_dir
                        break
        logging.info(f"writing data to temp file: {temp_dir}")
        np.savez(
            temp_dir,
            stays_back_v_s=stays_back_v_s,
            stays_back_angle=stays_back_angle,
            stays_wall_v_s=stays_wall_v_s,
            stays_wall_angle=stays_wall_angle,
            stays_start_v_s=stays_start_v_s,
            stays_start_angle=stays_start_angle,
            stays_slow_v_s=stays_slow_v_s,
            stays_slow_angle=stays_slow_angle,
            destr_back_v_s=destr_back_v_s,
            destr_back_angle=destr_back_angle,
            destr_wall_v_s=destr_wall_v_s,
            destr_wall_angle=destr_wall_angle,
            destr_start_v_s=destr_start_v_s,
            destr_start_angle=destr_start_angle,
            fetch_dir=fetch_dir
        )
    
    end_reach_v_s = np.concatenate((stays_back_v_s, stays_slow_v_s, destr_back_v_s))
    end_reach_angle = np.concatenate((stays_back_angle, stays_slow_angle, destr_back_angle))

    repelled_back_v_s = np.concatenate((stays_start_v_s, destr_start_v_s))
    repelled_back_angle = np.concatenate((stays_start_angle, destr_start_angle))

    # ---------------------------------------------------------------Machine Learning for curves-------------------------------------------------------------------
    datapoints = {
        "end_reach_v_s": end_reach_v_s / a,
        "end_reach_angle": end_reach_angle,
        "repelled_back_v_s": repelled_back_v_s / a,
        "repelled_back_angle": repelled_back_angle,
        "stays_wall_v_s": stays_wall_v_s / a,
        "stays_wall_angle": stays_wall_angle,
        "destr_wall_v_s": destr_wall_v_s / a,
        "destr_wall_angle": destr_wall_angle,
    }
    isatomicstep = "atomic_step" in fetch_dir[0]
    name = "phasediagram"
    if isatomicstep: name += "_atomic_step"
    dest_boundary_file = name + "_with_boundaries.png"
    boundary_dir = os.path.join(dest_dir, dest_boundary_file)
    all_v_s = np.concatenate([datapoints[key] for key in datapoints if "v_s" in key])
    all_angles = np.concatenate([datapoints[key] for key in datapoints if "angle" in key])
    min_v_s_angle = np.min(all_angles)
    max_v_s_angle = np.max(all_angles)
    min_v_s_fac = np.min(all_v_s)
    max_v_s_fac = np.max(all_v_s)
    plotlims=np.array([[min_v_s_angle, max_v_s_angle], [min_v_s_fac, max_v_s_fac]])

    create_plot_with_boundaries(datapoints, isatomicstep, boundary_dir, plotlims)
    # ---------------------------------------------------------------Plotting datapoints with seperation between survives and destroyed-------------------------------------------------------------------

    plt.figure()
    # # =========================================normal plot separation=========================================
    # plt.plot(stays_back_v_s, stays_back_angle, "o", linewidth=4, markersize=4, label="survives, passes", color=(0, 0.9, 0))  # light green 
    # plt.plot(stays_slow_v_s, stays_slow_angle, "o", linewidth=4, markersize=4, label="survives, passes; t > 10ns", color=(0, 0.4, 0))   # dark green
    # plt.plot(stays_wall_v_s, stays_wall_angle, "o", linewidth=4, markersize=4, label="survives, gets stuck", color=(1.0, 0.5, 0.0)) # orange
    # plt.plot(stays_start_v_s, stays_start_angle, "ro", linewidth=4, markersize=4, label="survives, bounces back")   # red
    # plt.plot(destr_back_v_s, destr_back_angle, "^", linewidth=4, markersize=4, label="destroyed after passing", color=(0, 0.9, 0))  # light green
    # plt.plot(destr_wall_v_s, destr_wall_angle, "^", linewidth=4, markersize=4, label="destroyed on wall", color=(1.0, 0.5, 0.0))    # orange
    # plt.plot(destr_start_v_s, destr_start_angle, "r^", linewidth=4, markersize=4, label="destroyed at start")   # red

    #========================================= connecting different areas=========================================

    plt.plot(end_reach_v_s, end_reach_angle, "o", linewidth=4, markersize=4, label="survives, passes", color=(0, 0.9, 0))  # light green
    plt.plot(repelled_back_v_s, repelled_back_angle, "ro", linewidth=4, markersize=4, label="survives, bounces back")   # red
    plt.plot(stays_wall_v_s, stays_wall_angle, "o", linewidth=4, markersize=4, label="survives, gets stuck", color=(1.0, 0.5, 0.0)) # orange
    plt.plot(destr_wall_v_s, destr_wall_angle, "^", linewidth=4, markersize=4, label="destroyed on wall", color=(1.0, 0.5, 0.0))    # orange
    # ============================================================================================================

    # plt.legend(loc="upper left", fontsize="small")
    # plt.title("angled_current on straight wall")
    plt.xlabel(r"$|\vec{\bm{v}}_{\mathrm{s}}|$ [nm/ns]")
    plt.ylabel(r"$\varphi_{\mathrm{s}}$ [deg]")
    plt.tight_layout()
    # plt.savefig("1.5T_alex_phase_diagram.png", dpi=300)
    out_path = os.path.join(dest_dir, dest_file)
    plt.savefig(out_path, dpi=800, transparent=True)
    # plt.show()


def main():
    dest_dir = "OUTPUT"

    # =========================================new one 2 no step=========================================
    dest_file = "phasediagram_exact_edge_2_try.png"
    # fetch_dir = "OUTPUT/ROMMING_same_beta_atomistic_angled_vs_comparison_open_heun_1.5_-20.0_-9.0"
    fetch_dir = "OUTPUT/ROMMING_same_beta_2_high_angles_atomistic_angled_vs_comparison_open_heun_1.5_-20.5_-9.0"
    fetch_dir_2 = "OUTPUT/ROMMING_same_beta_2_low_angles_atomistic_angled_vs_comparison_open_heun_1.5_-20.5_-5.875"
    fetch_dir_3 = "OUTPUT/ROMMING_same_beta_high_v_s_atomistic_angled_vs_comparison_open_heun_1.5_-21.5_-3.0"
    fetch_dir = np.array([fetch_dir, fetch_dir_2, fetch_dir_3])

    create_phase_diagram_v_s_factor_vs_v_s_angle(fetch_dir=fetch_dir, dest_file=dest_file, dest_dir=dest_dir)


if __name__ == "__main__":
    main()