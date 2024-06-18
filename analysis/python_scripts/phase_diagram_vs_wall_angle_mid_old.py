import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import os

import glob

from sklearn import svm

import logging  # enabling display of logging.info messages


def fetch_traj_q_file(fetch_file):
    # fetch the traj_q file from the fetch folder
    traj_q_file = glob.glob(fetch_file, recursive=True)

    amount_of_traj_q_files = len(traj_q_file)

    # error catching
    if amount_of_traj_q_files != 1:
        logging.error("There should be exactly one traj_q.npy file.")
        if amount_of_traj_q_files == 0:
            logging.error("No traj_q.npy file was found.")
        elif amount_of_traj_q_files > 1:
            logging.error(f"{amount_of_traj_q_files} traj_q.npy files were found.")
        exit()

    return np.load(traj_q_file[0])


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# def skyrmion_profile(r, R, Delta):
#     """
#     The tanh-like function describing the radial profile of a skyrmion.
#     """
#     return np.tanh((r - R) / Delta)


# def fit_skyrmion_radius(mz, center):
#     """
#     Fit the mz component of the magnetization to determine the skyrmion radius using scipy's curve_fit.

#     Parameters:
#     mz (2D numpy array): The mz component of the magnetization.
#     center (tuple): The coordinates of the center of the skyrmion (y, x).

#     Returns:
#     float: The radius R of the skyrmion.
#     """
#     y, x = np.indices(mz.shape)
#     r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

#     # Bin m_z values by radial distance to get average m_z for each r
#     bins = np.linspace(0, np.max(r), 400)
#     bin_indices = np.digitize(r, bins)
#     binned_mz = [np.mean(mz[bin_indices == i]) for i in range(1, len(bins))]

#     # Initial guesses for R and Delta
#     p0 = [10, 1]

#     # Fit using curve_fit
#     popt, _ = curve_fit(skyrmion_profile, bins[:-1], binned_mz, p0=p0)

#     return popt[0]  # Return the fitted R value


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


# skyr_path = "needed_files/Neel_Skyr_2_big.npy"

# skyr = np.transpose(np.load(skyr_path), (1, 0, 2))

# # swap x and y axis
# skyr[:, :, 0], skyr[:, :, 1] = (
#     skyr[:, :, 1],
#     skyr[:, :, 0].copy(),
# )

# # skyr = np.load(skyr_path)

# min_idx_skyr = np.unravel_index(np.argmin(skyr[:, :, 2]), skyr.shape[:2])

# print(min_idx_skyr)

# R = fit_skyrmion_radius(skyr[:, :, 2], min_idx_skyr)

# print(R)

# # save skyr normally
# skyr[min_idx_skyr[0], min_idx_skyr[1], 2] = 1
# plt.imsave("skyrrrr.png", skyr[:, :, 2], cmap="seismic")


# # collect vert and horiz data
# vert = skyr[min_idx_skyr[0], :, 2]
# horiz = skyr[:, min_idx_skyr[1], 2]


# # random array shape (100, 400, 3)
# array = np.random.rand(4, 5, 3)
# min_idx = np.unravel_index(np.argmin(array[:, :, 2]), array.shape[:2])


# print(min_idx)
# print(array[min_idx])
# print(array)

# save the topological charge and location at the current time
# output.q_location_tracker[index_t, 0] = t
# output.q_location_tracker[index_t, 1] = q_sum

# tracker_array = spin.spins_evolved[:, :, 2] - relaxed_init_spins[:, :, 2]
# tracker_array[tracker_array > 0] = 0

# output.q_location_tracker[index_t, 2:] = spin.find_skyr_center(tracker_array)

# empty array
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


base_dir = "OUTPUT/ROMMING_same_beta_atomistic_angled_vs_comparison_open_heun_1.5_-20.0_-9.0"

traj_q_pattern = os.path.join(base_dir, "**", "traj_q.npy")

# -----------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------Testing value-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

# file_current = "OUTPUT/results_current/sample_1_1.5_deg_5.0_v_s_fac/traj_q.npy"
# file = "OUTPUT/results_3.5_9.5_deg_22_40_vs/sample_13/traj_q.npy"
# file_2 = "OUTPUT/results_3_5_deg_22_40_vs/sample_53/traj_q.npy"

# traj_q = np.load(file_current)
# filtered_traj_q = traj_q[~np.all(traj_q == 0, axis=1)]
# q_gradient = np.gradient(filtered_traj_q[:, 1] - filtered_traj_q[0, 1] + 1)

# # lowest index where gradient is below -0.5 --> where the skyrmion is destroyed
# idx_stop = np.where(q_gradient < -0.3)[0]
# # print(idx_stop)
# # if idx_stop.size > 0:
# #     # print(idx_stop)
# #     adj_traj_q = filtered_traj_q[: idx_stop[-1], :]
# # else:
# #     adj_traj_q = filtered_traj_q

# dirname = os.path.dirname(file_current)

# v_s_fac_and_wall_angle = np.load(os.path.join(dirname, "v_s_fac_and_wall_angle.npy"))
# v_s_fac = v_s_fac_and_wall_angle[0]
# wall_angle = v_s_fac_and_wall_angle[1]

# q_last = filtered_traj_q[-1, 1]
# t_last = filtered_traj_q[-1, 0]
# x_last = filtered_traj_q[-1, 2]
# y_last = filtered_traj_q[-1, 3]
# x_6_back = filtered_traj_q[-6, 2]


# ---------------------------------------------------------------Prints-------------------------------------------------------------------

# print(v_s_fac, wall_angle)
# print(q_last, x_last, y_last, x_6_back)
# # print(traj_q)
# np.savetxt(os.path.join(base_dir, "traj_q_sample_59_new_new_new.txt"), traj_q)
# # print(filtered_traj_q)


# -----------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------Loop-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

# for wall angled, v_s_x = const, v_s_y = 0
x_threashold_back = 450
x_threashold_mid = 180

# for v_s_angled not wall angled
x_threashold_back = 30
x_threashold_mid = 470

q_threashold = 0.2

traj_q_dirs = glob.iglob(traj_q_pattern, recursive=True)

for traj_q_idx, traj_q_dir in enumerate(traj_q_dirs):

    traj_q = fetch_traj_q_file(traj_q_dir)

    q = traj_q["topological_charge"]

    # delete the indices in traj_q where abs(topo charge) is not close to 0 (+- 0.1)
    sim_finished = np.where(np.logical_or(abs(q) < 0.9, abs(q) > 1.1))
    traj_q = np.delete(traj_q, sim_finished)
    q = np.delete(q, sim_finished)

    # # possibility for further condition
    # condition = True
    # valid_indices = np.where(np.logical_and(t != 0, condition))[0]
    # filtered_traj_q = traj_q[~np.all(traj_q == 0, axis=1)]

    # Extract data fields
    x = traj_q["x0"]
    y = traj_q["y0"]
    t = traj_q["time"]


    # q_gradient = np.gradient(filtered_traj_q[:, 1] - filtered_traj_q[0, 1] + 1)
    # lowest index where gradient is below -0.5 --> where the skyrmion is destroyed
    # idx_stop = np.where(q_gradient < -0.3)[0]
    # # print(idx_stop)
    # if idx_stop.size > 0:
    #     # print(idx_stop)
    #     adj_traj_q = filtered_traj_q[: idx_stop[-1] + 1, :]
    # else:
    #     adj_traj_q = filtered_traj_q

    dirname = os.path.dirname(traj_q_dir)

    v_s_fac_and_wall_angle = np.load(
        os.path.join(dirname, "v_s_fac_and_wall_angle.npy")
    )
    v_s_fac = v_s_fac_and_wall_angle[0]
    wall_angle = v_s_fac_and_wall_angle[1]

    q_init = q[0]
    q_sum_last = q[-1]
    q_last = q_sum_last - q_init
    t_last = t[-1]
    x_last = x[-1]
    y_last = y[-1]
    x_7_back = x[-7]
    y_7_back = y[-7]
    x_before_last = x[-2]
    x_8_back =x[-8]
    end_times = (10, 30, 50)

    q_threashold = 0.2

    # Skyrmion is moving until end
    if t_last not in end_times:
        x_last = x_7_back
        x_before_last = x_8_back
    # if t_last == 20:
    #     print(file)

    movement_threshold = 0.05

    negative_velocity = (
        x_last - x_8_back < -0.05
    )  # bool representing if the skyrmion is moving to the right

    # if v_s_fac < 5:
    logging.info(f"SAMPLE {traj_q_idx}")
    logging.info(f"v_s_fac: {v_s_fac}")
    logging.info(f"wall_angle: {wall_angle}")
    logging.info(f"t_last: {t_last}")
    logging.info(f"x_last: {x_last}")

    if abs(q_last) < q_threashold:
        if x_last < x_threashold_back:
            destr_back_v_s = np.append(destr_back_v_s, v_s_fac)
            destr_back_angle = np.append(destr_back_angle, wall_angle)
        elif x_last < x_threashold_mid:
            destr_wall_v_s = np.append(destr_wall_v_s, v_s_fac)
            destr_wall_angle = np.append(destr_wall_angle, wall_angle)
        else:
            destr_start_v_s = np.append(destr_start_v_s, v_s_fac)
            destr_start_angle = np.append(destr_start_angle, wall_angle)
    else:
        if x_last < x_threashold_back:
            stays_back_v_s = np.append(stays_back_v_s, v_s_fac)
            stays_back_angle = np.append(stays_back_angle, wall_angle)
        elif x_last < x_threashold_mid:
            if t_last not in end_times:
                # print(t_last)
                stays_wall_v_s = np.append(stays_wall_v_s, v_s_fac)
                stays_wall_angle = np.append(stays_wall_angle, wall_angle)
            else:
                if negative_velocity:
                    stays_slow_v_s = np.append(stays_slow_v_s, v_s_fac)
                    stays_slow_angle = np.append(stays_slow_angle, wall_angle)
                else:
                    stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
                    stays_start_angle = np.append(stays_start_angle, wall_angle)
        else:
            stays_start_v_s = np.append(stays_start_v_s, v_s_fac)
            stays_start_angle = np.append(stays_start_angle, wall_angle)

# ---------------------------------------------------------------Machine Learning for curves-------------------------------------------------------------------

# phase1_points = np.column_stack((stays_wall_v_s, stays_wall_angle))
# print(phase1_points.shape)
# phase2_points = np.column_stack(
#     (
#         np.concatenate((stays_back_v_s, stays_slow_v_s, destr_back_v_s), axis=0),
#         np.concatenate((stays_back_angle, stays_slow_angle, destr_back_angle), axis=0),
#     )
# )
# print(phase2_points.shape)

# xlim = (None, 27)
# ylim = (None, None)

# phase1_red_points = filter_points(phase1_points, xlim, ylim)
# phase2_red_points = filter_points(phase2_points, xlim, ylim)

# # Labels: 1 for phase1 and -1 for phase2
# labels = [1] * len(phase1_red_points) + [-1] * len(phase2_red_points)

# # Combine data
# data = np.vstack((phase1_red_points, phase2_red_points))

# # Train SVM
# clf = svm.SVC(kernel="poly", degree=2, C=1e6)
# clf.fit(data, labels)

# # Visualization
# plt.scatter(phase1_points[:, 0], phase1_points[:, 1], color="r", label="Phase 1")
# plt.scatter(phase2_points[:, 0], phase2_points[:, 1], color="b", label="Phase 2")

# # Plot decision boundary
# ax = plt.gca()
# xlim_ax = ax.get_xlim()
# ylim_ax = ax.get_ylim()
# xlim = (xlim[0] if xlim[0] is not None else xlim_ax[0], xlim[1] if xlim[1] is not None else xlim_ax[1])
# ylim = (ylim[0] if ylim[0] is not None else ylim_ax[0], ylim[1] if ylim[1] is not None else ylim_ax[1])

# # Create grid to show the result and evaluate the model
# gridsize = 500
# xx = np.linspace(xlim[0], xlim[1], gridsize)
# yy = np.linspace(ylim[0], ylim[1], gridsize)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)


# # 3. Extract points from the decision boundary
# threshold = 0.02
# boundary_points = xy[np.abs(Z.ravel()) < threshold]  # where `threshold` is a small value like 0.05

# print(boundary_points)

# # 4. Fit a 2D polynomial
# coeffs = np.polyfit(boundary_points[:, 0], boundary_points[:, 1], 2)
# print(coeffs)

# # 5. gather data points from the coeffs
# x_poly = np.linspace(xlim[0], xlim[1], 100)
# y_poly = coeffs[0] * x_poly**2 + coeffs[1] * x_poly + coeffs[2]

# # print(Z.shape)
# # print(Z)

# # Plot decision boundary and margins
# ax.contour(XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"])

# # plot the polynomial
# plt.plot(x_poly, y_poly, color="k", label="Decision boundary x^2 fit")

# plt.legend(fontsize='small')
# plt.savefig("wall_angle_v_s_alex_fitted.png", dpi=300)
# plt.show()

# ---------------------------------------------------------------Plotting the datapoints-------------------------------------------------------------------


# plt.figure()
# plt.plot(
#     stays_back_v_s,
#     stays_back_angle,
#     "o",
#     linewidth=4,
#     markersize=4,
#     label="survives, passes",
#     color=(0, 0.9, 0),  # rgb light green
# )
# plt.plot(
#     stays_slow_v_s,
#     stays_slow_angle,
#     "o",
#     linewidth=4,
#     markersize=4,
#     label="survives, passes; t > 10ns",
#     color=(0, 0.4, 0),  # rgb dark green
# )
# plt.plot(
#     stays_wall_v_s,
#     stays_wall_angle,
#     "o",
#     linewidth=4,
#     markersize=4,
#     label="survives, gets stuck",
#     color=(1.0, 0.5, 0.0),  # rgb orange
# )
# plt.plot(
#     stays_start_v_s,
#     stays_start_angle,
#     "ro",
#     linewidth=4,
#     markersize=4,
#     label="survives, bounces back",
# )
# plt.plot(
#     destr_back_v_s,
#     destr_back_angle,
#     "^",
#     linewidth=4,
#     markersize=4,
#     label="destroyed after passing",
#     color=(0, 0.9, 0),  # rgb light green
# )
# plt.plot(
#     destr_wall_v_s,
#     destr_wall_angle,
#     "^",
#     linewidth=4,
#     markersize=4,
#     label="destroyed on wall",
#     color=(1.0, 0.5, 0.0),  # rgb orange
# )
# plt.plot(
#     destr_start_v_s,
#     destr_start_angle,
#     "r^",
#     linewidth=4,
#     markersize=4,
#     label="destroyed at start",
# )

# plt.legend(loc="upper right", fontsize="small")
# plt.title("angled_current on straight wall")
# plt.xlabel("v_s_factor")
# plt.ylabel("v_s_angle [deg]")
# plt.tight_layout()
# plt.savefig("wall_v_s_angled_alex.png", dpi=300)
# plt.show()
# ---------------------------------------------------------------Plotting datapoints with seperation between survives and destroyed-------------------------------------------------------------------

plt.figure()
plt.plot(
    stays_back_v_s,
    stays_back_angle,
    "o",
    linewidth=4,
    markersize=4,
    label="survives, passes",
    color=(0, 0.9, 0),  # rgb light green
)
plt.plot(
    stays_slow_v_s,
    stays_slow_angle,
    "o",
    linewidth=4,
    markersize=4,
    label="survives, passes; t > 10ns",
    color=(0, 0.4, 0),  # rgb dark green
)
plt.plot(
    stays_wall_v_s,
    stays_wall_angle,
    "o",
    linewidth=4,
    markersize=4,
    label="survives, gets stuck",
    color=(1.0, 0.5, 0.0),  # rgb orange
)
plt.plot(
    stays_start_v_s,
    stays_start_angle,
    "ro",
    linewidth=4,
    markersize=4,
    label="survives, bounces back",
)
plt.plot(
    destr_back_v_s,
    destr_back_angle,
    "^",
    linewidth=4,
    markersize=4,
    label="destroyed after passing",
    color=(0, 0.9, 0),  # rgb light green
)
plt.plot(
    destr_wall_v_s,
    destr_wall_angle,
    "^",
    linewidth=4,
    markersize=4,
    label="destroyed on wall",
    color=(1.0, 0.5, 0.0),  # rgb orange
)
plt.plot(
    destr_start_v_s,
    destr_start_angle,
    "r^",
    linewidth=4,
    markersize=4,
    label="destroyed at start",
)

# ---------------------------------------------------------------Machine Learning for curves-------------------------------------------------------------------

# # combining all datapoints for stays and destr
# no_success_phase = np.column_stack(
#     (
#         np.concatenate((stays_wall_v_s,), axis=0),
#         np.concatenate((stays_wall_angle,), axis=0),
#     )
# )
# print(no_success_phase.shape)
# success_phase = np.column_stack(
#     (
#         np.concatenate((stays_back_v_s, stays_slow_v_s, destr_back_v_s), axis=0),
#         np.concatenate((stays_back_angle, stays_slow_angle, destr_back_angle), axis=0),
#     )
# )
# print(success_phase.shape)

# xlim = (None, 27)
# ylim = (None, None)

# phase1_red_points = filter_points(no_success_phase, xlim, ylim)
# phase2_red_points = filter_points(success_phase, xlim, ylim)

# # Labels: 1 for phase1 and -1 for phase2
# labels = [1] * len(phase1_red_points) + [-1] * len(phase2_red_points)

# # Combine data
# data = np.vstack((phase1_red_points, phase2_red_points))

# # Train SVM
# clf = svm.SVC(kernel="poly", degree=2, C=1e6)
# clf.fit(data, labels)

# # Visualization
# plt.scatter(no_success_phase[:, 0], no_success_phase[:, 1], color="r", label="Phase 1")
# plt.scatter(success_phase[:, 0], success_phase[:, 1], color="b", label="Phase 2")

# # Plot decision boundary
# ax = plt.gca()
# xlim_ax = ax.get_xlim()
# ylim_ax = ax.get_ylim()
# xlim = (
#     xlim[0] if xlim[0] is not None else xlim_ax[0],
#     xlim[1] if xlim[1] is not None else xlim_ax[1],
# )
# ylim = (
#     ylim[0] if ylim[0] is not None else ylim_ax[0],
#     ylim[1] if ylim[1] is not None else ylim_ax[1],
# )

# # Create grid to show the result and evaluate the model
# gridsize = 500
# xx = np.linspace(xlim[0], xlim[1], gridsize)
# yy = np.linspace(ylim[0], ylim[1], gridsize)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)


# # 3. Extract points from the decision boundary
# threshold = 0.02
# boundary_points = xy[
#     np.abs(Z.ravel()) < threshold
# ]  # where `threshold` is a small value like 0.05

# print(boundary_points)

# # 4. Fit a 2D polynomial
# coeffs = np.polyfit(boundary_points[:, 0], boundary_points[:, 1], 2)
# print(coeffs)

# # 5. gather data points from the coeffs
# x_poly = np.linspace(xlim[0], xlim[1], 100)
# y_poly = coeffs[0] * x_poly**2 + coeffs[1] * x_poly + coeffs[2]

# # print(Z.shape)
# # print(Z)

# # Plot decision boundary and margins
# ax.contour(XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"])

# # plot the polynomial
# plt.plot(x_poly, y_poly, color="k", label="Decision boundary x^2 fit")

# plt.legend(fontsize="small")
# plt.savefig("wall_angle_v_s_alex_fitted.png", dpi=300)
# plt.show()

# ---------------------------------------------------------------Plotting-------------------------------------------------------------------


plt.legend(loc="upper left", fontsize="small")
plt.title("angled_current on straight wall")
plt.xlabel("v_s_factor")
plt.ylabel("v_s_angle [deg]")
plt.tight_layout()
# plt.savefig("1.5T_alex_phase_diagram.png", dpi=300)
out_name = "phasediagram_straight.png"
out_dir = "OUTPUT"
out_path = os.path.join(out_dir, out_name)
plt.savefig(out_path, dpi=300)
plt.show()


# ---------------------------------------------------------------other prints-------------------------------------------------------------------


# print(destr_back_angle)
# print(destr_back_v_s)
# print(stays_back_angle)
# print(stays_back_v_s)
# print(destr_wall_angle)
# print(destr_wall_v_s)
# print(stays_wall_angle)
# print(stays_wall_v_s)
# print(destr_start_angle)
# print(destr_start_v_s)
# print(stays_start_angle)
# print(stays_start_v_s)

# print(traj_q)
# np.savetxt(os.path.join(base_dir, "filtered_traj_q.txt"), filtered_traj_q)
