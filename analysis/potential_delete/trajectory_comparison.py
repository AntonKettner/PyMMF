import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# base_dir = "traj_comp_methods"

# pattern = os.path.join(base_dir, "**", "traj_q.npy")
# plt.figure()

# plt.title("Wall retention: Comparison of Methods")

# for file in glob.glob(pattern, recursive=True):
#     traj_q = np.load(file)
#     filtered_traj_q = traj_q[~np.all(traj_q == 0, axis=1)]

#     # isolate the method name located between "current_wall_retention_" and "_low_res/sample_" in file
#     method = file.split("current_wall_retention_")[1].split("/sample_")[0]

#     # print(method)

#     # plt.plot(traj_q[:, 0], traj_q[:, 2], label=f"Method = {method}, x", linewidth=2.5)  # x vs t
#     # plt.plot(traj_q[:, 0], traj_q[:, 3], label=f"Method = {method}, y", linewidth=2.5)  # y vs t
#     plt.plot(
#         filtered_traj_q[:, 2],
#         filtered_traj_q[:, 3],
#         label=f"Method = {method}",
#         linewidth=2.5,
#     )  # y vs x

# plt.legend(loc="lower right", fontsize="small")
# plt.savefig("wall_retention_comparison_methods_y_x.png", dpi=300)
# plt.show()


folder = "results_current_wall_retention_2/sample_1_0_deg_0_v_s_fac"
file = "/traj_q.npy"

folder_2 = "results_current_wall_retention_anti/sample_1_0_deg_0_v_s_fac"


# load file
traj_q = np.load(folder + file)
traj_q_2 = np.load(folder_2 + file)

print(traj_q.shape)
# array = np.ones(100).reshape((10, 10))
# plt.imshow(
#             array,
#             cmap="RdBu",
#             vmin=-1,
#             vmax=1,
#             )

# plot [:0] against [:2]
plt.figure()

plt.title("Wall retention: Comparison of Helicities")
plt.plot(traj_q[:, 0], traj_q[:, 2] - traj_q[0, 2], label="Helicity = 0, x")
plt.plot(traj_q[:, 0], traj_q[:, 3] - traj_q[0, 3], label="Helicity = 0, y")
plt.plot(traj_q_2[:, 0], traj_q_2[:, 2] - traj_q_2[0, 2], label="Helicity = pi, x")
plt.plot(traj_q_2[:, 0], traj_q_2[:, 3] - traj_q_2[0, 3], label="Helicity = pi, y")
plt.xlabel("t [ns]")
plt.ylabel("x or y [0.3 nm]")
plt.legend(loc="lower right", fontsize="small")
plt.savefig("comparison_helicities_normalized.png", dpi=300)
plt.show()


# v_s_angles = np.linspace(7.4, 8, 7, endpoint=True)
# v_s_factors = np.linspace(1, 4, 8, endpoint=True)

# print(v_s_angles)
# print(v_s_factors)
