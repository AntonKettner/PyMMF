# reads in trajq file and plots the trajectory ([2] vs [3]) and the fits with a linear function

import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import glob
import matplotlib
import matplotlib.pyplot as plt


def setup_plt():

    matplotlib.use("Agg")
    # add the path manually if necessary
    font_path = "//afs/physnet.uni-hamburg.de/users/AU/akettner/.conda/envs/2_pycuda/fonts/cmunrm.ttf"
    matplotlib.font_manager.fontManager.addfont(font_path)

    plt.rcParams["font.family"] = "CMU Serif"
    plt.rcParams["font.serif"] = "CMU Serif Roman"
    plt.rcParams["mathtext.fontset"] = "cm"


def create_input_output_plot(fetchpath, destpath, dest_file):

    setup_plt()

    file = "traj_q.npy"

    pattern_file = os.path.join(fetchpath, "**", file)

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

    logging.info(f"traj_q data: {traj_q.dtype.names}")

    t = traj_q["time"]
    q = traj_q["topological_charge"]
    l = traj_q["left_count"]
    r = traj_q["right_count"]

    # delete all values where t > 200 or q == 0
    # indices_to_keep = np.where(np.logical_and(t < 180, q != 0))
    indices_to_keep = np.where(q != 0)
    t = t[indices_to_keep]
    q = q[indices_to_keep]
    l = l[indices_to_keep]
    r = r[indices_to_keep]

    # löschen von file in dest_path, falls es existiert
    if os.path.exists(f"{destpath}/{dest_file}"):
        os.remove(f"{destpath}/{dest_file}")

    # ansonsten erstellung von dest_path falls es nicht existiert
    if not os.path.exists(f"{destpath}"):
        os.makedirs(f"{destpath}")

    # ---------------------------------------------------------------Input Output plot-------------------------------------------------------------------

    # clear all values of q and r, where r is the same value as the previous one
    indices_to_remove_for_output = []
    for i in range(len(r) - 1):
        if r[i] == r[i + 1] and not r[i] == 0:
            indices_to_remove_for_output.append(i)
    r = np.delete(r, indices_to_remove_for_output)
    t_for_output = np.delete(t, indices_to_remove_for_output)
    r = r[:-1]
    t_for_output = t_for_output[:-1]

    indices_to_remove_for_input = []
    for i in range(len(q) - 1):
        if abs(q[i + 1]) - 0.1 < abs(q[i]) < abs(q[i + 1]) + 0.1:
            indices_to_remove_for_input.append(i)
    q = np.delete(q, indices_to_remove_for_input)
    t_for_input = np.delete(t, indices_to_remove_for_input)
    q = q[:-1]
    t_for_input = t_for_input[:-1]

    # =========================================for thesis input output plot=========================================
    plt.figure()
    plt.plot(t_for_output, r, "-o", label="Output", color="red")
    plt.plot(t_for_input, abs(q), "-o", label="Input", color="black")

    legend = plt.legend(loc="upper left", fontsize=15)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("black")  # Set the rim color to black
    plt.xlabel(r"Time $t$ [ns]", fontsize=18)
    plt.ylabel(r"No. of Skyrmions $N_{\mathrm{Sk}}$", fontsize=18)

    # Increase the size of the numbers on the axes
    plt.tick_params(axis="both", which="major", labelsize=14)

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{destpath}/{dest_file}", dpi=800, transparent=True)
    plt.close()

    # =========================================times between cavity pop=========================================
    plt.figure()
    std_delta_t = t_for_input[2] - t_for_input[1]
    times_between_cavity_pop = t_for_output[1:] - t_for_output[:-1]
    r = r[1:]
    times_between_cavity_pop = times_between_cavity_pop[r != 0]
    r = r[r != 0]
    plt.plot(r, times_between_cavity_pop, "o", color="black")
    logging.info(f"times_between_cavity_pop: {times_between_cavity_pop}")
    logging.info(f"r: {r}")
    plt.ylim(std_delta_t - 3, std_delta_t + 3)
    plt.xlabel(r"Output $N_{\mathrm{Sk}}$", fontsize=15)
    plt.ylabel(r"Time after last Output $\Delta t$ [ns]", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{destpath}/Out_Dt_with_lims_{dest_file}", dpi=800, transparent=True)
    plt.close()

    # ========================================Times between cavity pop no limits==========================================
    plt.figure()
    plt.plot(r, times_between_cavity_pop, "o", color="black")
    plt.xlabel(r"Output $N_{\mathrm{Sk}}$", fontsize=15)
    plt.ylabel(r"Time after last Output $\Delta t$ [ns]", fontsize=15)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{destpath}/Out_Dt_no_lims_{dest_file}", dpi=800, transparent=True)
    plt.close()

    return times_between_cavity_pop


def main():

    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/results_{mode}_test_multitrack_{boundary}_boundary_rk4_{B_ext}_B_ext_{vs}_vs"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_FINAL_SIM_big_slowatomistic_ReLU_2.5_r_open_heun_1.5_2.5_0/sample_1_0_deg_2.5_v_s_fac"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_FINAL_ReLU_for_THESIS_test_both_betas_atomistic_ReLU_open_heun_1.5_1_0/sample_1_0_deg_1_v_s_fac"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_alt_B_ReLU_for_THESIS_Test_3_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0/sample_2_0_deg_1.0_v_s_fac"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_alt_B_ReLU_for_THESIS_Test_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0/sample_10_0_deg_1.0_v_s_fac"

    # dest_folder = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/input_output"

    # # create_input_output_plot(fetch_folder_name, dest_folder, dest_file)

    # # for multiple folders in folder:
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_alt_B_ReLU_for_THESIS_Test_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_alt_B_ReLU_for_THESIS_Test_3_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"
    # # for original low beta
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_alt_B_Mask_final_ReLU_simplification_bigger_11_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_big_beta_bias_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"
    # # for low limit vs big beta
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_big_beta_bias_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.9_0"
    # # for mid vs big beta
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_big_beta_bias_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_0.95_0"
    # dest_folder = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/input_output"
    # # for upper limit vs big beta
    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_big_beta_bias_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"

    # fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/ongoing_work/OUTPUT/ROMMING_big_beta_bias_test_FINAL_Mask_final_ReLU_high_beta_modular_atomistic_ReLU_changed_capacity_open_heun_1.5_1.0_0"
    fetch_folder_name = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/PyMMF/OUTPUT/Thesis_Fig_15_ReLU"
    dest_folder = f"/afs/physnet.uni-hamburg.de/users/AU/akettner/Projekt_PyCUDA/PyMMF/OUTPUT/input_output"

    # fetch dirs not files in fetch_folder_name
    for entry in os.scandir(fetch_folder_name):
        if entry.is_dir():
            fetch_folder = os.path.join(fetch_folder_name, entry.name)
            logging.info(f"fetch_folder: {fetch_folder}")
            dest_file = f"input_output_{entry.name}.png"
            try:
                create_input_output_plot(fetch_folder, dest_folder, dest_file)
            except:
                logging.error(f"Error for {entry.name}")


if __name__ == "__main__":
    main()
