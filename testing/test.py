import unittest as ut
import os
import sys
import subprocess
import logging
import numpy as np

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Navigate one level up from the current script's directory
parent_directory = os.path.abspath(os.path.join(__file__, "..", ".."))
logging.info(f"Parent directory: {parent_directory}")

# original cwd:
orig_cwd = os.getcwd()

# Add the parent directory to sys.path and change the cwd to the parent directory
sys.path.insert(0, parent_directory)
os.chdir(parent_directory)

# from skyrmion_simulation.python_scripts.skyrmion_simulation import main as ss


class TestSkyrmionSimulations(ut.TestCase):

    def setUp(self):
        self.output_file = "test_output.txt"

        # delete pycache in the parent dir and every subdirectory
        for root, dirs, files in os.walk(parent_directory):
            for dir in dirs:
                if dir == "__pycache__":
                    os.system(f"rm -r {os.path.join(root, dir)}")

        # execute setup_env.sh in testing directory
        os.system(f"bash {os.path.join(parent_directory, 'testing', 'setup_env_physnet.sh')}")

        # self.sim_types = [
        #     "skyrmion_creation", "wall_retention", "wall_retention_new", "wall_ret_test",
        #     "wall_ret_test_new", "angled_vs_comparison", "angled_wall_comparison", "x_current",
        #     "creation_gate", "antiferromagnet_simulation", "pinning_tests", "first_results_replica",
        #     "wall_retention_reverse_beta", "ReLU", "ReLU_larger_beta", "ReLU_changed_capacity"
        # ]

        self.sim_types = [
            "x_current",
            "wall_retention",
            "wall_ret_test_close",
            "wall_ret_test_far",
            "skyr_creation",  # q_r_vs_time plot wrong
            "x_current_SkH_test",
            "angled_vs_on_edge",  # depending on mask_dir either with or without atomic step
            "pinning_tests",
            "ReLU",  # fix plot input output
            "ReLU_larger_beta",  # fix plot input output
        ]

    def test_simulation(self):
        for sim_type in self.sim_types:
            with self.subTest(sim_type=sim_type):
                logging.warning(f"RUNNING TEST: {sim_type}")
                result = subprocess.run(
                    ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
                )

                # save the consolt output to a file
                os.chdir(orig_cwd)
                with open(self.output_file, "w") as f:
                    f.write(result.stderr)
                    logging.info(f"Console Output written to {self.output_file}")

                # verify that the process was successful
                self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")


if __name__ == "__main__":
    ut.main()


# model_type              = "atomistic"
# calculation_method      = "heun"
# boundary                = "open"
# apply_bottom_angle      = False
# bottom_angles           = np.array([0])
# v_s_factors             = np.array([25])
# samples                 = bottom_angles.shape[0] * v_s_factors.shape[0]
# pivot_point             = (250, 100)
# final_skyr_No           = 1
# t_max                   = 1
# t_relax_skyr            = 0
# t_relax_no_skyr         = 0.3
# t_circ_buffer           = 0.01
# No_sim_img              = 20
# cc_steps                = 600000
# len_circ_buffer         = min(max(int(t_circ_buffer * No_sim_img / t_max), 5), 50)
# time_per_img            = t_max / No_sim_img
# t_last_skyr_frac        = 1
# save_pics               = True
# save_npys               = False
# save_npy_end            = True
# track_radius            = True
# check_variance          = True
# check_skyrmion_presence = True
# critical_variance       = 1e-6
# t_pics                  = np.linspace(0, t_max, No_sim_img + 1, endpoint=True)[1:]
# steps_per_avg           = 1
# learning_rate           = np.array([1, 1])
# smallest_error_yet      = 1000
# cons_reach_threashold   = 10


# # Berechnung der Anzahl an Bildern, nach denen je ein Skyrmion gesetzt wird
# if not final_skyr_No == 0:
#     every__pic_set_skyr = np.floor(No_sim_img * t_last_skyr_frac / final_skyr_No)
# else:
#     every__pic_set_skyr = No_sim_img + 1
