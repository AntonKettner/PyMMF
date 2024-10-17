import unittest as ut
import shutil
import glob
import numpy as np
import os
import sys
import subprocess
import logging

# import the module to be tested
class TestSkyrmionSimulations(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.info("Setting up the test class")

        # create output directory if it does not exist, else remove all files in it
        if os.path.exists("testing/output"):
            shutil.rmtree("testing/output")
        os.makedirs("testing/output")

    def test_mask_availability(self):
        needed_masks = [
            "Mask_final_ReLU_high_beta_modular.png",
            "Mask_final_ReLU_simplification_bigger_11.png",
            "Mask_track_narrowing_stuck.png",
            "Mask_track_free.png",
            "Mask_track_beta_vs_alpha.png",
            "Mask_track_test.png",
            "Mask_track_100100atomic.png",
        ]

        for mask in needed_masks:
            self.assertTrue(os.path.exists(f"needed_files/{mask}"), f"Mask {mask} not found")

        logging.info("All masks are available")

    def test_ReLU(self):
        sim_type = "ReLU"
        logging.warning(f"RUNNING TEST: {sim_type}")
        result = subprocess.run(
            ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
        )

        # save the console output to a file
        output_file = f"testing/output/output_{sim_type}.txt"
        with open(output_file, "w") as f:
            f.write(result.stderr)
            logging.info(f"Console Output written to {output_file}")

        # verify that the process was successful
        self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")

        # =========================================verify that the pop times for the ReLU are all equal=========================================

        # load pop_times from the OUTPUT folder
        file_name      = "pop_times.npy"
        dir_rel        = f"Thesis_Fig_15_{sim_type}"
        file_pattern   = os.path.join(parent_directory, "OUTPUT", dir_rel, "**", file_name)
        pop_times_file = glob.glob(file_pattern, recursive=True)
        pop_times      = np.load(pop_times_file[0])

        if pop_times.size == 0:
            self.fail(f"pop_times is empty for {sim_type}")
        elif np.all(pop_times[:-1] == pop_times[0]):
            logging.warning(f"pop_times[:-1] are equal for {sim_type}")
            logging.info(f"pop_times: {pop_times} for {sim_type}")

    def test_skyr_creation(self):
        sim_type = "skyrmion_creation"
        logging.warning(f"RUNNING TEST: {sim_type}")
        result = subprocess.run(
            ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
        )

        # save the console output to a file
        output_file = f"testing/output/output_{sim_type}.txt"
        with open(output_file, "w") as f:
            f.write(result.stderr)
            logging.info(f"Console Output written to {output_file}")

        # verify that the process was successful
        self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")

        # =========================================verify that q_r_time plot is correct=========================================

        file_name       = "traj_q.npy"
        dir_rel         = f"Thesis_Fig_8_{sim_type}"
        file_pattern    = os.path.join(parent_directory, "OUTPUT", dir_rel, "**", file_name)
        traj_q_file     = glob.glob(file_pattern, recursive=True)
        traj_q          = np.load(traj_q_file[0])
        q               = traj_q["topological_charge"]
        t               = traj_q["time"]
        indices_to_keep = np.where(t != 0)
        q               = q[indices_to_keep]
        t               = t[indices_to_keep]

        if not 0.9 < abs(q[-1]) < 1.1:
            self.fail(f"Topological charge at the end of {sim_type}: {q[-1]}")

    def test_wall_ret_test(self):
        sim_type = "wall_ret_test"
        logging.warning(f"RUNNING TEST: {sim_type}")
        result = subprocess.run(
            ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
        )

        # save the console output to a file
        output_file = f"testing/output/output_{sim_type}.txt"
        with open(output_file, "w") as f:
            f.write(result.stderr)
            logging.info(f"Console Output written to {output_file}")

        # verify that the process was successful
        self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")

        # =========================================verify that the distance was found correctly=========================================

        file_name       = "traj_q.npy"
        dir_rel         = "wall_ret_test_middle_wall_ret_test"
        file_pattern    = os.path.join(parent_directory, "OUTPUT", dir_rel, "**", file_name)
        traj_q_file     = glob.glob(file_pattern, recursive=True)
        traj_q          = np.load(traj_q_file[0])
        vsx             = traj_q["v_s_x"]
        indices_to_keep = np.where(vsx != 0)
        vsx             = vsx[indices_to_keep]
        vsy             = traj_q["v_s_y"]
        vsy             = vsy[indices_to_keep]
        vs_value        = np.sqrt(vsx**2 + vsy**2)
        optimum         = 59.54

        if not optimum - 5 < vs_value[-1] < optimum + 5:
            self.fail(f"vs_strength_value at 2r away from wall {sim_type}: {vs_value[-1]}")

    def test_x_current(self):
        sim_type = "x_current"
        logging.warning(f"RUNNING TEST: {sim_type}")
        result = subprocess.run(
            ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
        )

        # save the console output to a file
        output_file = f"testing/output/output_{sim_type}.txt"
        with open(output_file, "w") as f:
            f.write(result.stderr)
            logging.info(f"Console Output written to {output_file}")

        # verify that the process was successful
        self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")

        # =========================================verify that end_x > start_x=========================================

        file_name      = "traj_q.npy"
        dir_rel        = "x_current_x_current"
        file_pattern   = os.path.join(parent_directory, "OUTPUT", dir_rel, "**", file_name)
        traj_q_file    = glob.glob(file_pattern, recursive=True)
        traj_q         = np.load(traj_q_file[0])
        x              = traj_q["x0"]
        y              = traj_q["y0"]
        movement_angle = np.degrees(np.arctan((y[-1] - y[0]) / (x[-1] - x[0])))

        if not -3 < movement_angle < 3:
            self.fail(f"skyrmion moving into the wrong direction: angle={movement_angle} for {sim_type}")

if __name__ == "__main__":
    # logging config
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Navigate one level up from the current script's directory
    parent_directory = os.path.abspath(os.path.join(__file__, "..", ".."))
    logging.info(f"Parent directory: {parent_directory}")

    # Add the parent directory to sys.path and change the cwd to the parent directory
    sys.path.insert(0, parent_directory)
    os.chdir(parent_directory)

    # Run the tests
    ut.main()
