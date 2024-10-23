import unittest as ut
import os
import sys
import subprocess
import logging

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


# import the module to be tested
class TestSkyrmionSimulations(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.info("Setting up the test class")

        # create output directory if it does not exist, else remove all files in it
        if not os.path.exists("output"):
            os.makedirs("output")
        else:
            for file in os.listdir("output"):
                os.remove(os.path.join("output", file))

        cls.sim_types = [
            # "wall_repulsion",
            # # "x_current",
            # "wall_rep_test_close",
            # "wall_rep_test_far",
            # "x_current_SkH_test",
            "skyrmion_creation",  # q_r_vs_time plot wrong
            "pinning_tests",
            "ReLU",  # fix plot input output
            "ReLU_larger_beta",  # fix plot input output
            # "angled_vs_on_edge",  # depending on mask_dir either with or without atomic step
        ]

        # execute setup_test_env_physnet.sh in testing directory
        os.system(f"bash testing/setup_test_env_physnet.sh")

    def test_simulation(self):
        for sim_type in self.sim_types:
            self.sim_type = sim_type
            logging.info(f"Running the test case for {sim_type}")
            with self.subTest(sim_type=sim_type):

                logging.warning(f"RUNNING TEST: {sim_type}")
                result = subprocess.run(
                    ["python", "skyrmion_simulation/python_scripts/skyrmion_simulation.py", "--sim_type", sim_type], capture_output=True, text=True
                )

                # save the console output to a file
                output_file = f"output/output_{sim_type}.txt"
                # os.chdir(self.orig_cwd)
                with open("testing/" + output_file, "w") as f:
                    f.write(result.stderr)
                    logging.info(f"Console Output written to {output_file}")

                # verify that the process was successful
                self.assertEqual(result.returncode, 0, f"Process failed with return code {result.returncode} for {sim_type}")

    @classmethod
    def tearDownClass(cls):
        logging.info("Tearing down the test class")


if __name__ == "__main__":
    ut.main()
