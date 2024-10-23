# native imports
import logging  # enabling display of logging.info messages
import math

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden
from numpy.linalg import norm as value

# local imports
from constants import cst
from simulation import sim
from gpu import GPU
import pycuda.driver as cuda
from scipy.stats import uniform_direction

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class spin:
    """
    class containing the major operations on the spinfield for a skyrmion simulation.
    """

    # ---------------------------------------------------------------Methods-------------------------------------------------------------------

    @classmethod
    def norm(cls, array):
        """
        Normalizes the given array.
        """
        array[sim.mask] /= value(array[sim.mask], axis=-1, keepdims=True)
        return array

    @classmethod
    def masking(cls, array):
        """
        Masks the given array with sim.mask.
        """
        return array * sim.mask[..., np.newaxis]

    @classmethod
    def racetrack_bottom_angle(cls, center, angle):
        """
        relevant for "angled_wall_comparison"
        Modifies the input mask to set values below a specified line to False.

        Parameters:
        - mask: 2D numpy array
        - center: Tuple specifying a point on the line
        - angle: Angle of the line in degrees. 0 is horizontal, positive values are counter-clockwise.

        Returns:
        - Modified mask
        """
        # sets the mask to the original mask
        sim.load_mask(sim.orig_mask_dir)

        if sim.apply_bottom_angle:
            # ---------------------------------------------------------------Set everything below line in area to false---------------------------------------------
            # leaving this space on both sides
            s = 130

            # width of the mask
            width = sim.mask.shape[0]

            # slope of the line
            m = np.tan(np.radians(angle))

            # For each column x, compute the y value of the line and set values below it to False.
            for x in range(width - 2 * s):
                y_line = int(m * (x + s - center[0]) + (sim.y_size - center[1]))
                sim.mask[x + s, :y_line] = False

            # ---------------------------------------------------------------Set everything below this y to false---------------------------------------------
            y = int(m * (s - center[0]) + (sim.y_size - center[1]))  # y value at x = 0
            for x in range(width):
                sim.mask[x, :y] = False

        return sim.mask

    @classmethod
    def initialize_spinfield(cls):
        """
        Sets the initial spinfield for a given simulation as well as the spins_id as a class attribute.

        Parameters:
        sim_no (int): The index of the current simulation.

        Returns:
        spinfield (numpy.ndarray): The initial spinfield for the simulation.
        calqtopo (float): The topological charge density of the spinfield.
        """
        logging.info(f"setting init_spinfield")

        # define init state from random numbers between -0.1 and 0.1 in x and y direction and 1 in z direction
        # x, y = -0.1 to 0.1    # z = 1
        x_and_y = np.random.rand(sim.x_size, sim.y_size, 2) * 0.2 - 0.1
        z = np.ones((sim.x_size, sim.y_size, 1))
        spinfield = np.dstack((x_and_y, z))

        # transform into antiferromagnet
        if sim.sim_type == "antiferromagnet_simulation":
            spinfield[::2, ::2, 2] = -1
            spinfield[1::2, 1::2, 2] = -1

        # # INITSTATE SOME RANDOM NUMBERS AT X,Y
        # x = 200
        # y = 200
        # rnd_size = 10
        # var = int(rnd_size / 2)
        # spinfield[x - var : x + var, y - var : y + var] = math.rnd_vecs(
        #     (rnd_size, rnd_size)
        # )

        # # INITSTATE ALL RANDOM NUMBERS IN X,Y
        # spinfield = math.rnd_vecs((spin.field_size_x, spin.field_size_y))

        # mask and norm the spins
        spinfield = spin.masking(spin.norm(spinfield))

        # set boundary
        if sim.boundary == "ferro":
            logging.info("boundary: ferromagnetic")
            spinfield = spin.set_ferromagnetic_boundary(spinfield)

        elif sim.boundary == "open":
            logging.info("boundary: open\n")

        spinfield = spinfield.astype(np.float32)

        # allocate memory on the GPU for everything and the spinfield
        GPU.transfer_to_GPU(spinfield)

        return spinfield

    @staticmethod
    def sigmoid(x, exp_Factor=-20):
        """
        Calculates the sigmoid function for a given input:
        interpolating for insertion of new skyrmion

        Parameters:
        x (float): The input value.
        exp_Factor (float): The exponential factor. Default is -20.

        Returns:
        float: The output of the sigmoid function for the given input.
        """
        return 1 / (1 + np.exp(exp_Factor * (min(x, 1) - 1)))

    @staticmethod
    def rnd_vecs(shape):
        """
        Generates random vectors on the surface of a unit sphere for each point in the given shape.

        Parameters:
        shape (tuple): Shape of the array (e.g., (100, 200)).

        Returns:
        numpy array: A shape[0] x shape[1] x 3 array of random vectors.
        """
        # Number of points

        vectors = uniform_direction(3).rvs(shape[0] * shape[1], random_state=np.random.default_rng())
        vectors = vectors.reshape(shape[0], shape[1], 3)
        return vectors

    @staticmethod
    def set_ferromagnetic_boundary(spinfield):
        """
        Sets all the spins right at the edge of the mask that are now 0 in all directions to 1 in z direction.

        Parameters:
        spinfield (numpy.ndarray): A 3D numpy array representing the spin field.
            (first two dimensions are the x and y coordinates, third dimension is the x, y, and z component of the spin)

        Returns:
        numpy.ndarray: A 3D numpy array representing the spin field with the ferromagnetic boundary set.
        """

        NN_pos_odd_row = cst.NN_pos_odd_row[:, :-1]
        NN_pos_even_row = cst.NN_pos_even_row[:, :-1]

        for i in range(sim.x_size):
            for j in range(sim.y_size):
                if spinfield[i, j, :].all() == 0:
                    # Determine the nearest neighbor positions based on whether the row is even or odd
                    NN_pos = NN_pos_even_row if j % 2 == 0 else NN_pos_odd_row

                    # Iterate over the nearest neighbor positions
                    for dx, dy in NN_pos:
                        nx, ny = i + dx, j + dy

                        # Check if the nearest neighbor is within the spinfield and is not zero
                        if 0 <= nx < sim.x_size and 0 <= ny < sim.y_size and not spinfield[nx, ny, :].all() == 0:
                            spinfield[i, j, :] = np.array([0, 0, 1])
                            break  # Exit the loop as soon as a non-zero neighbor is found
        return spinfield

    @staticmethod
    def calculate_learning_rate(t, step_size=10, base_lr=0.1, max_lr=1.1):
        """
        Adjust learning rate in a cyclical manner based on step size.
        :param old_error: Previous error value.
        :param new_error: Current error value.
        :param past_learning_rate: Previous learning rate.
        :param step_size: Number of steps in half a cycle.
        :param base_lr: Minimum learning rate.
        :param max_lr: Maximum learning rate.
        :return: Adjusted learning rate.
        """
        # Calculate the cycle
        cycle = np.floor(1 + t / (2 * step_size))
        # Calculatechange in time within a cycle
        x = np.abs(t / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))

        return np.array([lr, lr])

    @staticmethod
    def set_skyr(spins, skyrmion, x, y):
        """
        Places a skyrmion at the specified position.

        Args:
        spins (numpy.ndarray): The spin configuration.
        x, y (int): The x and y coordinates of the center of the skyrmion.

        Returns:
        numpy.ndarray: The updated spin configuration with the skyrmion placed at the specified position.
        """
        # set x, y to be ints
        x = int(x)
        y = int(y)
        logging.info("setting skyrmion")
        skyr_size_x = skyrmion.shape[0]
        skyr_size_y = skyrmion.shape[1]
        # skyr_radius_set = math.ceil(2 * sim.r_skyr / (cst.a * 1e9))
        logging.info(f"r_skyr: {sim.r_skyr}")
        skyr_radius_set = sim.r_skyr * 10

        # skyr array boundary on each side
        boundary = (skyr_size_x + skyr_size_y) / 2 / 20

        skyr_center_x = int(skyr_size_x / 2)
        skyr_center_y = int(skyr_size_y / 2)

        for i in range(skyr_size_x):
            # Schleife ueber alle Positionen von dem einzelnen Skyrmion
            for j in range(skyr_size_y):
                if boundary < i < skyr_size_x - boundary and boundary < j < skyr_size_y - boundary:
                    # Index des Spins in spins, der dem Spin in singleSkyr entspricht
                    idx = x - int(skyr_center_x) + i
                    idy = y - int(skyr_center_y) + j

                    # normierter Abstand von i und j von der Skyrmionmitte --> bei 1 am Rand des Skyrmions
                    # dist = 4 * sim.r_skyr
                    if j % 2 == 0 and sim.sim_type == "skyrmion_creation":
                        x_pos = i + 1 / 2
                    else:
                        x_pos = i
                    if sim.model_type == "atomistic":
                        dist = value([(x_pos - skyr_center_x), (j - skyr_center_y) * np.sqrt(3) / 2]) / (skyr_radius_set)
                    elif sim.model_type == "continuum":
                        dist = value([(i - skyr_center_x), (j - skyr_center_y)]) / (skyr_radius_set)
                    else:
                        dist = "undefined"
                        raise ValueError("Invalid model type. Please choose either 'atomistic' or 'continuum'.")

                    # Abfrage, ob der Spin in spins liegt
                    if 0 <= idx < sim.x_size and 0 <= idy < sim.y_size:
                        # Es wird zwischen der vorherigen Spinkonfigeruration und dem Skyrmion mittels des Abstandes und einer Sigmoid-Funktion interpoliert
                        sig_skyr = (-2 * spin.sigmoid(dist)) + 1
                        sig_spinfield = 1 - sig_skyr
                        spins[idx, idy] = skyrmion[i, j] * sig_skyr + spins[idx, idy] * sig_spinfield
                        # # Normalisierung der Spins
                        # spins[idx, idy] = spins[idx, idy] / np.sqrt(np.dot(spins[idx, idy], spins[idx, idy]))

        # norm and mask the spins
        spins = spin.masking(spin.norm(spins))

        # set boundary ferro if necessary
        if sim.boundary == "ferro":
            spins = spin.set_ferromagnetic_boundary(spins)

        return spins

    # Spinfield relaxieren mit festgehaltenem Skyrmion
    @staticmethod
    def relax_but_hold_skyr(
        spins,
        numerical_steps,
        skyrs_set=0,
    ):
        """
        Relax the spin field while holding the skyrmion in place.

        Args:
            spins (ndarray): The spin field.
            numerical_steps (function): The function to perform a simulation step.
            gpu_spins (DeviceAllocation): The device memory for the spin field.
            No (int): The sample number.
            bottom_angle (float): The bottom angle.
            v_s_fac (float): The v_s factor.
            skyrs_set (int, optional): The skyrs set. Defaults to 0.

        Returns:
            ndarray: The relaxed spin field.
        """

        if skyrs_set > 0:
            return spins
        else:
            steps_relax = int(sim.t_relax_no_skyr / sim.dt)

            # Print that the relaxation is starting
            logging.info(f"relaxing spinfield for {sim.t_relax_no_skyr} ns with {steps_relax} steps, no skyr\n")

            # transfer the temp vs 0 bool to GPU
            v_s_temp = False
            cuda.memcpy_htod(GPU.v_s_active_id, np.int32(v_s_temp))

        for _ in range(steps_relax):
            GPU.perform_numerical_steps(
                numerical_steps,
            )

        logging.info("relaxation done\n")

        # reset back v_s to original value
        if skyrs_set == 0:
            cuda.memcpy_htod(GPU.v_s_active_id, np.int32(sim.v_s_active))

        # replace relax mask with original mask
        cuda.memcpy_htod(GPU.mask_id, sim.mask) if skyrs_set > 0 else None

        # fetch the spins from the GPU into new variable
        relaxed_spins = np.empty_like(spins)
        cuda.memcpy_dtoh(relaxed_spins, GPU.spins_id)

        return relaxed_spins

    @staticmethod
    def find_skyr_center(m_z_diff, density_threashold=1.5):
        """
        Determine the center of the skyrmion using a weighted center of mass approach.
        Skyrmion is in the negative z direction.
        m_z_diff is m_z, reduces by the init spins and set to 0 where bigger then 0


        Parameters:
        m_z_diff (2D numpy array): The mz component of the magnetization.

        Returns:
        tuple: The coordinates (y_center, x_center) of the skyrmion center.
        """
        density = -m_z_diff  # goes from 0 to 2

        if sim.final_skyr_No <= 1:
            x_indices, y_indices = np.indices(m_z_diff.shape)

            # print(f"density_max: {np.max(density)}") if idx == 0 else None
            # print(f"density_min: {np.min(density)}") if idx == 0 else None

            sig = density  # **20

            # density[density < 1] = 0

            x_center = np.sum(x_indices * sig) / np.sum(sig)
            y_center = np.sum(y_indices * sig) / np.sum(sig)

            return x_center, y_center
        else:
            potential_centers = np.argwhere(density > density_threashold)

            # Filter out points that are not local minima within the specified radius
            accepted_centers = []
            for center in potential_centers:
                y, x = center
                if density[y, x] == np.max(
                    density[
                        max(0, y - math.ceil(sim.r_skyr)) : y + math.ceil(sim.r_skyr),
                        max(0, x - math.ceil(sim.r_skyr)) : x + math.ceil(sim.r_skyr),
                    ]
                ):
                    accepted_centers.append(center)

            return np.array(accepted_centers)


if __name__ == "__main__":
    print("This Code is not meant to be run directly, but imported from main.py.")