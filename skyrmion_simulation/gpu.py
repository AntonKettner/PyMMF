# native imports
import logging  # enabling display of logging.info messages
import math

# Third party imports
import numpy as np  # Das PIL - Paket wird benutzt um die Maske zu laden

# Local imports
from simulation import sim
import pycuda.driver as cuda

# logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class GPU:

    kernel_dir_euler = "kernels/ss_kernel_euler.c"
    kernel_dir_heun = "kernels/ss_kernel_heun.c"
    kernel_dir_rk4 = "kernels/ss_kernel_rk4.c"
    kernel_dirs = {"euler": kernel_dir_euler, "heun": kernel_dir_heun, "rk4": kernel_dir_rk4}

    @classmethod
    def __init__(cls):

        # BLOCK AND GRID SIZES BASED ON MASK
        cls.block_dim_x = 8
        cls.block_dim_y = 8
        cls.grid_dim_x = math.ceil(sim.x_size / cls.block_dim_x)
        cls.grid_dim_y = math.ceil(sim.y_size / cls.block_dim_y)

        # TO BLOCK MEMORY ON GPU
        cls.avgTex = np.zeros([sim.x_size, sim.y_size], dtype=np.float32)
        cls.spins_evolved = np.zeros((sim.x_size, sim.y_size, len(sim.skyr[0, 0, :])), dtype=np.float32)
        cls.q_topo_results = np.zeros(sim.x_size * sim.y_size, dtype=np.float32)

    @classmethod
    def transfer_to_GPU(cls, spinfield):
        """
        Allocates memory on the GPU and transfers the mask, the average image of z components to the GPU.

        Parameters:
        cls (object): An instance of the class.

        Returns:
        None
        """
        # Allocation of Mem on GPU and transfer of the mask, the Average image
        cls.mask_id = cuda.mem_alloc(sim.mask.nbytes)
        cuda.memcpy_htod(cls.mask_id, sim.mask)

        # cls.avgTex         = np.zeros([sim.x_size, sim.y_size], dtype=np.float32)
        cls.avgTex_id = cuda.mem_alloc(cls.avgTex.nbytes)
        cuda.memcpy_htod(cls.avgTex_id, cls.avgTex)

        # Allocation and transfer the v_s_active bool to GPU
        cls.v_s_active_id = cuda.mem_alloc(np.int32(0).nbytes)
        cuda.memcpy_htod(cls.v_s_active_id, np.int32(sim.v_s_active))

        # As textured memory needs the kernel for sending the texture, here only the definition that it is an np array
        cls.cuda_v_s = cuda.np_to_array(sim.v_s, order="C")

        # save space for q topo calculation on GPU
        cls.q_topo_id = cuda.mem_alloc(GPU.q_topo_results.nbytes)
        cuda.memcpy_htod(cls.q_topo_id, GPU.q_topo_results)

        if sim.calculation_method == "rk4":
            cls.k1_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            cls.k2_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            cls.k3_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            # cls.pot_next_spin_id = None
            # cls.k4_gpu = cuda.mem_alloc(cls.spins_evolved.nbytes)     # not needed because no new step from there is calculated
        if sim.calculation_method == "heun":
            cls.pot_next_spin_id = cuda.mem_alloc(cls.spins_evolved.nbytes)
            # cls.k1_id = cls.k2_id = cls.k3_id = None  #! Critical maybe

        # allocate memory on GPU and transfer the spinfield
        cls.spins_id = cuda.mem_alloc(spinfield.nbytes)  # Allocate memory on GPU
        cuda.memcpy_htod(cls.spins_id, spinfield)  # Copy spins to GPU

        # # allocate memory on GPU for the current
        # cls.v_s_id = cuda.mem_alloc(sim.v_s.nbytes)

        # # transfer the v_s to GPU
        # cls.v_s_id = cuda.mem_alloc(sim.v_s.nbytes)
        # cuda.memcpy_htod(cls.v_s_id, sim.v_s)

        cuda.Context.synchronize()

    @classmethod
    def free_GPU_memory(cls):
        """
        Frees the memory allocated on the GPU.

        Parameters:
        cls (object): An instance of the class.

        Returns:
        None
        """
        # Free the memory allocated on the GPU
        cls.mask_id.free()
        cls.avgTex_id.free()
        cls.v_s_active_id.free()
        cls.q_topo_id.free()
        cls.spins_id.free()
        if sim.calculation_method == "rk4":
            cls.k1_id.free()
            cls.k2_id.free()
            cls.k3_id.free()
        elif sim.calculation_method == "heun":
            cls.pot_next_spin_id.free()
        cuda.Context.synchronize()

    @staticmethod
    def perform_numerical_step(step_func, *args):
        step_func(
            *args,
            block=(GPU.block_dim_x, GPU.block_dim_y, 1),
            grid=(GPU.grid_dim_x, GPU.grid_dim_y, 1),
        )

    @staticmethod
    def perform_numerical_steps(numerical_steps):
        if sim.calculation_method == "rk4":
            steps_args = [
                (numerical_steps[0], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.v_s_active_id),
                (numerical_steps[1], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.k2_id, GPU.v_s_active_id),
                (numerical_steps[2], GPU.spins_id, GPU.mask_id, GPU.k2_id, GPU.k3_id, GPU.v_s_active_id),
                (numerical_steps[3], GPU.spins_id, GPU.mask_id, GPU.k1_id, GPU.k2_id, GPU.k3_id, GPU.v_s_active_id),
            ]
        elif sim.calculation_method == "euler":
            # print(numerical_steps[0])
            steps_args = [(numerical_steps, GPU.spins_id, GPU.mask_id, GPU.v_s_active_id)]
        elif sim.calculation_method == "heun":
            steps_args = [
                (numerical_steps[0], GPU.spins_id, GPU.mask_id, GPU.pot_next_spin_id, GPU.v_s_active_id),
                (numerical_steps[1], GPU.spins_id, GPU.mask_id, GPU.pot_next_spin_id, GPU.v_s_active_id),
            ]
        else:
            raise ValueError(f"Unknown method: {sim.calculation_method}")

        for step_args in steps_args:
            GPU.perform_numerical_step(*step_args)


if __name__ == "__main__":
    print("This Code is not meant to be run directly, but imported from main.py.")