import json
import sqlite3
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm
import glob
import logging
import gzip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# import matplotlib.ticker as tck


def find_skyr_center(m_z, idx):
    """
    Determine the center of the skyrmion using a weighted center of mass approach.

    Parameters:
    m_z (2D numpy array): The mz component of the magnetization.

    Returns:
    tuple: The coordinates (y_center, x_center) of the skyrmion center.
    """
    x_indices, y_indices = np.indices(m_z.shape)
    density = -(m_z)  # goes from 0 to 2

    print(f"density_max: {np.max(density)}") if idx == 0 else None
    print(f"density_min: {np.min(density)}") if idx == 0 else None

    sig = density**20

    # density[density < 1] = 0

    x_center = np.sum(x_indices * sig) / np.sum(sig)
    y_center = np.sum(y_indices * sig) / np.sum(sig)

    return x_center, y_center


if __name__ == "__main__":
    # folder_name = "results_wall_retention_2_rk4_1.5_B_ext"
    folder_name = "OUTPUT/x_current_x_current/sample_1_0_deg_1_v_s_fac"

    # dest_folder = "OUTPUT/results_final_atomistic_skyrmion_creation_2.5_r_open_heun_1.5_0.5_0/sample_1_0_deg_0.5_v_s_fac/json_data"
    # dest_folder = "energy_wall_collision_1/3_Skyr_vert_curr/e_plot_comp_vs_100/json_data"
    dest_folder = os.path.join(folder_name, "json_data_2")
    # background = np.load("needed_files/background.npy")[:, :, 2]

    # löschen von dest_folder, falls es existiert
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    os.makedirs(f"{dest_folder}")

    x_min_list = []

    y_min_list = []

    spinfield_right_min = np.array([])

    spinfield_left_min = np.array([])

    integral_0 = np.array([])
    integral_90 = np.array([])
    integral_180 = np.array([])
    integral_270 = np.array([])

    radius = 13

    # --------------------------------------- json calc Section---------------------------------------

    # search for a racetrack.png file in the folder and subfolders
    folder_path = folder_name

    mask_pattern = os.path.join(folder_path, "**", "racetrack.png")

    mask_path = glob.glob(mask_pattern, recursive=True)[0]

    logging.info(f"mask_path: {mask_path}")

    scaledownfactor = 1
    # import the mask
    mask = np.ascontiguousarray(np.array(np.array(Image.open(mask_path), dtype=bool)[:, :, 0].T)[::scaledownfactor, ::scaledownfactor])

    print(mask.shape)

    npy_paths_pattern = os.path.join(folder_path, "**", "Spins_at_t_*.npy")

    npy_paths = list(glob.iglob(npy_paths_pattern, recursive=True))
    print(npy_paths[0])
    print(npy_paths[1])
    print(npy_paths[2])
    print(npy_paths[3])

    with tqdm(total=len(npy_paths), desc="Calc. jsons", unit=f"spinfield") as pbar:
        for index, filepath in enumerate(npy_paths):
            # ---------------------------------------------------------------Mandaroty: loading spinfield, getting ||-------------------------------------------------------------------

            # spinfield = np.load(f"{folder_name}/{filename}", allow_pickle=True)[2:-2, 2:-2, :]
            spinfield = np.load(filepath, allow_pickle=True)
            print(spinfield.shape) if index == 0 else None
            # spinfield_value = np.linalg.norm(spinfield, axis=-1)

            # # not necessary for jsons
            # # spinfield_value = spinfield[:, :, 0]
            # spinfield_value = spinfield[:, :, 2] - background

            # # # set all negative values of density to 0
            # spinfield_value[spinfield_value > 0] = 0

            # ---------------------------------------------------------------scale spinfield-------------------------------------------------------------------

            spinfield = spinfield[::scaledownfactor, ::scaledownfactor, :]
            mask = mask[::scaledownfactor, ::scaledownfactor]

            # scale to center --> cut a third of the length, height and depth from each border
            times_fac = 4
            center_width = spinfield.shape[0] // times_fac
            center_height = spinfield.shape[1] // times_fac
            # cut out center of spinfield
            spinfield = spinfield[
                (spinfield.shape[0] - center_width) // 2 : (spinfield.shape[0] + center_width) // 2,
                (spinfield.shape[1] - center_height) // 2 : (spinfield.shape[1] + center_height) // 2,
                :,
            ]

            mask = mask[
                (spinfield.shape[0] - center_width) // 2 : (spinfield.shape[0] + center_width) // 2,
                (spinfield.shape[1] - center_height) // 2 : (spinfield.shape[1] + center_height) // 2,
            ]

            # -----------------------------------------------------------------------------------------------------------------------------------------
            # ---------------------------------------------------------------json calc Section---------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------------------------------------------

            x_values, y_values = np.meshgrid(np.arange(spinfield.shape[1]), np.arange(spinfield.shape[0]))
            # set z-values to 0
            z_values = np.zeros(spinfield.shape[:2])

            # Flatten the arrays into lists
            x = x_values.flatten().tolist()
            y = y_values.flatten().tolist()
            z = z_values.flatten().tolist()

            # normalize the B_eff to the min value of all time --> Beff_tot --> 952; B_eff_DM --> 39.6; B_eff_exch --> 937; B_eff_aniso --> 9.3
            # spinfield /= 0.0375
            scalefactor = np.sqrt(np.sum(spinfield**2, axis=-1))

            print(spinfield.shape) if index == 0 else None

            # normalize the spinfield
            spinfield[mask] /= np.linalg.norm(spinfield[mask], axis=-1, keepdims=True)

            # Extract the spin vectors and flatten them into lists
            u = spinfield[:, :, 0]
            v = spinfield[:, :, 1]
            w = spinfield[:, :, 2]

            # definition of the polar and azimuthal angles in radians; polar_angle from 0 to 180, azimuthal_angle from 0 to 360
            polar_angle = np.arccos(w) * 180 / np.pi
            # calculate denominator first
            denom = np.sqrt(u**2 + v**2)

            # where denominator is 0, set to 1e-8 to avoid division by zero
            denom[denom == 0] = 1e-8

            # Calculate azimuthal angle, handling edge cases
            azimuthal_angle = np.zeros_like(u)
            valid_mask = denom > 1e-8
            azimuthal_angle[valid_mask] = (np.sign(v[valid_mask]) * 
                                         np.arccos(np.clip(u[valid_mask] / denom[valid_mask], -1, 1))) * 180 / np.pi

            # # test for correct conversion of x, y coordinate
            # polar_angle[11,6] = 2

            # conversion into unity rotations --> left handed coordinate system
            x_Rot = polar_angle - 90
            y_Rot = -azimuthal_angle + 90


            db_path = os.path.join(dest_folder, f"spinfield_{index}.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table without any special settings
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS spinfields(
                frame_id INTEGER PRIMARY KEY,
                x_coords BLOB,
                z_coords BLOB,
                y_coords BLOB,
                x_rot BLOB,
                y_rot BLOB,
                scalefactor BLOB
            )
            ''')
            
            # Compress arrays before storage
            def compress_data(data):
                json_str = json.dumps(data).encode('utf-8')
                compressed = gzip.compress(json_str)[2:]  # Remove gzip header
                return compressed
            
            data_tuple = (
                index,
                compress_data(x),
                compress_data(y),
                compress_data(z),
                compress_data(x_Rot.flatten().tolist()),
                compress_data(y_Rot.flatten().tolist()),
                compress_data(scalefactor.flatten().tolist())
            )

            # Debug prints
            print("\nData types in data_tuple:")
            for i, item in enumerate(data_tuple):
                print(f"Item {i}: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"Shape: {item.shape}")
                if isinstance(item, list):
                    print(f"Length: {len(item)}")
                    if len(item) > 0:
                        print(f"First element type: {type(item[0])}")

            cursor.execute('''
            INSERT INTO spinfields VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data_tuple)

            conn.commit()

            # Don't forget to close at the end:
            conn.close()

            postfix_dict = {
                "polar-min": np.min(polar_angle).round(2),
                "polar-Max": np.max(polar_angle).round(2),
                "azimuth-min": np.min(azimuthal_angle).round(2),
                "azimuth-Max": np.max(azimuthal_angle).round(2),
                "scalefactor-min": np.min(scalefactor).round(2),
                "scalefactor-Max": np.max(scalefactor).round(2)
            }
            pbar.set_postfix(postfix_dict)

            pbar.update(1)
