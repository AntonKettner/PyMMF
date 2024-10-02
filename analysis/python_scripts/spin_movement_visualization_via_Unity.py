import json
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# import matplotlib.ticker as tck


def linear_fit(x, y):
    """
    Fit x and y data with a linear model and return the slope, intercept, and fitted data.

    Parameters:
    x (array-like): The x-coordinates of the data.
    y (array-like): The y-coordinates of the data.

    Returns:
    tuple: slope (a), intercept (b), fitted data (y_fit)
    """
    x = np.array(x)[:]  # Ensure x is a numpy array
    y = np.array(y)[:]  # Ensure y is a numpy array too, for consistency

    # Perform linear fit
    a, b = np.polyfit(x, y, 1)

    # Generate the fitted data
    y_fit = a * x + b

    return a, b, y_fit


def format_func(value, tick_number):
    return "{:.1g}".format(value)


def angled_line_and_r(matrix, x, y, x_plus=1, y_plus=1, red=0, max_r=200):
    # the default is 45 deg
    values = []

    # Get matrix dimensions
    rows, cols = matrix.shape

    while 0 <= x < rows and 0 <= y < cols:
        values.append(matrix[x, y] - red)
        x += x_plus
        y += y_plus

    step_length = np.sqrt(x_plus**2 + y_plus**2)

    # cut off at max_r
    if len(values) * step_length > max_r:
        values = values[: int(np.floor(max_r / step_length))]

    r = np.arange(len(values)) * step_length

    return np.array(values), r


def avg(matrix, n):
    # Ensure matrix is a numpy array
    matrix = np.asarray(matrix)

    # Check if matrix is 1D
    if len(matrix.shape) != 1:
        raise ValueError("The input matrix should be a 1D numpy array.")

    # Create a uniform kernel of size 2*n + 1
    kernel_size = 2 * n + 1
    kernel = np.ones(kernel_size) / kernel_size

    # Convolve data with the kernel
    averaged_data = np.convolve(matrix, kernel, mode="same")

    # Handle edge cases
    for i in range(n):
        averaged_data[i] = matrix[: i + n + 1].mean()
        averaged_data[-(i + 1)] = matrix[-(i + n + 1) :].mean()

    return averaged_data


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
    folder_name = "OUTPUT/test_until_ROMMING_FINAL_SIMatomistic_first_results_replica_2.5_r_open_heun_1.5_6_0/sample_1_0_deg_6_v_s_fac"

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

            # ---------------------------------------------------------------follow trajectory-------------------------------------------------------------------

            # # # very rough technique to find the center of the skyrmion
            # x_min, y_min = np.unravel_index(np.argmin(spinfield_value), spinfield_value.shape)

            # # weighted mass center --> #TODO: calculates further away then the center of the skyrmion
            # # follow trajectory of the center of the skyrmion
            # x_min, y_min = find_skyr_center(spinfield_value, index)
            # if x_min > 100:
            #     x_min_list.append(x_min)
            # else:
            #     x_min_list.append(250)
            # y_min_list.append(y_min)

            # if index % 159 == 0 and x_min > 100:
            #     plt.imshow(
            #         (spinfield_value).T,
            #         origin="lower",
            #         cmap="seismic",
            #         interpolation="none",
            #     )
            #     # plt.colorbar()
            #     # plt.title("m_z Component with Manually Computed Center")
            #     plt.scatter(
            #         x_min_list[-1], y_min_list[-1], color="yellow", marker="x", s=100
            #     )  # mark the computed center
            #     # plt.show()
            #     plt.xlim(x_min_list[-1] - radius, x_min_list[-1] + radius)
            #     plt.ylim(y_min_list[-1] - radius, y_min_list[-1] + radius)
            #     plt.savefig(f"{dest_folder}/spinfield_{index:04d}.png", dpi=500)
            #     plt.close()
            # ---------------------------------------------------------------save spinfield image-------------------------------------------------------------------

            # # save image of entire spinfield
            # E_sum = 0.037513 * -4
            # spinfield_value -= E_sum

            # print(f"spinfield_value_max: {np.max(spinfield_value)}")
            # print(f"spinfield_value_min: {np.min(spinfield_value)}")

            # plt.imsave(
            #     f"{dest_folder}/spinfield_{index}.png",
            #     spinfield_value.T[::-1, :],
            #     cmap="seismic",
            #     vmin=-0.0004,
            #     vmax=0.0004,
            # )
            # plt.close()

            # ---------------------------------------------------------------center Spinfield around Skyr-------------------------------------------------------------------

            # # make sum along y axis of spinfield_value

            # spinfield_sum_y = np.sum(spinfield_value, axis=1)

            # # split spinfield into 2 parts, right and left of x_min
            # spinfield_right = spinfield_sum_y[x_min_list[-1] : x_min_list[-1] + 101]
            # spinfield_left = spinfield_sum_y[x_min_list[-1] - 100 : x_min_list[-1] + 1]

            # # invert x axis of spinfield_left
            # spinfield_left = spinfield_left[::-1]

            # current_min_left = np.argmin(spinfield_left)
            # current_min_right = np.argmin(spinfield_right)

            # # get the minima of these curves:
            # spinfield_left_min = np.append(spinfield_left_min, current_min_left)
            # spinfield_right_min = np.append(spinfield_right_min, current_min_right)

            # ---------------------------------------------------------------save directional energy plot-------------------------------------------------------------------

            # # save plot of energy in one dir
            # E_sum_one = 0.0374645 * (-4) / 2

            # # alt: 0.037513 * (-4)

            # # e_plot = spinfield_value[x_min_list[-1] - 60 : x_min_list[-1] + 60, y_min_list[-1]] - E_sum_one

            # # print(f"e_plot_shape: {e_plot.shape}") if index == 0 else None

            # # # Generate x values for vertical / horizontal
            # # r = np.arange(e_plot.shape[0])

            # # split values into angles starting right and going counter clockwise in 45° steps
            # # e_plot_0, r_0 = spinfield_value[x_min_list[-1] : x_min_list[-1] + 101, y_min_list[-1]] - E_sum_one

            # e_plot_0, r_0 = angled_line_and_r(
            #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=1, y_plus=0, red=E_sum_one
            # )

            # # e_plot_45, r_45 = angled_line_and_r(
            # #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=1, y_plus=1, red=E_sum_one
            # # )

            # e_plot_90, r_90 = angled_line_and_r(
            #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=0, y_plus=1, red=E_sum_one
            # )

            # # e_plot_135, r_135 = angled_line_and_r(
            # #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=-1, y_plus=1, red=E_sum_one
            # # )

            # e_plot_180, r_180 = angled_line_and_r(
            #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=-1, y_plus=0, red=E_sum_one
            # )

            # # e_plot_225, r_225 = angled_line_and_r(
            # #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=-1, y_plus=-1, red=E_sum_one
            # # )

            # e_plot_270, r_270 = angled_line_and_r(
            #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=0, y_plus=-1, red=E_sum_one
            # )

            # # e_plot_315, r_315 = angled_line_and_r(
            # #     spinfield_value, x=x_min_list[-1], y=y_min_list[-1], x_plus=1, y_plus=-1, red=E_sum_one
            # # )

            # # e_plot_180 = spinfield_value[x_min_list[-1] - 100 : x_min_list[-1] + 1, y_min_list[-1]] - E_sum_one
            # # e_plot_180 = e_plot_180[::-1]

            # # Generate x values

            # # # Create the plot with Skyrmion in Center
            # # plt.figure()
            # # plt.plot(r, e_plot, "r-")

            # # Create a plot
            # plt.figure()

            # # plot with overlayed all angles
            # plt.plot(r_0, e_plot_0, color="blue", label="0")
            # # plt.plot(r_45, e_plot_45, color="green", label="45")
            # plt.plot(r_90, e_plot_90, color="red", label="90")
            # # plt.plot(r_135, e_plot_135, color="purple", label="135")
            # plt.plot(r_180, e_plot_180, color="orange", label="180")
            # # plt.plot(r_225, e_plot_225, color="pink", label="225")
            # plt.plot(r_270, e_plot_270, color="brown", label="270")
            # # plt.plot(r_315, e_plot_315, color="black", label="315")

            # # plot at every angle of 45°

            # # add point if y pos is changing
            # if len(y_min_list) > 1 and y_min_list[-1] < y_min_list[-2]:
            #     plt.plot(1, 0, "ro")

            # # add point if x pos is changing
            # if len(x_min_list) > 1 and not x_min_list[-1] == x_min_list[-2]:
            #     plt.plot(5, 0, "go")

            # # add standard legend
            # plt.legend(loc="upper right")

            # time = os.path.splitext(filename)[0].split("_")[3]
            # # plt.yscale("log")
            # plt.title(f"Energy at t = {time} from Skyr center to top and bottom")
            # plt.xlabel("y around skyr (-100 to 100) [0.4 nm]")
            # plt.ylabel("Energy [eV] - E_ferromagnetic")
            # # plt.ylim([-0.00015, 0.00045])
            # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
            # plt.tight_layout()
            # # Save the plot
            # plt.savefig(f"{dest_folder}/energy_{index:04d}.png")
            # plt.close()

            # -----------------------------------------------------plot of integrals of functions over time--------------------------------------------

            # # set integral limit
            # limit = 150

            # # save the integral data in integral_90 and integral_270
            # integral_0 = np.append(integral_0, np.trapz(e_plot_0[:limit], r_0[:limit]))
            # integral_90 = np.append(integral_90, np.trapz(e_plot_90[:limit], r_90[:limit]))
            # integral_180 = np.append(integral_180, np.trapz(e_plot_180[:limit], r_180[:limit]))
            # integral_270 = np.append(integral_270, np.trapz(e_plot_270[:limit], r_270[:limit]))

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
            azimuthal_angle = (np.sign(v) * np.arccos(u / np.sqrt(u**2 + v**2))) * 180 / np.pi

            azimuthal_angle = np.nan_to_num(azimuthal_angle, nan=0)

            # # test for correct conversion of x, y coordinate
            # polar_angle[11,6] = 2

            # conversion into unity rotations --> left handed coordinate system
            x_Rot = polar_angle - 90
            y_Rot = -azimuthal_angle + 90

            # as conversion into unity rotations --> left handed coordinate system --> y and z are switched
            data = {
                "x": x,
                "z": y,
                "y": z,
                "x_Rot": x_Rot.flatten().tolist(),
                "y_Rot": y_Rot.flatten().tolist(),
                "scalefactor": scalefactor.flatten().tolist(),
            }

            # # with this I think in unity x, z should work, not z, x
            # y *= -1
            # data = {
            #     "x": x,
            #     "y": z,
            #     "z": y,
            #     "x_Rot": x_Rot.flatten().tolist(),
            #     "y_Rot": y_Rot.flatten().tolist(),
            #     "scalefactor": scalefactor.flatten().tolist(),
            # }

            # print(f"polar_angle_min: {np.min(polar_angle):.2f}")
            # print(f"polar_angle_min: {np.min(polar_angle):.2f}")
            # print(f"azimuthal_angle_min: {np.min(azimuthal_angle):.2f}")
            # print(f"azimuthal_angle_min: {np.min(azimuthal_angle):.2f}")

            # # debug prints
            # position = (37,0)
            # print(f"x, y, z {position}: {spinfield[position[0], position[1],:]}")
            # print(f"azimuthal_angle[{position}]: {azimuthal_angle[position]:.2f}")
            # print(f"polar_angle[{position}]: {polar_angle[position]:.2f}")
            # print(f"x_Rot[{position}]: {x_Rot[position]:.2f}")
            # print(f"y_Rot[{position}]: {y_Rot[position]}")

            # if np.min(polar_angle) > 180 or np.min(polar_angle) < 0 or np.min(azimuthal_angle) > 360 or np.min(azimuthal_angle) < 0:
            #     print("The angles are not in the correct range!")

            # Save as JSON
            with open(f"{dest_folder}/spinfield_{index}.json", "w") as f:
                json.dump(data, f)

            postfix_dict = {
                "polar-min": np.min(polar_angle).round(2),
                "polar-Max": np.max(polar_angle).round(2),
                "azimuth-min": np.min(azimuthal_angle).round(2),
                "azimuth-Max": np.max(azimuthal_angle).round(2),
                "scalefactor-min": np.min(scalefactor).round(2),
                "scalefactor-Max": np.max(scalefactor).round(2),
            }
            pbar.set_postfix(postfix_dict)

            pbar.update(1)

    # ---------------------------------------------------------------Trajectory plot-------------------------------------------------------------------

    # plt.figure()
    # a, b, fit = linear_fit(x_min_list, y_min_list)
    # plt.plot(x_min_list, y_min_list, "ro", label="Trajectory")
    # plt.plot(x_min_list[::159], y_min_list[::159], "go", label="Trajectory")

    # plt.plot(
    #     x_min_list[:],
    #     fit,
    #     "g-",
    #     label=f"Linear Fit,\na = {float(a):.5g},\nb = {float(b):.5g}",
    # )
    # print(f"a: {a}")
    # plt.title("Graph of Trajectory (center of skyr)")
    # plt.legend(loc="upper right")
    # plt.xlabel("x [0.4 nm]")
    # plt.ylabel("y [0.4 nm]")
    # # plt.xlim([240, 580])
    # # plt.ylim([60, 400])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_folder}/trajectory_davor.png", dpi=800)
    # plt.close()

    # x_y_array = np.array([x_min_list, y_min_list])

    # np.save(f"{dest_folder}/trajectory.npy", x_y_array)

    # ---------------------------------------------------------------minima over time plot-------------------------------------------------------------------

    # x = np.arange(spinfield_right_min.shape[0])

    # plt.figure()
    # plt.plot(x[1:], spinfield_left_min[1:], "r-")
    # plt.plot(x[1:], spinfield_right_min[1:], "g-")
    # plt.title("comparison of right and left Energy minima distance from center of Skyr")
    # plt.xlabel("t")
    # plt.ylabel("y [0.4 nm]")
    # # plt.xlim([240, 580])
    # plt.ylim([0, 40])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_folder}/Minima_distances.png", dpi=500)
    # plt.close()

    # ---------------------------------------------------------------integral plot-------------------------------------------------------------------

    # x_0 = np.arange(integral_0.shape[0])
    # x_90 = np.arange(integral_90.shape[0])
    # x_180 = np.arange(integral_180.shape[0])
    # x_270 = np.arange(integral_270.shape[0])

    # print(integral_90)
    # print(integral_270)

    # plt.figure()
    # plt.plot(x_0, avg(integral_0, 20), color="blue", label=0)
    # plt.plot(x_90, avg(integral_90, 20), color="red", label=90)
    # plt.plot(x_180, avg(integral_180, 20), color="orange", label=180)
    # plt.plot(x_270, avg(integral_270, 20), color="brown", label=270)
    # plt.title("integrals over time")
    # plt.xlabel("t")
    # plt.ylabel("E_tot bis r = 150")

    # # add standard legend
    # plt.legend(loc="upper right")

    # # plt.xlim([240, 580])
    # # plt.ylim([0, 40])
    # # plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(format_func))
    # # plt.tight_layout()
    # # Save the plot
    # plt.savefig(f"{dest_folder}/integrals_over time.png", dpi=500)
    # plt.close()

    # ---------------------------------------------------------------video from pngs with ffmpeg-------------------------------------------------------------------

    # os.system(
    #     f"ffmpeg -framerate 5 -pattern_type glob -i '{dest_folder}/*.png' -vcodec mpeg4 -y {dest_folder}/movie.mp4"
    # )
