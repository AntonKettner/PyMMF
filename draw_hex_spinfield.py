import logging
import math
import gc
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree


def calculate_upscaling_array(field_size_x, field_size_y, factor):
    """
    Determine the upscaling locations that each pixel in the original hex array will be upscaled to."""

    logging.info(f"CALCULATING UPSCALING ARRAY FOR ATOMISTIC UPSCALING FACTOR {factor} and FIELD SIZE {field_size_x} x {field_size_y}\n")

    # Create arrays for i and j indices
    i, j = np.indices((field_size_x, field_size_y))

    # Calculate x and y values
    x = i + 0.5 * ((j + 1) % 2)
    y = j * np.sqrt(3) / 2

    # Combine x and y values into a single array
    orig_locations = np.dstack([x, y])

    # Create arrays for i and j indices
    i, j = np.indices((factor * field_size_x, math.ceil(field_size_y * factor * np.sqrt(3) / 2))) / factor

    # Combine x and y values into a single array
    downscaled_locations_in_orig = np.dstack([i, j])

    # Flatten the original locations and downscaled locations arrays
    orig_locations_flat = orig_locations.reshape(-1, 2)
    downscaled_locations_in_orig_flat = downscaled_locations_in_orig.reshape(-1, 2)

    # Create a KDTree from the original locations
    tree = cKDTree(orig_locations_flat)

    # Find the indices of the nearest neighbors in the original locations
    _, indices = tree.query(downscaled_locations_in_orig_flat)

    # Reshape the indices to match the shape of the upscaled array
    indices = indices.reshape(field_size_x * factor, math.ceil(field_size_y * factor * np.sqrt(3) / 2))

    # Use the indices to fill the upscaled indices array
    upscaled_indices = np.stack(np.unravel_index(indices, (field_size_x, field_size_y)), axis=-1)

    logging.info(f"UPSCALING ARRAY SHAPE {upscaled_indices.shape}\n")

    return upscaled_indices


def draw_hexagonal_spinfield(orig_array, colormap, pic_name, x_size_orig, y_size_orig, upscaling_array, factor):

    # logging.warning(f"upscaling_indices shape: {output.upscaling_indices.shape}")
    # logging.warning(f"orig_array shape: {orig_array.shape}")
    # Calculate the valid indices for orig_array

    if orig_array.shape == (x_size_orig, y_size_orig):
        valid_x_index = upscaling_array[..., 0] % orig_array.shape[0]
        valid_y_index = upscaling_array[..., 1] % orig_array.shape[1]
    else:
        temp_upscaling_indices = calculate_upscaling_array(orig_array.shape[0], orig_array.shape[1], factor)
        valid_x_index = temp_upscaling_indices[..., 0] % orig_array.shape[0]
        valid_y_index = temp_upscaling_indices[..., 1] % orig_array.shape[1]

    # Use the valid indices to index orig_array
    upscaled_hex_array = orig_array[valid_x_index, valid_y_index]
    plt.imsave(
        pic_name,
        upscaled_hex_array.T[::-1, :],
        cmap=colormap,
        vmin=-1,
        vmax=1,
    )
    plt.close()
    gc.collect()
