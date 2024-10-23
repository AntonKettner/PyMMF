import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


def setup_plt_with_tex():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    plt.rcParams["font.family"] = "serif"


def setup_plt():
    # add the path manually if necessary
    font_path = "//afs/physnet.uni-hamburg.de/users/AU/akettner/.conda/envs/2_pycuda/fonts/cmunrm.ttf"
    matplotlib.font_manager.fontManager.addfont(font_path)

    plt.rcParams["font.family"] = "CMU Serif"
    plt.rcParams["font.serif"] = "CMU Serif Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
