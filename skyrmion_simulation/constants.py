import numpy as np


class cst:
    """
    COMMENT AI GENERATED!!!
    A class that contains the physical constants and material parameters used in the simulation of skyrmions.

    ATTRIBUTES
    s (float)                      : Spin quantum number.
    g_el_neg (float)               : Electron g-factor.
    mu_b (float)                   : Bohr magneton in eV/T.
    mu_free_spin (float)           : Magnetic moment of free spins in eV/T.
    mu_0 (float)                   : Vacuum permeability in Vs/(Am).
    gamma_el (float)               : Gyromagnetic ratio in 1/(ns*T).
    alpha (float)                  : Gilbert damping constant.
    beta (float)                   : Dimensionless non-adiabaticity parameter.
    a (float)                      : Lattice constant in meters.
    h (float)                      : Atomic layer height in meters.
    p (float)                      : Spin polarization.
    e (float)                      : Elementary charge in C.
    coords (dict)                  : Dictionary containing the coordinates x, y, and z.
    rot_matrix_90 (numpy.ndarray)  : Rotation matrix for 90 degrees.
    Temp (float)                   : Temperature in Kelvin.
    v_s_to_j_c_factor (float)      : Conversion factor from spin velocity to current density.
    A_density (float)              : Exchange interaction density in J/m.
    DM_density (float)             : Dzyaloshinskii-Moriya interaction density in J/m^2.
    K_mu (float)                   : Anisotropy in the z-direction in J/m^3.
    B_ext (float)                  : External magnetic field in T.
    B_fields (numpy.ndarray)       : Array of external magnetic fields.
    M_s (float)                    : Saturation magnetization in A/m.
    K_density (float)              : Anisotropy density in J/m^3.
    mu_s (float)                   : Magnetic moment per spin in eV/T.
    skyr_name_ext (str)            : Extension name for skyrmion.
    B_a_quadr (float)              : Effective exchange field for quadratic lattice.
    B_d_quadr (float)              : Effective DM field for quadratic lattice.
    B_k_quadr (float)              : Effective anisotropy field for quadratic lattice.
    B_a_hex (float)                : Effective exchange field for hexagonal lattice.
    B_d_hex (float)                : Effective DM field for hexagonal lattice.
    B_k_hex (float)                : Effective anisotropy field for hexagonal lattice.
    E_a_quadr (float)              : Effective exchange energy for quadratic lattice.
    E_d_quadr (float)              : Effective DM energy for quadratic lattice.
    E_k_quadr (float)              : Effective anisotropy energy for quadratic lattice.
    E_a_hex (float)                : Effective exchange energy for hexagonal lattice.
    E_d_hex (float)                : Effective DM energy for hexagonal lattice.
    E_k_hex (float)                : Effective anisotropy energy for hexagonal lattice.
    NNs (int)                      : Number of nearest neighbors.
    hex_image_scalefactor (int)    : Scale factor for hexagonal images.
    A_Field (float)                : Exchange field.
    DM_Field (float)               : DM field.
    K_Field (float)                : Anisotropy field.
    dAdAtom (float)                : Area per atom.
    dVdAtom (float)                : Volume per atom.
    NN_vecs (numpy.ndarray)        : Nearest neighbor vectors.
    NN_pos_even_row (numpy.ndarray): Nearest neighbor positions for even rows.
    NN_pos_odd_row (numpy.ndarray) : Nearest neighbor positions for odd rows.
    rotate_anticlock (bool)        : Flag to rotate vectors anticlockwise.
    DM_vecs (numpy.ndarray)        : Dzyaloshinskii-Moriya vectors.

    METHODS
    __init__(cls, rotate_anticlock=False): Initializes the class with an optional argument to rotate vectors anticlockwise.
    rotate_vecs_90(self, vecs)           : Rotates the given vectors by 90 degrees clockwise or anticlockwise depending on the value of rotate_anticlock.
    """

    # ---------------------------------------------------------------Attributes: Physical constants-------------------------------------------------------------------

    s = 1 / 2
    g_el_neg = 2.00231930436256
    mu_b = 5.7883818012e-5
    mu_free_spin = 3 * mu_b / 6
    mu_0 = 4 * np.pi * 1e-7
    gamma_el = 176.1
    alpha = 0.1
    beta = alpha / 2
    betas = np.array([beta], dtype=np.float32)
    a = 0.271e-9
    h = 0.4e-9  # atomic layer height according to literature -> s. thesis
    p = 1  # spin polarisation
    e = 1.602176634e-19
    coords = {"x": 0, "y": 1, "z": 2}
    rot_matrix_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)  # Rotation Matrix to create the DM-vecs
    Temp = 4  # in K

    # # Konstanten Schaeffer 2018 from Romming Pd/Fe bilayer on Ir(111)
    A_density = 2e-12  # Austauschwechselwirkung in J/m
    DM_density = 3.9e-3  # DM - Wechselwirkung in J/m^2
    K_mu = 2.5e6  # Anisotropie in Z-Richtung in J/m^3
    B_ext = 1.5  # z-Magnetfeld in T
    B_fields = np.array([B_ext])  # for multiple B_exts
    M_s = 1.1e6  # Saettigungsmagnetisierung in A/m
    K_density = K_mu  # - mu_0 * M_s**2 / 2
    mu_s = 3 * mu_b
    skyr_name_ext = "schaeffer_1.5"
    r_skyr = 1.393  # in nm

    @classmethod
    def __init__(cls, model_type="atomistic", rotate_anticlock=False):
        """
        calculates the DM-vecs based on rotation and NN-vecs.

        Args:
            rotate_anticlock (bool, optional): If True, rotates the field collection
                anticlockwise by 90 degrees. Defaults to False.
        """

        cls.v_s_to_j_c_factor = -2 * cls.e / (cls.a**3 * cls.p)

        # -------------my Conversion to Effective B_field constants in a 2D QUADRATIC LATTICE from Micromagnetic Constants -----------------------

        cls.B_a_quadr = 4 * cls.A_density / (cls.a**2 * cls.M_s) * (1 / 2)

        cls.B_d_quadr = 2 * cls.DM_density / (cls.a * cls.M_s) * (1 / 2)

        cls.B_k_quadr = cls.K_density / cls.M_s

        # -------------Hagemeister 2018 Conversion zu atomistischen B_Feld-Konstanten (hexagonal lattice) -----------------------

        cls.B_a_hex = 4 * cls.A_density / (cls.a**2 * cls.M_s) * (1 / 3)

        cls.B_d_hex = 2 * cls.DM_density / (cls.a * cls.M_s) * (1 / 3)

        cls.B_k_hex = cls.K_density / cls.M_s

        # -------------my Conversion zu atomistischen Energie-Konstanten die funktioniert (quadr lattice)-----------------------

        cls.E_a_quadr = 4 * cls.A_density * cls.mu_s / (cls.a**2 * cls.M_s) * (1 / 2)

        cls.E_d_quadr = 2 * cls.DM_density * cls.mu_s / (cls.a * cls.M_s) * (1 / 2)

        cls.E_k_quadr = cls.K_density * cls.mu_s / cls.M_s

        # -------------Hagemeister 2018 Conversion zu atomistischen Energie-Konstanten (hexagonal lattice)-----------------------

        cls.E_a_hex = 4 * cls.A_density * cls.mu_s / (cls.a**2 * cls.M_s) * (1 / 3)

        cls.E_d_hex = 2 * cls.DM_density * cls.mu_s / (cls.a * cls.M_s) * (1 / 3)

        cls.E_k_hex = cls.K_density * cls.mu_s / cls.M_s

        NN_vec_1 = np.array([1, 0, 0])

        if model_type == "atomistic":
            cls.NNs = 6
            cls.hex_image_scalefactor = 4
            cls.A_Field = cls.B_a_hex
            cls.DM_Field = cls.B_d_hex
            cls.K_Field = cls.B_k_hex
            # cls.angle_between_NNs = 2 * np.pi / NNs  # against the clock rotation
        if model_type == "continuum":
            cls.NNs = 4
            cls.A_Field = cls.B_a_quadr
            cls.DM_Field = cls.B_d_quadr
            cls.K_Field = cls.B_k_quadr
            cls.dAdAtom = cls.a**2 * 2 * np.sqrt(3)
            cls.dVdAtom = cls.dAdAtom * cls.h
        rotation_by_angle = lambda angle: np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # Nearest neighbor vectors
        cls.NN_vecs = np.empty((cls.NNs, 3), dtype=np.float32)
        NN_angle = 2 * np.pi / cls.NNs
        for i in range(cls.NNs):
            cls.NN_vecs[i, :] = np.around((rotation_by_angle(i * NN_angle) @ NN_vec_1), decimals=5)

        if model_type == "atomistic":
            cls.NN_pos_even_row = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, -1, 0]])
            cls.NN_pos_odd_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]])
        if model_type == "continuum":
            cls.NN_pos_even_row = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
            cls.NN_pos_odd_row = cls.NN_pos_even_row

        cls.rotate_anticlock = rotate_anticlock
        cls.DM_vecs = cst.rotate_vecs_90(cls.NN_vecs)

    @classmethod
    def rotate_vecs_90(cls, vecs):
        """
        Rotate the given vectors by 90 degrees either clockwise or anticlockwise, depending on the value of `rotate_anticlock`.

        Args:
            vecs (numpy.ndarray): A 2D array of shape (N, 3) containing the vectors to be rotated.

        Returns:
            numpy.ndarray: A 2D array of shape (N, 3) containing the rotated vectors.
        """
        if not cls.rotate_anticlock:
            return np.ascontiguousarray(np.dot(cst.rot_matrix_90, vecs.T).T)
        else:
            return np.ascontiguousarray(np.dot(cst.rot_matrix_90.T, vecs.T).T)


if __name__ == "__main__":
    print("This Code is not meant to be run directly, but imported from main.py.")