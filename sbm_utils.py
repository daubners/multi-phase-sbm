import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from IPython.display import clear_output
from scipy.optimize import curve_fit

class VoxelFields:
    def __init__(self, num_x: int, num_y: int, spacing):
        """
        Initialize voxel grid with given dimensions and cell size.
        """
        self.Nx = num_x
        self.Ny = num_y

        if not isinstance(spacing, (list, tuple)) or len(spacing) != 2:
            raise ValueError("spacing must be a list or tuple with two elements (dx, dy)")
        if not all(isinstance(x, (int, float)) for x in spacing):
            raise ValueError("All elements in spacing must be integers or floats")
        if (np.max(spacing)/np.min(spacing) > 10):
            import warnings
            warnings.warn("Simulations become very questionable for largely different spacings e.g. dz >> dx.")
        self.spacing = spacing
        self.origin = (0, 0, 0)
        self.fields = {}

    def add_field(self, name: str, array=None):
        """
        Initializes field data associated given name.
        If an array is provided, it checks if it is a numpy array
        with the correct shape; otherwise, it initializes a zero array.
        """
        if array is not None:
            # Check if array is a numpy array and if it has the right shape
            if isinstance(array, np.ndarray):
                if array.shape == (self.Nx, self.Ny):
                    self.fields[name] = array.copy()
                else:
                    raise ValueError(f"The provided array must have the shape ({self.Nx}, {self.Ny}).")
            elif isinstance(array, (int, float)):
                # Create a constant-valued numpy array
                self.fields[name] = np.full((self.Nx, self.Ny), array)
            else:
                raise TypeError("The provided array must be a numpy array or a constant float.")
        else:
        # Initialize a zero array if no array is provided
            self.fields[name] = np.zeros((self.Nx, self.Ny))

    def add_voxel_sphere(self, name: str, center_x, center_y, radius, label=1):
        """
        Create a voxelized representation of a sphere in 3D array based on
        given midpoint and radius in terms of pixel resolution.
        """
        x, y = np.ogrid[:self.Nx, :self.Ny]

        distance_squared = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2
        mask = distance_squared <= radius**2
        self.fields[name][mask] = label

    def export_to_vtk(self, filename="output.vtk", field_names=None):
        """
        Export a 3D numpy array to VTK format for visualization in VisIt or ParaView.
        """
        # Create a structured grid from the array
        grid = pv.ImageData()
        grid.dimensions = (self.Nx + 1, self.Ny + 1, 1)
        grid.spacing = self.spacing
        grid.origin = self.origin

        if field_names is not None:
            names = field_names
        else:
            names = list(self.fields.keys())

        for name in names:
            grid.cell_data[name] = self.fields[name].flatten(order="F")  # Fortran order flattening
        grid.save(filename)

    def plot_field(self, fieldname, figsize=(5,5), dpi=200, colormap='viridis'):
        plt.figure(figsize=figsize, dpi=dpi)
        end1 = self.spacing[0] * self.Nx
        end2 = self.spacing[1] * self.Ny
        label1, label2 = ['X', 'Y']

        plt.imshow(self.fields[fieldname].T, extent=[0, end1, 0, end2], origin='lower', cmap=colormap)
        plt.colorbar()
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(f"Field data: {fieldname}")
        plt.show()

class MultiPhaseSolver():
    """
    Compute multiphase-field evolution based on Allen-Cahn equations.
    Each phase should be labelled with a unique integer in given array.
    Curvature effects are removed from evolution equation such that shape is preserved.
    Set dx=1 and mobility M=1 in de-dimensionalized equation for solution.
    """
    def __init__(self, data: VoxelFields, fieldname, eps=3, stabilize = 0.0):
        self.data = data
        self.eps = eps
        self.stabilize = stabilize
        labelled_array = data.fields[fieldname]
        self.phase_count = np.unique(labelled_array).size
        self.phasefields = np.zeros((self.phase_count,*labelled_array.shape))
        self.labels = []
        for i, label in enumerate(np.unique(labelled_array)):
            self.phasefields[i] = (labelled_array == label).astype(float)
            self.labels.append(f"phi{int(label)}")

    def calc_functional_derivatives(self):
        # Define threshold close to zero to avoid division by zero
        zero = 1e-10
        df_dphi = np.zeros_like(self.phasefields)
        dfgrad_dphi = np.zeros_like(self.phasefields)
        sum_phi_squared = np.zeros_like(dfgrad_dphi[0])
        sum_dfgrad_dphi = np.zeros_like(dfgrad_dphi[0])

        # Construct slices for better readability
        # x-1: left, x+1: right, y-1: bottom, y+1: top
        center = np.s_[1:-1,1:-1]
        left   = np.s_[ :-2,1:-1]
        right  = np.s_[2:  ,1:-1]
        bottom = np.s_[1:-1, :-2]
        top    = np.s_[1:-1,2:  ]

        for i in range(self.phase_count):
            field = np.pad(self.phasefields[i], 1, mode='edge')

            norm2 = 0.25 * ((field[right] - field[left])**2) + 0.25 * ((field[top] - field[bottom])**2)
            # As we will divide by norm2, we need to take care of small values
            bulk = np.where(norm2 <= zero)
            norm2[bulk] = np.inf

            eLe = 0.25 * ((field[right] - field[left])**2) * (field[right] - 2*field[center] + field[left]) \
                + 0.25 * ((field[top] - field[bottom])**2) * (field[top]   - 2*field[center] + field[bottom]) \
                + 0.125 * (field[right] - field[left]) * (field[top] - field[bottom]) * (field[2:,2:] + field[:-2, :-2] - field[:-2,2:] - field[2:,:-2])

            laplace = field[right] - 2*field[center] + field[left] + field[top] - 2*field[center] + field[bottom]
            dfgrad_dphi[i] = self.eps*(self.stabilize*laplace + (1.0-self.stabilize)*eLe/norm2)
            sum_dfgrad_dphi += dfgrad_dphi[i]

            sum_phi_squared += field[center]*field[center]

        # Assemble derivatives of gradient and potential terms
        for i in range(self.phase_count):
            df_dphi[i] = sum_dfgrad_dphi-dfgrad_dphi[i] \
                + 27/2/self.eps*self.phasefields[i]*(sum_phi_squared - self.phasefields[i]*self.phasefields[i]) \
                + 9/2/self.eps*((self.phasefields[i])**3 - self.phasefields[i])

        return df_dphi

    def enforce_gibbs_simplex_contraint(self):
        sum = np.zeros_like(self.phasefields[0])
        for i in range(self.phase_count):
            self.phasefields[i] = np.maximum(self.phasefields[i], 0)
            sum += self.phasefields[i]

        # Normalize the fields to ensure their sum equals 1
        for i in range(self.phase_count):
            self.phasefields[i] /= sum

    def solve_without_curvature(self, steps=1000, frames=10, dt=0.02, convergence = 0.01, verbose=True):
        self.n_out = int(steps/frames)
        df_dphi = np.zeros_like(self.phasefields)

        for it in range(steps):
            if it % self.n_out == 0:
                for i in range(self.phase_count):
                    self.data.add_field(self.labels[i], self.phasefields[i])
                if verbose == 'plot':
                    clear_output(wait=True)
                    self.data.plot_field(self.labels[0])

            df_dphi = self.calc_functional_derivatives()
            sum_df_dphi = np.zeros_like(df_dphi[0])
            for i in range(self.phase_count):
                sum_df_dphi += df_dphi[i]

            for i in range(self.phase_count):
                self.phasefields[i] -= dt * (df_dphi[i] - sum_df_dphi/self.phase_count)

            self.enforce_gibbs_simplex_contraint()

        for i in range(self.phase_count):
            self.data.add_field(self.labels[i], self.phasefields[i])

# Constant factor: elementary charge divided by Boltzman constant and room temperature
e_over_kbT = 1.602176634e-19 / (1.380649e-23*300)
kbT_over_e = (1.380649e-23*300) / 1.602176634e-19

def free_energy_log(x, A, K, c0, c1, B, factor=kbT_over_e):
    """f = A * x + K * ((x-c0) * ln(x-c0) + (c1 - x) * ln(c1 - x)) + B"""
    # Avoid log of zero by adding a small epsilon
    epsilon = 1e-10
    c0_minus_x_safe = np.clip(x - c0, epsilon, None)
    c1_minus_x_safe = np.clip(c1 - x, epsilon, None)
    return (A*x +  K*(c0_minus_x_safe * np.log(c0_minus_x_safe) + c1_minus_x_safe * np.log(c1_minus_x_safe)) + B)*factor

def chemical_potential_log(x, A, K, c0, c1, factor=kbT_over_e):
    """f = A + K * (ln(x - c0) - ln(c1 - x))"""
    # Avoid log of zero by adding a small epsilon
    epsilon = 1e-10
    c0_minus_x_safe = np.clip(x - c0, epsilon, None)
    c1_minus_x_safe = np.clip(c1 - x, epsilon, None)
    return (A + K*(np.log(c0_minus_x_safe) - np.log(c1_minus_x_safe)))*factor

def ci_of_mu_log(mu, A, K, c0, c1, factor=kbT_over_e):
    """f = A + K * (ln(x - c0) - ln(c1 - x))"""
    return (c0 + c1 * np.exp((mu/factor-A)/K)) / (1+np.exp((mu/factor-A)/K))

def dci_dmu_log(mu, A, K, c0, c1, factor=kbT_over_e):
    """f = A + K * (ln(x - c0) - ln(c1 - x))"""
    return (c1-c0) / factor / K * np.exp((mu/factor-A)/K) / (1+np.exp((mu/factor-A)/K)) / (1+np.exp((mu/factor-A)/K))

def free_energy_quad(x, A, K, B, factor=kbT_over_e):
    """f = A*x^2 + K*x + B"""
    return (A*x*x + K*x + B)*factor

def chemical_potential_quad(x, A, K, factor=kbT_over_e):
    """f = 2*A*x + K"""
    return (2*A*x + K) * factor

def ci_of_mu_quad(mu, A, K, factor=kbT_over_e):
    """f = (mu - K) / (2*A)"""
    return (mu/factor - K) / (2*A)

def dci_dmu_quad(mu, A, K, factor=kbT_over_e):
    """f = 1 / (2*A)"""
    return 1 / factor / (2*A) * np.ones_like(mu)

def fit_experimental_voltage(phases, fit_type, limits, domain_limit, voltage_data, manual_bounds=None):
    # Fit functions to the data
    param_fit = {}
    c_range = 0.2

    for i in range(len(phases)):
        subset = (voltage_data[0] >= limits[i][0]) & (voltage_data[0] <= limits[i][1])
        fit_func = globals()[f"chemical_potential_{fit_type[i]}"]
        if fit_type[i] == "log":
            initial_guess = [-100, 10.0, limits[i][0], limits[i][1]]
            lower_bounds = [-300, 1, max(domain_limit[0], limits[i][0]-c_range), limits[i][1]]
            upper_bounds = [300, 50, limits[i][0], min(limits[i][1]+c_range, domain_limit[1])]
            if manual_bounds:
                initial_guess = [-100, 10.0, manual_bounds[0][i][0], manual_bounds[1][i][1]]
                lower_bounds = [-300, 5, manual_bounds[0][i][0], manual_bounds[0][i][1]]
                upper_bounds = [300, 50, manual_bounds[1][i][0], manual_bounds[1][i][1]]
        elif fit_type[i] == "quad":
            initial_guess = [50.0, 1]
            lower_bounds = [40, -1000]
            upper_bounds = [1000, 1000]
        param_fit[phases[i]], _ = curve_fit(fit_func, voltage_data[0][subset], -voltage_data[1][subset], p0=initial_guess, bounds=(lower_bounds, upper_bounds))

    phase_limits = np.copy(np.array(limits))
    for i in range(1,len(phases)):
        mu_first = globals()[f"chemical_potential_{fit_type[i-1]}"]
        mu_second = globals()[f"chemical_potential_{fit_type[i]}"]
        mu_mean = 0.5*(mu_first(limits[i-1][1], *param_fit[phases[i-1]]) + mu_second(limits[i][0], *param_fit[phases[i]]))
        c_first = globals()[f"ci_of_mu_{fit_type[i-1]}"]
        c_second = globals()[f"ci_of_mu_{fit_type[i-1]}"]
        phase_limits[i-1][1] = c_first(mu_mean, *param_fit[phases[i-1]])
        phase_limits[i][0] = c_second(mu_mean, *param_fit[phases[i]])

    B_fit = {}
    f_func  = globals()[f"free_energy_{fit_type[0]}"]
    B_fit[phases[0]] = (0 - f_func(phase_limits[0][0], *param_fit[phases[0]], 0.0))*e_over_kbT

    for i in range(1,len(phases)):
        f_first  = globals()[f"free_energy_{fit_type[i-1]}"]
        mu_first = globals()[f"chemical_potential_{fit_type[i-1]}"]
        f_second = globals()[f"free_energy_{fit_type[i]}"]
        B_fit[phases[i]] = (f_first(phase_limits[i-1][1], *param_fit[phases[i-1]], B_fit[phases[i-1]]) \
                + mu_first(phase_limits[i-1][1], *param_fit[phases[i-1]])*(phase_limits[i][0]-phase_limits[i-1][1]) \
                - f_second(phase_limits[i][0], *param_fit[phases[i]], 0.0) ) *e_over_kbT

    f_func  = globals()[f"free_energy_{fit_type[-1]}"]
    V_ref = -f_func(phase_limits[-1][1], *param_fit[phases[-1]], B_fit[phases[-1]]) / (phase_limits[-1][1]-phase_limits[0][0])

    for i in range(len(phases)):
        if fit_type[i] == "log":
            param_fit[phases[i]][0] += V_ref*e_over_kbT
        elif fit_type[i] == "quad":
            param_fit[phases[i]][1] += V_ref*e_over_kbT
        B_fit[phases[i]] -= V_ref*e_over_kbT*limits[0][0]

    return V_ref, param_fit, B_fit, phase_limits