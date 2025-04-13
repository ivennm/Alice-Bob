import numpy as np
from scipy.linalg import sqrtm
import pickle

def fidelity(rho, rho_ref):
    """
    Compute the fidelity F(rho, rho_ref) = (Tr(sqrt( sqrt(rho_ref) * rho * sqrt(rho_ref) )))^2.

    Parameters:
      rho : np.ndarray
          Reconstructed (or noisy) density matrix.
      rho_ref : np.ndarray
          Reference (ideal) density matrix.

    Returns:
      float
          Fidelity value (squared).
    """
    sqrt_rho_ref = sqrtm(rho_ref)
    intermediate = sqrt_rho_ref @ rho @ sqrt_rho_ref
    sqrt_intermediate = sqrtm(intermediate)
    fidelity_value = np.trace(sqrt_intermediate)
    fidelity_value = np.real(fidelity_value)  # remove any small imaginary residuals
    return fidelity_value**2

# ------------------------------
# Load the matrix from your pickle file.
# ------------------------------
pickle_file = "noisy_wigner_0.pickle"  # update with the proper path

with open(pickle_file, "rb") as f:
    data = pickle.load(f)

# Check if the data is a tuple (common when saving a grid and function data)
if isinstance(data, tuple):
    # Assume the file was saved as (x_values, y_values, matrix)
    x_values, y_values, noisy_matrix = data
    print("Loaded coordinate grids:")
    print("x_values:", x_values.shape)
    print("y_values:", y_values.shape)
    print("Noisy matrix shape:", noisy_matrix.shape)
else:
    # Otherwise assume data is the density matrix (or Wigner function array) itself.
    noisy_matrix = np.array(data)
    print("Noisy matrix shape:", noisy_matrix.shape)

# ------------------------------
# Define a reference density matrix.
# ------------------------------
# NOTE: The fidelity function is defined for density matrices.
# If your loaded 'noisy_matrix' represents a Wigner function or a different object,
# you'll need to convert it into an appropriate density matrix representation.
#
# For demonstration purposes, we create a dummy reference density matrix.
# Here we use a normalized identity matrix of the same dimension.
#
# (Replace this with your ideal density matrix if available.)
dim = noisy_matrix.shape[0]
rho_ref = np.eye(dim) / dim

# ------------------------------
# Compute and print the fidelity.
# ------------------------------
F = fidelity(noisy_matrix, rho_ref)
print("Fidelity (squared) between noisy matrix and reference density matrix:", F)
