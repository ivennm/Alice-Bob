import numpy as np
from scipy.linalg import sqrtm

def fidelity(rho, rho_ref):
    """
    Compute the fidelity F(rho, rho_ref) = (Tr(sqrt( sqrt(rho_ref) * rho * sqrt(rho_ref) )))^2.

    Parameters:
    rho : numpy array
        Density matrix of the state.
    rho_ref : numpy array
        Reference density matrix.

    Returns:
    float
        Fidelity value as a real number.
    """
    # Compute the square root of the reference density matrix
    sqrt_rho_ref = sqrtm(rho_ref)

    # Compute the intermediate product
    intermediate = sqrt_rho_ref @ rho @ sqrt_rho_ref

    # Compute the square root of the intermediate product
    sqrt_intermediate = sqrtm(intermediate)

    # Calculate the trace of the square root
    fidelity_value = np.trace(sqrt_intermediate)

    # Take the real part to avoid small imaginary numerical errors
    fidelity_value = np.real(fidelity_value)

    # Return the fidelity squared
    return fidelity_value**2


# -----------------------------
# Example Usage:
# -----------------------------

# Example target density matrix (rho)
rho = np.array([
    [0.7, 0.2 + 0.1j, 0, 0],
    [0.2 - 0.1j, 0.3, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Normalize to ensure trace = 1
rho = rho / np.trace(rho)

# Example reference density matrix (rho_ref) – vacuum state |0><0| in 4D space
rho_ref = np.zeros((4, 4), dtype=complex)
rho_ref[0, 0] = 1.0

# Compute fidelity
F = fidelity(rho, rho_ref)
print("Fidelity between rho and rho_ref:", F)
