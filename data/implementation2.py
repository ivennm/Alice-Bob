import numpy as np

def fidelity_with_pure_state(rho, psi):
    """
    Computes fidelity between a density matrix `rho` and a pure state `psi`.

    Parameters:
    rho : numpy.ndarray
        Density matrix of the quantum state (NxN).
    psi : numpy.ndarray
        State vector (Nx1 or 1D array of length N) representing the pure reference state.

    Returns:
    float
        Fidelity F = <psi|rho|psi> as a real number.
    """
    # Ensure psi is a column vector
    psi = psi.reshape(-1, 1)

    # Compute fidelity as <psi|rho|psi>
    fidelity = np.real(psi.conj().T @ rho @ psi)

    return fidelity.item()  # return as scalar

# -----------------------------
# Example Usage:
# -----------------------------
# Define a 4x4 density matrix (rho)
rho = np.array([
    [0.7, 0.2 + 0.1j, 0, 0],
    [0.2 - 0.1j, 0.3, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Normalize the trace
rho = rho / np.trace(rho)

# Define the pure reference state |0⟩ in 4D Hilbert space
psi_ref = np.array([1, 0, 0, 0], dtype=complex)

# Compute fidelity
F = fidelity_with_pure_state(rho, psi_ref)
print("Fidelity with pure reference state:", F)
