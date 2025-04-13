import numpy as np
import cvxpy as cp
from scipy.linalg import expm, sqrtm
import pickle
import matplotlib.pyplot as plt
import cmath

def fidelity(rho, rho_ref):
    # Compute the square root of the reference density matrix.
    sqrt_rho_ref = sqrtm(rho_ref)

    #Compute the intermediate product sqrt(rho_ref) * rho * sqrt(rho_ref)
    intermediate = sqrt_rho_ref @ rho @ sqrt_rho_ref
    
    #Compute the square root of the intermediate product.
    sqrt_intermediate = sqrtm(intermediate)

    #Calculate the trace and ensure the result is real (to avoid small imaginary parts).
    fidelity_value = np.trace(sqrt_intermediate)
    fidelity_value = np.real(fidelity_value)

    #Return fidelity squared.
    return fidelity_value**2

# ---------------------------
# Load the pickle data
# ---------------------------
file_path = "data/wigner_fock_zero.pickle"  # Update this if needed
with open(file_path, "rb") as f:
    data = pickle.load(f)

# ---------------------------
# Parse the data
# ---------------------------
x = data[0]  # 1D array for Re(α)
y = data[1]  # 1D array for Im(α)
W = data[2]  # 2D array for Wigner function values
a=np.zeros(len(x)*len(y),dtype=complex)
for i in range(len(y)):
    for k in range(len(x)):
        a[i*len(y)+k]=x[k]+y[i]*1j
    
w=np.zeros(len(x)*len(y))
for i in range(len(y)):
    for k in range(len(x)):
        w[i*len(y)+k]=W[i][k]

def operator_for_alpha(alpha, n):
    a = np.zeros((n, n), dtype=complex)
    for j in range(1, n):
        a[j - 1, j] = np.sqrt(j)
    ad = a.conjugate().T
    D_alpha = expm(alpha * ad - np.conjugate(alpha) * a)
    P = np.diag([(-1)**j for j in range(n)])  # Fixed exponentiation
    DPD = D_alpha @ P @ D_alpha.conjugate().T
    I = np.eye(n, dtype=complex)
    E_alpha = 0.5 * (I + DPD)
    return E_alpha

def compute_model_wigner(rho, E_alpha):
    p_alpha = cp.real(cp.trace(rho @ E_alpha))
    W_model = (2 / np.pi) * (2 * p_alpha - 1)  # Fixed syntax
    return W_model

if __name__ == "__main__":
    n = 30
    alpha_list = a
    # Simulated Wigner values for a vacuum state
    wigner_measured = w

    # Define optimization variable
    rho = cp.Variable((n, n), complex=True)

    # Constraints: rho is Hermitian, positive semi-definite, and trace 1
    constraints = [rho >> 0, cp.trace(rho) == 1, rho == rho.H]

    # Construct loss function
    loss_terms = []
    for alpha, W_meas in zip(alpha_list, wigner_measured):
        E_alpha = operator_for_alpha(alpha, n)
        W_model = compute_model_wigner(rho, E_alpha)
        loss_terms.append(cp.square(W_model - W_meas))
    loss = cp.sum(loss_terms)

    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    result = prob.solve(solver=cp.MOSEK, verbose=True)  # You can replace SCS with MOSEK if installed

    print("Optimal loss:", loss.value)
    print("Reconstructed density matrix rho:")
    print(rho.value)

    # Normalize rho so that trace(rho) = 1.
    rho_val = rho.value / np.trace(rho.value)

    #Example reference state: the vacuum state (|0><0|) embedded in a n-dimensional space.
    rho_ref = np.zeros((n, n), dtype=complex)
    rho_ref[0, 0] = 1.0  # Only the first element is nonzero.

    #Compute the fidelity.
    F = fidelity(rho_val, rho_ref)
    print("Fidelity between rho and rho_ref:", F)