import cvxpy as cp
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cmath

# ---------------------------
# Load the pickle data
# ---------------------------
file_path = "data/wigner_fock_one.pickle"  # Update this if needed
with open(file_path, "rb") as f:
    data = pickle.load(f)

# ---------------------------
# Parse the data
# ---------------------------
x = data[0]  # 1D array for Re(α)
y = data[1]  # 1D array for Im(α)
W = data[2]  # 2D array for Wigner function values
a=np.zeros(len(x)*len(y),dtype=complex)
for i in range(len(x)):
    for k in range(len(y)):
        a[i*len(y)+k]=x[i]+y[k]*1j
    
w=np.zeros(len(x)*len(y))
for i in range(len(x)):
    for k in range(len(y)):
        w[i*len(y)+k]=W[i][k]


# Parameters
n = 10  # Hilbert space dimension

# Phase-space sample points (complex numbers)
alpha_list = a

# Measured Wigner function values at those points (dummy data)
wigner_measured = w

def operator_for_alpha(alpha, n):
    """
    Placeholder to construct operator A(alpha) = D(alpha) P D(alpha)†
    For now, returns a random Hermitian matrix influenced by alpha.
    """
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = (X + X.conj().T) / 2  # make it Hermitian
    return A

def compute_wigner(rho, alpha):
    """
    Compute the modeled Wigner function at alpha using:
    W(α) = (2/π) Re{ Tr[ρ A(α)] }
    """
    A_alpha = operator_for_alpha(alpha, n)
    return (2 / np.pi) * cp.real(cp.trace(rho @ A_alpha))

# Define the density matrix variable
rho = cp.Variable((n, n), complex=True)

# Constraints for a valid quantum state
constraints = [
    rho >> 0,                  # positive semidefinite
    cp.trace(rho) == 1,        # trace 1
    rho == rho.H               # Hermitian
]

# Objective: minimize squared error between model and measured Wigner values
loss_terms = []
for alpha, w_exp in zip(alpha_list, wigner_measured):
    w_model = compute_wigner(rho, alpha)
    loss_terms.append((w_model - w_exp) ** 2)

loss = cp.sum(loss_terms)
objective = cp.Minimize(loss)
prob = cp.Problem(objective, constraints)

result = prob.solve(solver=cp.SCS, verbose=True)
# Check if the problem was solved successfully

# Output
print("Optimal loss:", loss.value)
print("Reconstructed density matrix rho:")
print(rho.value)


