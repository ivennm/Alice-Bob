import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import sys

try:
    # Parameters
    alpha = 2.0             # Amplitude of coherent states
    kappa = 0.1             # Dissipation rate
    N = 20                  # Hilbert space truncation
    tlist = np.linspace(0, 10, 100)  # Time evolution points

    print(f"Initializing simulation with parameters: alpha={alpha}, kappa={kappa}, N={N}")

    # Create even cat state: (|alpha> + |-alpha>) / sqrt(2 + 2 e^{-2|alpha|^2})
    coh = coherent(N, alpha)
    cat_state = (coh + coherent(N, -alpha)).unit()
    
    # Verify the state is properly normalized
    print(f"Cat state norm: {cat_state.norm()}")

    # Collapse operator (photon loss)
    a = destroy(N)
    c_ops = [np.sqrt(kappa) * a]

    # Hamiltonian (none in this example, pure dissipation)
    H = 0 * a.dag() * a

    print("Starting master equation solver...")
    # Solve master equation
    result = mesolve(H, cat_state, tlist, c_ops, [])
    print("Master equation solved successfully")

    # Reconstruct and plot Wigner functions at selected times
    xvec = np.linspace(-5, 5, 200)
    snapshots = [0, 20, 40, 60, 80, 99]
    
    print("Creating plots...")
    fig, axes = plt.subplots(1, len(snapshots), figsize=(18, 3))
    
    for i, idx in enumerate(snapshots):
        print(f"Processing snapshot {i+1}/{len(snapshots)} at time t={tlist[idx]:.2f}")
        w = wigner(result.states[idx], xvec, xvec)
        axes[i].contourf(xvec, xvec, w, 100, cmap='RdBu_r')
        axes[i].set_title(f"t = {tlist[idx]:.2f}")
        axes[i].set_xlabel("q")
        axes[i].set_ylabel("p")

    plt.suptitle("Wigner Function Evolution of a Dissipative Cat State", fontsize=16)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    print("Stack trace:")
    import traceback
    traceback.print_exc() 
    
