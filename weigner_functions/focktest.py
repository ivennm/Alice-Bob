import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Step 1: Define parameters
N = 20             # Hilbert space size (should be > max n)
xvec = np.linspace(-5, 5, 500)

# Step 2: Loop through Fock states |n⟩
for n in range(4):  # You can change this to range(N) if you want more
    fock_state = fock(N, n)
    
    # Step 3: Compute the Wigner function
    W = wigner(fock_state, xvec, xvec)
    
    # Step 4: Plot
    plt.figure(figsize=(6,5))
    plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title(f"Wigner function for Fock state |{n}⟩")
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
