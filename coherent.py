import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Step 1: Define parameters
N = 2  # number of Fock states (Hilbert space size)
alpha = 0 + 0j  # coherent state displacement

# Step 2: Create the coherent state
coherent_state = coherent(N, alpha)

# Step 3: Generate the Wigner function
xvec = np.linspace(-5, 5, 500)
W = wigner(coherent_state, xvec, xvec)

# Step 4: Plot the Wigner function
plt.figure(figsize=(6,5))
plt.contourf(xvec, xvec, W, 100)
plt.xlabel("x")
plt.ylabel("p")
plt.title(f"Wigner function for α = {alpha}")
plt.colorbar()
plt.grid(True)
plt.show()

# Step 5 (optional): Reconstruct α from the peak of the Wigner function
ix, iy = np.unravel_index(np.argmax(W), W.shape)
x0, p0 = xvec[iy], xvec[ix]
alpha_reconstructed = (x0 + 1j * p0) / np.sqrt(2)
print(f"Reconstructed α: {alpha_reconstructed:.3f}")
