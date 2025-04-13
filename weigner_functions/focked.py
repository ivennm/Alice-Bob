import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Step 1: Define parameters
N = 10      # Hilbert space dimension (must be > n)
n = 1       # Fock state to visualize
xvec = np.linspace(-5, 5, 500)

# Step 2: Create Fock state |n⟩
fock_state = fock(N, n)

# Step 3: Compute the Wigner function
W = wigner(fock_state, xvec, xvec)

# Step 4: Plot with dark background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(xvec, xvec, W, 100, cmap='Blues')

# Style customization
ax.set_facecolor("#0c1a3a")  # deep blue background
ax.set_title(f"Wigner Function for Fock State |{n}⟩", color='white')
ax.set_xlabel("x", color='white')
ax.set_ylabel("p", color='white')
ax.tick_params(colors='white')
plt.colorbar(contour)

# Grid with white lines
ax.grid(True, color='white', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
