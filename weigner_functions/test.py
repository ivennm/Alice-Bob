import matplotlib.pyplot as plt
import jax.numpy as jnp
from dynamiqs.states import fock
from dynamiqs.wigner import wigner_function
from dynamiqs.grid import PhaseSpaceGrid

# Choose the Fock state |n⟩
n = 3  # Change this to try other Fock states
dim = 20  # Hilbert space cutoff dimension (must be > n)

# Compute the Fock state vector
psi = fock(n=n, dim=dim)

# Create a phase space grid for Wigner function computation
grid = PhaseSpaceGrid(x_max=5.0, x_points=100, p_max=5.0, p_points=100)

# Compute the Wigner function
wigner = wigner_function(psi, grid)

# Plot the Wigner function
X, P = jnp.meshgrid(grid.x, grid.p)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
contour = ax.contourf(X, P, wigner, levels=100, cmap="RdBu_r")
plt.colorbar(contour)
ax.set_title(f"Wigner function of Fock state |{n}⟩")
ax.set_xlabel("x (position)")
ax.set_ylabel("p (momentum)")
plt.tight_layout()
plt.show()
