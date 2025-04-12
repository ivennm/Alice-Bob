import matplotlib.pyplot as plt
import jax.numpy as jnp
from dynamiqs import fock, wigner

# Create Fock state |n⟩
n = 3
dim = n + 10  # Ensure dim > n
psi = fock(dim, n)

# Compute Wigner function
wig = wigner(psi)

# Plot manually using matplotlib
x = jnp.linspace(-5, 5, wig.shape[0])
y = jnp.linspace(-5, 5, wig.shape[1])
X, Y = jnp.meshgrid(x, y)

fig, ax = plt.subplots()
c = ax.contourf(X, Y, wig, levels=100, cmap="RdBu_r")
fig.colorbar(c)
ax.set_title(f"Wigner Function of Fock State |{n}⟩")
ax.set_xlabel("x (position)")
ax.set_ylabel("p (momentum)")
plt.show()
