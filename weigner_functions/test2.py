import matplotlib.pyplot as plt
from dynamiqs import fock
from dynamiqs.wigner import wigner
from dynamiqs.plot import plot_wigner

# Create Fock state |n⟩
n = 3
dim = 20
psi = fock(n, dim)

# Compute Wigner function
wig = wigner(psi)

# Plot the Wigner function
plot_wigner(wig)
plt.title(f"Wigner Function of Fock State |{n}⟩")
plt.show()
