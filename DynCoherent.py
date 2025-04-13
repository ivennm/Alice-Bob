import numpy as np
import matplotlib.pyplot as plt
import dynamiqs as dq

# -------------------------------
# Step 1: Define parameters
# -------------------------------
N = 1                   # Number of Fock states (Hilbert space dimension)
alpha = 0 + 0j          # Coherent state displacement

# -------------------------------
# Step 2: Create the coherent state
# -------------------------------
# dq.coherent(dim, alpha) returns the ket of the coherent state as a QArray.
coh_state = dq.coherent(N, alpha)

# -------------------------------
# Step 3 & 4: Plot the Wigner function
# -------------------------------
# Instead of passing custom grid arrays, we let Dynamiqs use its default grid.
plt.style.use('dark_background')  # Dark background for the entire plot
plt.rcParams['axes.facecolor'] = '#000080'
fig, ax = plt.subplots(figsize=(6, 5))

# Custom styling

ax.tick_params(colors='white')             # White ticks
ax.grid(True, color='white', linestyle='--', linewidth=0.5)  # White dashed grid lines

dq.plot.wigner(coh_state, ax=ax)  # Provide the axis via keyword 'ax'
ax.set_title(f"Wigner function for α = {alpha}")
plt.show()

# -------------------------------
# Step 5 (optional): Reconstruct α from the peak of the Wigner function
# -------------------------------
# If you need to extract the Wigner data yourself (using a custom grid),
# you would need to use another function or compute it manually.
#
# For this example, if you still want to work with a custom grid,
# you could use your own interpolation routine.
#
# For now, if we assume you have a Wigner array 'W' computed by some other means,
# here is an example of reconstructing α from it:
#
# (Remove or replace the following block as needed)
#
# Example: manually computed Wigner function from some external routine 
xvec = np.linspace(-5, 5, 500)
pvec = np.linspace(-5, 5, 500)
# Here we simulate retrieving W via the default Dynamiqs method converted to a NumPy array.
W = dq.plot.wigner(coh_state)  # using default grid
W = np.array(W)  # convert to NumPy array for processing
print(W)
