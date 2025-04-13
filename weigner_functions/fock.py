import numpy as np
import matplotlib.pyplot as plt
import dynamiqs as dq

# -------------------------------
# Step 1: Define parameters
# -------------------------------
# Set the Hilbert space dimension to be larger than the Fock state number.
dim = 10            # Total number of Fock states in the Hilbert space
n_state = 1         # Fock state index (e.g. |1>)

# -------------------------------
# Step 2: Create the Fock state
# -------------------------------
# dq.fock(dim, n_state) returns the ket of the Fock state |n_state⟩ as a QArray.
fock_state = dq.fock(dim, n_state)

# -------------------------------
# Step 3: Plot the Wigner function for the Fock state
# -------------------------------
# We'll use the built-in grid provided by Dynamiqs for plotting.
plt.style.use('dark_background')               # Set the dark background for the plot
plt.rcParams['axes.facecolor'] = '#000080'       # Set the axis background color

# Create a figure and axis with custom styling
fig, ax = plt.subplots(figsize=(6, 5))
ax.tick_params(colors='white')                 # White ticks for better visibility
ax.grid(True, color='white', linestyle='--', linewidth=0.5)  # White dashed grid lines

# Use Dynamiqs built-in function for plotting the Wigner function
dq.plot.wigner(fock_state, ax=ax)              # Pass the axis via the 'ax' keyword
ax.set_title(f"Wigner function for Fock state |{n_state}>")

plt.show()
