import pickle
import numpy as np
import matplotlib.pyplot as plt
import cmath

# ---------------------------
# Load the pickle data
# ---------------------------
file_path = "data/wigner_cat_minus.pickle"  # Update this if needed
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
    
w=np.zeros(len(x)*len(y),dtype=complex)
for i in range(len(x)):
    for k in range(len(y)):
        print(W[i][k])
        w[i*len(y)+k]=W[i][k]
