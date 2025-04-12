import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, tensor, wigner, qeye

# ── Parameters ────────────────────────────────────────────────────────────────
alpha = 2.5         # coherent‐state amplitude
dim   = 40          # Fock‐space cutoff
Npts  = 200         # grid size for x and p

xvec = np.linspace(-5, 5, Npts)

# ── Build single‐mode coherent states ─────────────────────────────────────────
ket_p = coherent(dim, alpha)     # |+α⟩
ket_m = coherent(dim, -alpha)    # |−α⟩

# ── 2‑mode optical cat (analog of Cat₂) ───────────────────────────────────────
#    (|α,α⟩ + |−α,−α⟩)/√2
cat2 = (tensor(ket_p, ket_p) + tensor(ket_m, ket_m)).unit()

# ── 3‑mode optical cat (analog of Cat₃) ───────────────────────────────────────
#    (|α,α,α⟩ + |−α,−α,−α⟩)/√2
cat3 = (tensor(ket_p, ket_p, ket_p) + tensor(ket_m, ket_m, ket_m)).unit()

# ── Wigner for Cat₂, mode 0 ───────────────────────────────────────────────────
rho2_0 = cat2.ptrace(0)            # reduce to first mode
W2_0   = wigner(rho2_0, xvec, xvec)

plt.figure(figsize=(5,4))
plt.contourf(xvec, xvec, W2_0, 100, cmap="RdBu_r")
plt.title("Wigner of Cat₂ (mode 0)")
plt.xlabel("x")
plt.ylabel("p")
plt.colorbar(label="W(x,p)")
plt.tight_layout()


# ── Wigner for Cat₃, modes 0 & 1 ──────────────────────────────────────────────
#    tracing out mode 2 to get a 2‑mode joint Wigner
rho3_01 = cat3.ptrace([0,1])
W3_01   = wigner(rho3_01, xvec, xvec)

X, Y = np.meshgrid(xvec, xvec)
plt.figure(figsize=(5,4))
plt.contourf(X, Y, W3_01, 100, cmap="RdBu_r")
plt.title("Wigner of Cat₃ (modes 0 & 1)")
plt.xlabel("x₀")
plt.ylabel("x₁")
plt.axis("equal")
plt.colorbar(label="W(x₀,x₁)")
plt.tight_layout()

plt.show()
