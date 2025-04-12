import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, tensor, wigner

# ── Parameters ────────────────────────────────────────────────────────────────
alpha = 2.5
dim   = 30
Npts  = 200

xvec = np.linspace(-5, 5, Npts)

# ── Build single‐mode coherent states ─────────────────────────────────────────
ket_p = coherent(dim, alpha)
ket_m = coherent(dim, -alpha)

# ── Build 3‑mode optical cat state: (|α,α,α⟩ + |−α,−α,−α⟩)/√2 ──────────────
cat3 = (tensor(ket_p, ket_p, ket_p) + tensor(ket_m, ket_m, ket_m)).unit()

# ── Reduce to modes 0 & 1 and compute their joint Wigner function ───────────
rho3_01 = cat3.ptrace([0, 1])
W3_01   = wigner(rho3_01, xvec, xvec)

# ── Plot ──────────────────────────────────────────────────────────────────────
X, Y = np.meshgrid(xvec, xvec)
plt.figure(figsize=(5,4))
plt.contourf(X, Y, W3_01, 100, cmap="RdBu_r")
plt.title("Wigner Function of 3‑Mode Cat State (modes 0 & 1)")
plt.xlabel("x₀")
plt.ylabel("x₁")
plt.axis("equal")
plt.colorbar(label="W(x₀,x₁)")
plt.tight_layout()
plt.show()
