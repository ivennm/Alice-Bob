import dynamiqs as dq
import jax.numpy as jnp
alpha=2.0
N=50

psi=1/jnp.sqrt(2)* (dq.coherent(N,alpha) + dq.coherent(N,alpha))
wigner_state=dq.wigner(psi)
dq.plot.wigner(psi)