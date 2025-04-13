import dynamiqs as dq
import jax.numpy as jnp
alpha=2.0
alpha2=2.0 *jnp.exp(1j*2*jnp.pi/3)
alpha3=2.0 *jnp.exp(1j*4*jnp.pi/3)
N=50

psi=1/jnp.sqrt(3)* (dq.coherent(N,alpha) + dq.coherent(N,alpha2) +dq.coherent(N,alpha3))
wigner_state=dq.wigner(psi)
dq.plot.wigner(psi)