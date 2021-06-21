from jbdl.rbdl.model.rigid_body_inertia import init_Ic_by_cholesky
import jax.numpy as jnp
l = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print(init_Ic_by_cholesky(l))