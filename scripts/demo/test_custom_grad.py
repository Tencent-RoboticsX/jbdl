import jax.numpy as jnp
from jax import custom_vjp
from jax import jit
from jax import jacrev
import time


@custom_vjp
def f(x):
  print("called f!")
  return jnp.sin(x)

@jit
def f_fwd(x):
  print("called f_fwd!")
  return f(x), jnp.cos(x)

@jit
def f_bwd(cos_x, y_bar):
  print("called f_bwd!")
  return (cos_x * y_bar,)

f.defvjp(f_fwd, f_bwd)
start = time.time()
print(jacrev(f)(3.0))
print("duration", time.time()-start)
print("============")
start = time.time()
print(jacrev(f)(3.0))
print("duration", time.time()-start)
print("============")
start = time.time()
print(jacrev(f)(3.0))
print("duration", time.time()-start)
print("============")
start = time.time()
print(jacrev(f)(3.0))
print("duration", time.time()-start)
print("============")
start = time.time()
print(jacrev(f)(3.0))
print("duration", time.time()-start)
print("============")


