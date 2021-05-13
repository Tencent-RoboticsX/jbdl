import chex
import jax.numpy as jnp
from jax import lax

def calc_rankJc(flag_contact, nf):
    # int(np.sum( [1 for item in flag_contact if item != 0]) * nf
    init_carry = 0.0
    def f(carry, flag):
        one_hot = jnp.heaviside(flag, 0.0)
        new_carry = carry + one_hot
        return new_carry, one_hot 
    rankJc, one_hot = lax.scan(f, init_carry, flag_contact)
    rankJc = rankJc * nf
    return rankJc, one_hot


if __name__ == "__main__":
    flag_contact = jnp.array([1.0, 1.0, 2.0, 0.0])
    nf = 2.0

    print(calc_rankJc(flag_contact, nf))
    

