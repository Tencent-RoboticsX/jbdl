import jax.numpy as jnp
from jax import lax


def calc_rank_jc(flag_contact, nf):
    # int(np.sum( [1 for item in flag_contact if item != 0]) * nf
    init_carry = 0.0

    def f(carry, flag):
        one_hot = jnp.heaviside(flag, 0.0)
        new_carry = carry + one_hot
        return new_carry, one_hot

    rank_jc, one_hot = lax.scan(f, init_carry, flag_contact)
    rank_jc = rank_jc * nf
    return rank_jc, one_hot
