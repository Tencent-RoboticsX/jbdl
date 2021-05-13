import jax
import jax.numpy as jnp
from jax import lax
import os
from jbdl.rbdl.utils import ModelWrapper

a = jnp.array([1, 0, 1, 0])
b = jnp.array([2, 3, 0, 6])

def f(carry, x):
    x_a, x_b, x_c = x
    new_x = x_a + x_b + x_c
    return carry, new_x



CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
MODEL_DATA_PATH = os.path.join(SCRIPTS_PATH, "model_data") 
mdlw = ModelWrapper()
mdlw.load(os.path.join(MODEL_DATA_PATH, 'whole_max_v1.json'))
model = mdlw.model

flag_contact = jnp.array([0, 1, 1, 0])
idcontact = jnp.array(model["idcontact"])
contactpoint = jnp.array(model["contactpoint"])

print(flag_contact.shape)
print(idcontact.shape)
print(contactpoint.shape)

print(lax.scan(f, None,(flag_contact, idcontact, contactpoint)))