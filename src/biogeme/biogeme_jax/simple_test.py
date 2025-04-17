import biogeme_jax.numpy as jnp
import pandas as pd
from icecream import ic
from biogeme_jax import vmap

data = pd.DataFrame({"income": [1, 2, 3], "age": [10, 20, 30], "choice": [0, 1, 0]})
data_jax = jnp.array(data.to_numpy())


def simple_function(parameters, one_row):
    return one_row[2]


vectorized_function = vmap(
    lambda parameters, row: simple_function(parameters, row), in_axes=(None, 0)
)
result = vectorized_function([], data_jax)
ic(result)
