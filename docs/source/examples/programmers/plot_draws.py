"""

biogeme.draws
=============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 06:41:25
"""

# %%
import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from biogeme.draws import (
    description_of_native_draws,
    get_antithetic,
    get_halton_draws,
    get_latin_hypercube_draws,
    get_normal_wichura_draws,
    get_uniform,
    native_random_number_generators,
)
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())

# %%
# We set the seed so that the outcome of random operations is always the same.
np.random.seed(90267)

# %%
# Uniform draws
# -------------

# %%
# Uniform [0,1]. The output is transformed into a data frame just for the display.
draws = get_uniform(sample_size=3, number_of_draws=10, symmetric=False)
display(pd.DataFrame(draws))

# %%
draws = get_uniform(sample_size=3, number_of_draws=10, symmetric=True)
display(pd.DataFrame(draws))

# %%
# LatinHypercube: the Modified Latin Hypercube Sampling (MLHS, Hess et al., 2006)
# provides U[0,1] draws from a perturbed grid, designed for
# Monte-Carlo integration.

# %%
latin_hypercube = get_latin_hypercube_draws(sample_size=3, number_of_draws=10)
display(pd.DataFrame(latin_hypercube))

# %%
# The same method can be used to generate draws from U[-1,1]

# %%
latin_hypercube = get_latin_hypercube_draws(
    sample_size=5, number_of_draws=10, symmetric=True
)
display(pd.DataFrame(latin_hypercube))

# %%
# The user can provide her own series of U[0,1] draws.
my_unif = np.random.uniform(size=30)
display(pd.DataFrame(my_unif))

# %%
latin_hypercube = get_latin_hypercube_draws(
    sample_size=3, number_of_draws=10, symmetric=False, uniform_numbers=my_unif
)
display(pd.DataFrame(latin_hypercube))

# %%
# The uniform draws can also be arranged in a two-dimension array
my_unif = get_uniform(sample_size=3, number_of_draws=10)
display(pd.DataFrame(my_unif))

# %%
latin_hypercube = get_latin_hypercube_draws(
    sample_size=3, number_of_draws=10, uniform_numbers=my_unif
)
display(pd.DataFrame(latin_hypercube))

# %%
# Halton draws
# ------------

# %%
# One Halton sequence.
halton = get_halton_draws(sample_size=2, number_of_draws=10, base=3)
display(pd.DataFrame(halton))

# %%
# Several Halton sequences.
halton = get_halton_draws(sample_size=3, number_of_draws=10)
display(pd.DataFrame(halton))

# %%
# Shuffled Halton sequences.
halton = get_halton_draws(sample_size=3, number_of_draws=10, shuffled=True)
display(pd.DataFrame(halton))

# %%
# The above sequences were generated using the default base: 2. It is
# possible to generate sequences using different prime numbers.
halton = get_halton_draws(sample_size=1, number_of_draws=10, base=3)
display(pd.DataFrame(halton))

# %%
# It is also possible to skip the first items of the sequence. This is
# desirable in the context of Monte-Carlo integration.
halton = get_halton_draws(sample_size=1, number_of_draws=10, base=3, skip=10)
display(pd.DataFrame(halton))

# %%
# Antithetic draws
# ----------------

# %%
# Antithetic draws can be generated from any function generating uniform draws.
draws = get_antithetic(get_uniform, sample_size=3, number_of_draws=10)
display(pd.DataFrame(draws))

# %%
# Antithetic MLHS
draws = get_antithetic(get_latin_hypercube_draws, sample_size=3, number_of_draws=10)
display(pd.DataFrame(draws))

# %%
# Antithetic Halton.
draws = get_antithetic(get_halton_draws, sample_size=1, number_of_draws=10)
display(pd.DataFrame(draws))


# %%
# As antithetic Halton draws may be correlated, it is a good idea to
# skip the first draws.
def uniform_halton(sample_size: int, number_of_draws: int) -> np.ndarray:
    """Function generating uniform draws for the antithetic draws"""
    return get_halton_draws(number_of_draws, sample_size, skip=100)


# %%
draws = get_antithetic(uniform_halton, sample_size=3, number_of_draws=10)
display(pd.DataFrame(draws))

# %%
# Normal draws
# ------------

# %%
# Generate pseudo-random numbers from a normal distribution N(0,1)
# using the Algorithm AS241 Appl. Statist. (1988) Vol. 37, No. 3 by
# Wichura
draws = get_normal_wichura_draws(sample_size=3, number_of_draws=10)
display(pd.DataFrame(draws))

# %%
# The antithetic version actually generates half of the draws and
# complete them with their antithetic version
draws = get_normal_wichura_draws(sample_size=3, number_of_draws=10, antithetic=True)
display(pd.DataFrame(draws))

# %%
# The user can provide her own series of U[0,1] draws. In this
# example, we use the MLHS procedure to generate these draws. Note
# that, if the antithetic version is used, only half of the requested
# draws must be provided.
my_unif = get_latin_hypercube_draws(sample_size=3, number_of_draws=5)
display(pd.DataFrame(my_unif))

# %%
draws = get_normal_wichura_draws(
    sample_size=3, number_of_draws=10, uniform_numbers=my_unif, antithetic=True
)
display(pd.DataFrame(draws))

# %%
# The same with Halton draws.
my_unif = get_halton_draws(sample_size=2, number_of_draws=5, base=3, skip=10)
display(pd.DataFrame(my_unif))

# %%
draws = get_normal_wichura_draws(
    number_of_draws=10, sample_size=2, uniform_numbers=my_unif, antithetic=True
)
display(pd.DataFrame(draws))

# %%
# Biogeme provides a list of native draws
the_description = description_of_native_draws()
for name, description in the_description.items():
    print(f'{name}: {description}')

# %%
draw_type = 'NORMAL_HALTON2'
the_native_draws = native_random_number_generators
the_generator = the_native_draws[draw_type].generator
draws = the_generator(
    sample_size=2,
    number_of_draws=10,
)
print(f'Draws of type {draw_type}')
display(pd.DataFrame(draws))
