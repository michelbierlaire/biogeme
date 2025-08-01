"""

Model specification for the simple tutorial
===========================================

Example extracted from Ben-Akiva and Lerman (1985)

Michel Bierlaire, EPFL
Sun Jun 15 2025, 07:19:46
"""

from biogeme.expressions import Beta
from tutorial_data import auto_time, transit_time

asc_car = Beta('asc_car', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
utility_car = asc_car + b_time * auto_time
utility_transit = b_time * transit_time

# %%
# Next, we need to associate the utility function with the ID of the alternative. It is necessary to interpret
# correctly the value of the variable `choice`. We use a Python dictionary to do that.
utilities = {0: utility_car, 1: utility_transit}
