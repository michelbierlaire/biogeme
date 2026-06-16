"""

19. Calculation of individual level parameters
==============================================

Calculation of the individual level parameters for the model defined
in :ref:`plot_b05_normal_mixture`.

Michel Bierlaire, EPFL
Mon Nov 17 2025, 18:59:56
"""

import sys

from IPython.core.display_functions import display

from biogeme.bayesian_estimation import BayesianResults

# %%
# Retrieve estimation results
result_file_name = 'saved_results/b05_normal_mixture.nc'
try:
    the_estimation_results = BayesianResults.from_netcdf(filename=result_file_name)
except FileNotFoundError:
    print(f'File {result_file_name} does not exist')
    sys.exit()

# %%
# With Bayesian estimation, the individual-level parameters are automatically generated.
# We simply retrieve them from estimation results.
sim = the_estimation_results.posterior_mean_by_observation(var_name='b_time_rnd')
display(sim)
