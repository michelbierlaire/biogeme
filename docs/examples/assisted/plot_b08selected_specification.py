"""

One model among many
====================

We consider the model with 432 specifications defined in
:ref:`everything_spec_section`. We select one specification and estimate it.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:46:56 2023

"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from everything_spec import model_catalog, database, av

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# The code characterizing the specification should be copied from the
# .pareto file generated by the algorithm, or from one of the
# glossaries illustrated in earlier examples.
SPEC_ID = (
    'ASC:GA-LUGGAGE;'
    'B_COST_gen_altspec:generic;'
    'B_TIME:FIRST;'
    'B_TIME_gen_altspec:generic;'
    'model_catalog:logit;'
    'train_tt_catalog:power'
)

# %% The biogeme object for the selected model can be obtained from
# the spec_id, and used as usual.
the_biogeme = bio.BIOGEME.from_configuration(
    config_id=SPEC_ID,
    expression=model_catalog,
    database=database,
)
the_biogeme.modelName = 'my_favorite_model'

# %%
# Calculate of the null log-likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = results.get_estimated_parameters()

# %%
pandas_results
