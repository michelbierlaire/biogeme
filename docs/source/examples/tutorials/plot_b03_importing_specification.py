"""

Importing model specification
=============================

The Python language allows to organize your model specification in different files.

Michel Bierlaire, EPFL
Thu May 16 13:24:56 2024

"""

from biogeme.biogeme import BIOGEME
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# After having imported the necessary packages, the database and the choice variable are imported
# from the file `tutorial_data.py`
from tutorial_data import biogeme_database, choice

# %%
# Similarly, the specification of the utility functions is imported from the file `tutorial_model.py`.
from tutorial_model import utilities

# %%
# Once the ingredients have been imported, the rest of the script is exactly the same.
log_choice_probability = loglogit(utilities, None, choice)
biogeme_object = BIOGEME(biogeme_database, log_choice_probability)
biogeme_object.model_name = 'first_model'
biogeme_object.calculate_null_loglikelihood(avail={0: 1, 1: 1})
results = biogeme_object.estimate()
print(results.short_summary())
print(get_pandas_estimated_parameters(estimation_results=results))
