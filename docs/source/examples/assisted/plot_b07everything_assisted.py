"""

Combine many specifications: assisted specification algorithm
=============================================================

We combine many specifications, defined in :ref:`everything_spec_section`.
This leads to a total of 432 specifications.
The algorithm implemented in the AssistedSpecification object is used to
investigate some of these specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:59:08
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.assisted import AssistedSpecification
from biogeme.biogeme import BIOGEME
from biogeme.multiobjectives import loglikelihood_dimension
from biogeme.results_processing import EstimationResults, compile_estimation_results
from everything_spec import database, model_catalog

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07everything_assisted')

PARETO_FILE_NAME = 'b07everything_assisted.pareto'


# %%
# Function verifying that the estimation results are valid.
def validity(results: EstimationResults) -> tuple[bool, str | None]:
    """Function verifying that the estimation results are valid.

    The results are not valid if any of the time or cost coefficient is non-negative.
    """
    for parameter_index, parameter_name in enumerate(results.beta_names):
        parameter_value = results.beta_values[parameter_index]
        if 'TIME' in parameter_name and parameter_value >= 0:
            return False, f'{parameter_name} = {parameter_value}'
        if 'COST' in parameter_name and parameter_value >= 0:
            return False, f'{parameter_name} = {parameter_value}'
    return True, None


# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, model_catalog, generate_html=False, generate_yaml=False)
the_biogeme.model_name = 'b07everything'

# %%
# Estimate the parameters using assisted specification algorithm.
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
    validity=validity,
)

non_dominated_models = assisted_specification.run()

# %%
print(f'A total of {len(non_dominated_models)} models have been generated.')

# %%
compiled_results, specs = compile_estimation_results(
    non_dominated_models, use_short_names=True
)

# %%
display(compiled_results)

# %%
# Glossary
for short_name, spec in specs.items():
    print(f'{short_name}\t{spec}')
