"""

Combine many specifications: assisted specification algorithm
=============================================================

We combine many specifications, defined in :ref:`everything_spec_section`. 
This leads to a total of 432 specifications.
The algorithm implemented in the AssistedSpecification object is used to
investigate some of these specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:02:20 2023

"""
from typing import Optional
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.assisted import AssistedSpecification
from biogeme.multiobjectives import loglikelihood_dimension
from everything_spec import model_catalog, database
from biogeme.results import bioResults, compile_estimation_results

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07everything_assisted')

PARETO_FILE_NAME = 'b07everything_assisted.pareto'


# %%
# Function verifying that the estimation results are valid.
def validity(results: bioResults) -> tuple[bool, Optional[str]]:
    """Function verifying that the estimation results are valid.

    The results are not valid if any of the time or cost coefficient is non negative.
    """
    for beta in results.data.betas:
        if 'TIME' in beta.name and beta.value >= 0:
            return False, f'{beta.name} = {beta.value}'
        if 'COST' in beta.name and beta.value >= 0:
            return False, f'{beta.name} = {beta.value}'
    return True, None


# %%
# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b07everything'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

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
compiled_results

# %%
# Glossary
for short_name, spec in specs.items():
    print(f'{short_name}\t{spec}')
