"""File optima_run.py

:author: Michel Bierlaire, EPFL
:date: Mon Jul 10 17:43:56 2023

Assisted specification for the Optima case study
"""

import biogeme.logging as blog
from biogeme.assisted import AssistedSpecification
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import loglikelihood_dimension
from optima_spec import the_biogeme

PARETO_FILE_NAME = 'optima.pareto'

screen_logger = blog.get_screen_logger(blog.INFO)
screen_logger.info('optima example: assisted specification')

nbr = the_biogeme.loglike.number_of_multiple_expressions()
if nbr is None:
    print('The number of possible specifications is too large to be calculated')

screen_logger.info('Prepare assisted specification')

assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)

non_dominated_models = assisted_specification.run()

summary, description = compile_estimation_results(
    non_dominated_models, use_short_names=True
)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')
