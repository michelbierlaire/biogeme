"""File airline.py

:author: Michel Bierlaire, EPFL
:date: Fri Mar 31 10:45:43 2023

Assisted specification for the airline cases tudy
"""


import biogeme.logging as blog
from biogeme.assisted import AssistedSpecification
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import loglikelihood_dimension

from airline_spec import the_biogeme

screen_logger = blog.get_screen_logger(blog.INFO)
screen_logger.info('airline example: assisted specification')

PARETO_FILE_NAME = 'airline.pareto'

nbr = the_biogeme.loglike.number_of_multiple_expressions()
print('The number of possible specifications is {nbr}')

screen_logger.info('Prepare assisted specification')

assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)

screen_logger.info('Run assisted specification')

non_dominated_models = assisted_specification.run()

summary, description = compile_estimation_results(
    non_dominated_models, use_short_names=True
)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')
