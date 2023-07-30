"""File airline_results.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 16:26:35 2023

Assisted specification for the airline case study. Generate results
from the pareto file.

"""
try: 
    import matplotlib.pyplot as plt
    can_plot = True
except ModuleNotFoundError:
    can_plot = False

import biogeme.logging as blog
from biogeme.assisted import AssistedSpecification, ParetoPostProcessing
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import loglikelihood_dimension

from airline_spec import the_biogeme

screen_logger = blog.get_screen_logger(blog.INFO)
screen_logger.info('airline example: assisted specification')

PARETO_FILE_NAME = 'airline.pareto'

assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)

post_processing = ParetoPostProcessing(
    biogeme_object=the_biogeme,
    pareto_file_name=PARETO_FILE_NAME,
)
post_processing.log_statistics()

all_results = post_processing.reestimate(recycle=True)

summary, description = compile_estimation_results(all_results, use_short_names=True)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')

if can_plot:
    ax = post_processing.plot()
    plt.show()
