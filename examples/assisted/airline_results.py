"""File airline_results.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 16:26:35 2023

Assisted specification for the airline case study. Generate results
from the pareto file.

"""

import matplotlib.pyplot as plt
import biogeme.logging as blog
from biogeme.assisted import AssistedSpecification
from biogeme.results import compileEstimationResults, loglikelihood_dimension

from airline_spec import the_biogeme

screen_logger = blog.get_screen_logger(blog.DEBUG)
screen_logger.info('airline example: assisted specification')

PARETO_FILE_NAME = 'airline.pareto'

assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
    max_neighborhood=20,
)
print(assisted_specification.statistics())

all_results = assisted_specification.reestimate(recycle=True)

summary, description = compileEstimationResults(all_results, use_short_names=True)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')

ax = assisted_specification.plot()
plt.show()
