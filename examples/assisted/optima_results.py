"""File optima_results.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 16:40:23 2023

Assisted specification for the Optima case study. Generation of the
results from the Pareto file

"""
import matplotlib.pyplot as plt
import biogeme.logging as blog
from biogeme.assisted import AssistedSpecification
from biogeme.results import compileEstimationResults, loglikelihood_dimension
from optima_spec import the_biogeme

PARETO_FILE_NAME = 'optima.pareto'

screen_logger = blog.get_screen_logger(blog.DEBUG)
screen_logger.info('optima example: assisted specification')

nbr = the_biogeme.loglike.number_of_multiple_expressions()
if nbr is None:
    print('The number of possible specifications is too large to be calculated')

screen_logger.info('Prepare assisted specification')

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
