"""File b22multiple_models.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:05:40 2023

 Example of the estimation of several versions of the model. In this
case, the number fo specifications exceed the maximum limit, so a heuristic is applied.
It may actually end up enumerating all possiblities.
"""

import biogeme.biogeme_logging as blog
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import AIC_BIC_dimension
from biogeme.assisted import AssistedSpecification
from b22multiple_models_spec import the_biogeme, PARETO_FILE_NAME


logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b22multiple_models')


nbr = the_biogeme.loglike.number_of_multiple_expressions()
if nbr is None:
    print('There are too many possible specifications to be enumerated')
else:
    print(f'There are {nbr} possible specifications')
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=AIC_BIC_dimension,
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
