"""File b21multiple_models.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 16:58:49 2023

 Example of the estimation of several versions of the model using
 assisted specification algorithm Three alternatives: Train, Car and

"""
import biogeme.logging as blog
from biogeme.results import compileEstimationResults, loglikelihood_dimension
from biogeme.assisted import AssistedSpecification
from b21multiple_models_spec import the_biogeme, PARETO_FILE_NAME

logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b21multipleModels')

assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)
print('Algorithm info: ')
for m in assisted_specification.statistics():
    print(m)
non_dominated_models = assisted_specification.run()

summary, description = compileEstimationResults(
    non_dominated_models, use_short_names=True
)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')
