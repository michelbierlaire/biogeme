"""File b07everything_assisted.py

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:02:20 2023

Investigate various specifications:
- 3 models
    - logit
    - nested logit with two nests: public and private transportation
    - nested logit with two nests existing and future modes
- 3 functional form for the travel time variables
    - linear specification,
    - Box-Cox transform,
    - power series,
- 2 specification for the cost coefficients:
    - generic
    - alternative specific
- 2 specification for the travel time coefficients:
    - generic
    - alternative specific
- 4 segmentations for the constants:
    - not segmented
    - segmented by GA (yearly subscription to public transport)
    - segmented by luggage
    - segmented both by GA and luggage
-  3 segmentations for the time coefficients:
    - not segmented
    - segmented with first class
    - segmented with trip purpose

This leads to a total of 432 specifications.
The algorithm implemented in the AssistedSpecification object is used to
investigate some of these specifications.

"""
import biogeme.logging as blog
import biogeme.biogeme as bio
from biogeme.assisted import AssistedSpecification
from biogeme.multiobjectives import loglikelihood_dimension
from everything_spec import model_catalog, database
from results_analysis import report

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07everything_assisted')

PARETO_FILE_NAME = 'b07everything_assisted.pareto'

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b07everything'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# Estimate the parameters
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)

non_dominated_models = assisted_specification.run()

report(non_dominated_models)
