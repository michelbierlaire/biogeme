"""File b08selected_specification.py

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:46:56 2023

We consider the model with 432 specifications:
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
After running the assisted specification algorithm, we select one
specification and estimate it.

"""
import biogeme.logging as blog
import biogeme.biogeme as bio
from everything_spec import model_catalog, database, av

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07everything_assisted')

SPEC_ID = (
    'ASC:GA-LUGGAGE;'
    'B_COST_gen_altspec:generic;'
    'B_TIME:FIRST;'
    'B_TIME_gen_altspec:generic;'
    'model_catalog:logit;'
    'train_tt_catalog:power'
)

# The biogeme object can be obtained from the spec_id, and used as usual.
the_biogeme = bio.BIOGEME.from_configuration(
    config_id=SPEC_ID,
    expression=model_catalog,
    database=database,
)
the_biogeme.modelName = 'my_favorite_model'

the_biogeme.calculateNullLoglikelihood(av)

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)
