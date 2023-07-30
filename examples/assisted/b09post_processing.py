"""File b09post_processing.py

:author: Michel Bierlaire, EPFL
:date: Thu Jul 20 17:15:37 2023

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

After running the assisted specification algorithm, we use post
processing to re-estimate all Pareto optimal models, and display some
information about the algorithm

"""
try: 
    import matplotlib.pyplot as plt
    can_plot = True
except ModuleNotFoundError:
    can_plot = False
import biogeme.logging as blog
import biogeme.biogeme as bio
from biogeme.assisted import ParetoPostProcessing

from everything_spec import model_catalog, database

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b08selected_specification')

PARETO_FILE_NAME = 'b07everything_assisted.pareto'

the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b09post_processing'

post_processing = ParetoPostProcessing(
    biogeme_object=the_biogeme, pareto_file_name=PARETO_FILE_NAME
)

post_processing.reestimate(recycle=True)

post_processing.log_statistics()

if can_plot:
    _ = post_processing.plot()
    plt.show()
