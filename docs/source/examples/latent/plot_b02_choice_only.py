import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.models import loglogit
from biogeme.results_processing import (
    EstimationResults,
)

from choice_model import v
from optima import (
    Choice,
    read_data,
)

logger = blog.get_screen_logger(level=blog.INFO)

database = read_data()

# %%
# We integrate over omega using numerical integration
log_likelihood = loglogit(v, None, Choice)


# %%
# Create the Biogeme object
print('Create the biogeme object')
the_biogeme = BIOGEME(database, log_likelihood)
the_biogeme.model_name = 'b02_choice_only'


print('--- Estimate ---')
results: EstimationResults = the_biogeme.estimate()
