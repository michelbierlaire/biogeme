"""

Re-estimation of best models
============================

After running the assisted specification algorithm for the 432
specifications in :ref:`everything_spec_section`, we use post processing to
re-estimate all Pareto optimal models, and display some information
about the algorithm.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Thu Jul 20 17:15:37 2023

"""
try:
    import matplotlib.pyplot as plt

    can_plot = True
except ModuleNotFoundError:
    can_plot = False
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.assisted import ParetoPostProcessing

from everything_spec import model_catalog, database

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b08selected_specification')

PARETO_FILE_NAME = 'saved_results/b07everything_assisted.pareto'

# %%
# Create the biogeme object from the catalog.
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b09post_processing'

# %%
# Create the post processing object.
post_processing = ParetoPostProcessing(
    biogeme_object=the_biogeme, pareto_file_name=PARETO_FILE_NAME
)

# %%
# Re-estimate the models.
all_results = post_processing.reestimate(recycle=True)

# %%
# We retieve the first estimation results for illustration.
spec, results = next(iter(all_results.items()))

# %%
print(spec)

# %%
print(results.short_summary())

# %%
results.getEstimatedParameters()


# %%
# The following plot illustrates all models that have been
# estimated.  Each dot corresponds to a model. The x-coordinate
# corresponds to the Akaike Information Criterion (AIC). The
# y-coordinate corresponds to the Bayesian Information Criterion
# (BIC). Note that there is a third objective that does not appear on
# this picture: the number of parameters. If the shape of the dot is a
# circle, it means that it corresponds to a Pareto optimal model. If
# the shape is a cross, it means that the model has been Pareto
# optimal at some point during the algorithm and later removed as a
# new model dominating it has been found. If the shape is a start, it
# means that the model has been deemed invalid.
if can_plot:
    _ = post_processing.plot(
        label_x='Nbr of parameters',
        label_y='Negative log likelihood',
        objective_x=1,
        objective_y=0,
    )
