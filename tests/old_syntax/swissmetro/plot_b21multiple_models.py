""".. _plot_b21multiple_models:

Assisted specification
======================

 Example of the estimation of several versions of the model using
 assisted specification algorithm. The catalog of specifications is
 defined in :ref:`plot_b21multiple_models_spec` . All specifications
 are estimated. Have a look at :ref:`plot_b22multiple_models` for an
 example where the number of specifications is too high to be
 enumerated.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 16:58:49 2023

"""
import biogeme.biogeme_logging as blog
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import loglikelihood_dimension
from biogeme.assisted import AssistedSpecification
from plot_b21multiple_models_spec import the_biogeme, PARETO_FILE_NAME

logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b21multipleModels')

# %%
# Creation of the object capturing the assisted specification algorithm.
# Its constructor takes three arguments:
#
#    - the biogeme object containing the specifications and the
#      database,
#    - an object defining the objectives to minimize. Here, we use the
#      opposite of the log lieklihood and the number of estimated
#      parameters.
#    - the name of the file where the estimated are saved, and
#      organized into a Pareto set.
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name=PARETO_FILE_NAME,
)

# %%
# The algorithm is run.
non_dominated_models = assisted_specification.run()

# %%
summary, description = compile_estimation_results(
    non_dominated_models, use_short_names=True
)
print(summary)

# %%
# Explanation of the short names of the model.
for k, v in description.items():
    if k != v:
        print(f'{k}: {v} AIC={summary.at["Akaike Information Criterion", k]}')
