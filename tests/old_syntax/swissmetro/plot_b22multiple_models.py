""".. _plot_b22multiple_models:

Assisted specification
======================

Example of the estimation of several versions of the model using
assisted specification algorithm. The catalog of specifications is
defined in :ref:`plot_b22multiple_models_spec` . Compared to
:ref:`plot_b21multiple_models`, the number fo specifications exceeds
the maximum limit, so a heuristic is applied.  See `Bierlaire and
Ortelli, 2023
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_
for a detailed description of the use of the assisted specification
algorithm.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:05:40 2023


"""

import biogeme.biogeme_logging as blog
from biogeme.results import compile_estimation_results
from biogeme.multiobjectives import AIC_BIC_dimension
from biogeme.assisted import AssistedSpecification
from plot_b22multiple_models_spec import the_biogeme, PARETO_FILE_NAME


logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b22multiple_models')

# %%
nbr = the_biogeme.loglike.number_of_multiple_expressions()
if nbr is None:
    print('There are too many possible specifications to be enumerated')
else:
    print(f'There are {nbr} possible specifications')


# %%
# Creation of the object capturing the assisted specification algorithm.
# Its constructor takes three arguments:
#
#    - the biogeme object containing the specifications and the
#      database,
#    - an object defining the objectives to minimize. Here, we use
#      three objectives: AIC, BIC and number of parameters.
#    - the name of the file where the estimated are saved, and
#      organized into a Pareto set.
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=AIC_BIC_dimension,
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
        print(f'{k}: {v}')
