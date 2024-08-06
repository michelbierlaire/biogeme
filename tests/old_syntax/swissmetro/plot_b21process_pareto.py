""".. _plot_b21process_pareto:

Re-estimate the Pareto optimal models
=====================================

The assisted specification algorithm generates a file containg the
pareto optimal specification. This script is designed to re-estimate
the Pareto optimal models. The catalog of specifications is defined in
:ref:`plot_b21multiple_models_spec` .

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:46:14 2023

"""
import biogeme.biogeme_logging as blog

try:
    import matplotlib.pyplot as plt

    can_plot = True
except ModuleNotFoundError:
    can_plot = False
from biogeme_optimization.exceptions import OptimizationError
from biogeme.assisted import ParetoPostProcessing
from biogeme.results import compileEstimationResults
from plot_b21multiple_models_spec import the_biogeme

PARETO_FILE_NAME = 'saved_results/b21multiple_models.pareto'

logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b21process_pareto.py')

CSV_FILE = 'b21process_pareto.csv'
SEP_CSV = ','

# %%
# The constructor of the Pareto post processing object takes two arguments:
#
#    - the biogeme object,
#    - the name of the file where the algorithm has stored the
#      estimated models.
the_pareto_post = ParetoPostProcessing(
    biogeme_object=the_biogeme,
    pareto_file_name=PARETO_FILE_NAME,
)

# %%
the_pareto_post.log_statistics()

# %%
# Complete re-estimation of the best models, including the calculation
# of the statistics.
all_results = the_pareto_post.reestimate(recycle=False)

# %%
summary, description = compileEstimationResults(all_results, use_short_names=True)
print(summary)

# %%
print(f'Summary table available in {CSV_FILE}')
summary.to_csv(CSV_FILE, sep=SEP_CSV)

# %%
# Explanation of the short names of the models.
with open(CSV_FILE, 'a', encoding='utf-8') as f:
    print('\n\n', file=f)
    for k, v in description.items():
        if k != v:
            print(f'{k}: {v}')
            print(f'{k}{SEP_CSV}{v}', file=f)

# %%
# The following plot illustrates all models that have been estimated.
# Each dot corresponds to a model. The x-coordinate corresponds to the
# negative log-likelihood. The y-coordinate corresponds to the number
# of parameters. If the shape of the dot is a circle, it means that it
# corresponds to a Pareto optimal model. If the shape is a cross, it
# means that the model has been Pareto optimal at some point during
# the algorithm and later removed as a new model dominating it has
# been found.
if can_plot:
    try:
        _ = the_pareto_post.plot(
            label_x='Negative loglikelihood', label_y='Number of parameters'
        )
        plt.show()
    except OptimizationError as e:
        print(f'No plot available: {e}')
