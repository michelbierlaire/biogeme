""".. _plot_b22process_pareto:

Re-estimate the Pareto optimal models
=====================================

The assisted specification algorithm generates a file containg the
pareto optimal specification. This script is designed to re-estimate
the Pareto optimal models. The catalog of specifications is defined in
:ref:`plot_b22multiple_models_spec` .

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:25:41 2023


"""
try:
    import matplotlib.pyplot as plt

    can_plot = True
except ModuleNotFoundError:
    can_plot = False
from biogeme.assisted import ParetoPostProcessing
from biogeme.results import compileEstimationResults
from plot_b22multiple_models_spec import the_biogeme

PARETO_FILE_NAME = 'saved_results/b22multiple_models.pareto'

CSV_FILE = 'b22process_pareto.csv'
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
all_results = the_pareto_post.reestimate(recycle=True)

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
    _ = the_pareto_post.plot(label_x='AIC', label_y='BIC')
    plt.show()

# %%
# It is possible to plot two different objectives: AIC and number of parameters.
if can_plot:
    _ = the_pareto_post.plot(
        objective_x=0, objective_y=2, label_x='AIC', label_y='Number of parameters'
    )
    plt.show()

# %%
# It is possible to plot two different objectives: BIC and number of parameters.
if can_plot:
    _ = the_pareto_post.plot(
        objective_x=1, objective_y=2, label_x='BIC', label_y='Number of parameters'
    )
    plt.show()
