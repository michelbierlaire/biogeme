"""File b22process_pareto.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:25:41 2023

The assisted specification algorithm generates a file containg the
pareto optimal specification. This script is designed to re-estimate
the Pareto optimal models

"""
try: 
    import matplotlib.pyplot as plt
    can_plot = True
except ModuleNotFoundError:
    can_plot = False
from biogeme.assisted import ParetoPostProcessing
from biogeme.results import compileEstimationResults
from b22multiple_models_spec import the_biogeme, PARETO_FILE_NAME

CSV_FILE = 'b22process_pareto.csv'
SEP_CSV = ','

the_pareto_post = ParetoPostProcessing(
    biogeme_object=the_biogeme,
    pareto_file_name=PARETO_FILE_NAME,
)
the_pareto_post.log_statistics()

all_results = the_pareto_post.reestimate(recycle=True)

summary, description = compileEstimationResults(all_results, use_short_names=True)

print(summary)
print(f'Summary table available in {CSV_FILE}')
summary.to_csv(CSV_FILE, sep=SEP_CSV)
with open(CSV_FILE, 'a', encoding='utf-8') as f:
    print('\n\n', file=f)
    for k, v in description.items():
        if k != v:
            print(f'{k}: {v}')
            print(f'{k}{SEP_CSV}{v}', file=f)


if can_plot:
    _ = the_pareto_post.plot()
    plt.show()
