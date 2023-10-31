"""File b21process_pareto.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:46:14 2023

The assisted specification algorithm generates a file containg the
pareto optimal specification. This script is designed to re-estimate
the Pareto optimal models

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
from b21multiple_models_spec import the_biogeme, PARETO_FILE_NAME

logger = blog.get_screen_logger(blog.INFO)
logger.info('Example b21process_pareto.py')

CSV_FILE = 'b21process_pareto.csv'
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
    try:
        _ = the_pareto_post.plot()
        plt.show()
    except OptimizationError as e:
        print(f'No plot available: {e}')
