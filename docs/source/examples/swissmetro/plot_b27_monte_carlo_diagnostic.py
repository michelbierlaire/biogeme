"""
.. _plot_b27_monte_carlo_diagnostic:

27. Post-estimation Monte Carlo draw-stability diagnostic
=========================================================

This example estimates a small mixed logit model and then runs the
post-estimation Monte Carlo draw-stability diagnostic.  The diagnostic keeps
the estimated parameters fixed, evaluates the objective and its gradient with
fresh draw designs, and writes a separate YAML checkpoint and Markdown report.

The estimation is performed only when no saved result is available.  The
diagnostic can therefore be interrupted and resumed without re-estimating the
model.

Michel Bierlaire, EPFL
"""

import shutil
from pathlib import Path

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Draws, MonteCarlo, log
from biogeme.models import logit

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example plot_b27_monte_carlo_diagnostic.py')

# %%
# Parameters of the mixed logit model.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_cost = Beta('b_cost', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_time_s = Beta('b_time_s', 1, None, None, 0)

# The random coefficient is integrated by Monte Carlo simulation.
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'NORMAL')

# %%
# Utilities and availability conditions.
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

utilities = {1: v_train, 2: v_swissmetro, 3: v_car}
availability = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
conditional_probability = logit(utilities, availability, CHOICE)
log_probability = log(MonteCarlo(conditional_probability))

# %%
# The diagnostic is deliberately small enough for a documentation example.
# Its default draw schedule is replaced here by 0.5R, R, and 2R, with one
# fresh design per level and a five-minute time budget.
the_biogeme = BIOGEME(
    database,
    log_probability,
    user_notes=(
        'Post-estimation Monte Carlo draw-stability diagnostic for a mixed '
        'logit model using the Swissmetro data.'
    ),
    number_of_draws=2_000,
    seed=1223,
    calculating_second_derivatives='never',
    save_iterations=False,
    generate_html=False,
    monte_carlo_diagnostic_auto=False,
    monte_carlo_diagnostic_draw_factors='0.5,1.0,2.0',
    monte_carlo_diagnostic_replications=1,
    monte_carlo_diagnostic_time_budget=300,
    monte_carlo_diagnostic_max_draws=4_000,
)
the_biogeme.model_name = 'b27_monte_carlo'

# %%
# Load the archived result when available.  Otherwise, estimate once and save
# the normal estimation result in the current directory.  This makes the
# example work both from a clean checkout and from the JED archive.
saved_result = Path('saved_results') / f'{the_biogeme.model_name}.yaml'
estimation_file = (
    saved_result
    if saved_result.is_file()
    else Path(f'{the_biogeme.model_name}.yaml')
)
results = the_biogeme.estimate_or_load(yaml_file_name=str(estimation_file))

# JED archives root-level diagnostic files in ``saved_results``.  Restore an
# archived checkpoint before running so an interrupted diagnostic resumes in a
# fresh isolated working directory instead of starting from its first level.
saved_diagnostic = Path('saved_results') / 'b27_monte_carlo_diagnostic.yaml'
diagnostic_checkpoint = Path('b27_monte_carlo_diagnostic.yaml')
if saved_diagnostic.is_file() and not diagnostic_checkpoint.is_file():
    shutil.copy2(saved_diagnostic, diagnostic_checkpoint)

# %%
# Run the post-estimation diagnostic.  It never re-estimates the model and
# writes b27_monte_carlo_diagnostic.yaml and b27_monte_carlo_diagnostic.md.
diagnostic = the_biogeme.check_monte_carlo_stability(
    estimation_results=results,
    output_directory='.',
    basename='b27',
)

print(f'Execution status: {diagnostic.execution_status}')
print(f'Diagnostic conclusion: {diagnostic.diagnostic_conclusion}')
print(f'Recommendation: {diagnostic.recommendation}')
print(f'Raw diagnostic results: {diagnostic.yaml_file}')
print(f'Diagnostic report: {diagnostic.markdown_file}')
