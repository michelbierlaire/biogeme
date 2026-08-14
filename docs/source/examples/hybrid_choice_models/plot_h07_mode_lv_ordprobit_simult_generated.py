"""Simultaneous hybrid choice model with ordered-probit indicators
===================================================================


This example estimates a hybrid choice model where a latent variable,
interpreted as a car-centric attitude, influences the utilities of a
mode-choice model. The latent variable is described by a structural equation
and is measured through several attitudinal indicators using ordered-probit
measurement equations.

The model is estimated simultaneously: the measurement equations and the
choice model are combined into one likelihood function. Because the latent
variable is not observed, the likelihood is integrated by Monte Carlo
simulation.

The script is organized with ``# %%`` markers. Each marker denotes the start
of a new cell when the example is later converted into a notebook. New cells are
introduced whenever a new concept is presented, so that the generated notebook
can be read progressively as a pedagogical example. The comments therefore
describe both the statistical role of each block and the pedagogical message of
the corresponding notebook cell.

Tested with Biogeme 3.3.3.
Michel Bierlaire
Sat Jun 13 2026, 15:13:40

"""

from choice_latent_variables import (
    generate_availability,
    generate_utility_functions,
)
from number_of_draws import NUMBER_OF_DRAWS
from optima import Choice, read_data

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    Draws,
    MonteCarlo,
    MultipleProduct,
    MultipleSum,
    OrderedProbit,
    Variable,
    exp,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import (
    get_latex_estimated_parameters,
    get_latex_general_statistics,
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Load the Optima data set.
# -------------------------
# The data set contains the observed mode choices, explanatory variables, and
# attitudinal indicators used in the hybrid choice model.
database = read_data()


# %%
# Model parameters
# ----------------
# The parameters are organized according to their role in the model: thresholds
# for the Likert-scale indicators, standard deviations of measurement errors,
# coefficients of the measurement equations, and coefficients of the structural
# equation of the latent variable.
likert_delta_0_log = Beta('likert_delta_0_log', -0.86, None, None, 0)
likert_delta_0 = exp(likert_delta_0_log)
likert_delta_1_log = Beta('likert_delta_1_log', -0.43, None, None, 0)
likert_delta_1 = exp(likert_delta_1_log)
measurement_Envir01_sigma = 1.0
measurement_Envir02_sigma_log = Beta(
    'measurement_Envir02_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Envir02_sigma = exp(measurement_Envir02_sigma_log)
measurement_Envir06_sigma_log = Beta(
    'measurement_Envir06_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Envir06_sigma = exp(measurement_Envir06_sigma_log)
measurement_LifSty07_sigma_log = Beta(
    'measurement_LifSty07_sigma_log', 2.302585092994046, None, None, 0
)
measurement_LifSty07_sigma = exp(measurement_LifSty07_sigma_log)
measurement_Mobil03_sigma_log = Beta(
    'measurement_Mobil03_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Mobil03_sigma = exp(measurement_Mobil03_sigma_log)
measurement_Mobil05_sigma_log = Beta(
    'measurement_Mobil05_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Mobil05_sigma = exp(measurement_Mobil05_sigma_log)
measurement_Mobil08_sigma_log = Beta(
    'measurement_Mobil08_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Mobil08_sigma = exp(measurement_Mobil08_sigma_log)
measurement_Mobil09_sigma_log = Beta(
    'measurement_Mobil09_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Mobil09_sigma = exp(measurement_Mobil09_sigma_log)
measurement_Mobil10_sigma_log = Beta(
    'measurement_Mobil10_sigma_log', 2.302585092994046, None, None, 0
)
measurement_Mobil10_sigma = exp(measurement_Mobil10_sigma_log)
measurement_coefficient_car_centric_attitude_Envir01 = -1.0
measurement_coefficient_car_centric_attitude_Envir02 = Beta(
    'measurement_coefficient_car_centric_attitude_Envir02', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Envir06 = Beta(
    'measurement_coefficient_car_centric_attitude_Envir06', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_LifSty07 = Beta(
    'measurement_coefficient_car_centric_attitude_LifSty07', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Mobil03 = Beta(
    'measurement_coefficient_car_centric_attitude_Mobil03', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Mobil05 = Beta(
    'measurement_coefficient_car_centric_attitude_Mobil05', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Mobil08 = Beta(
    'measurement_coefficient_car_centric_attitude_Mobil08', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Mobil09 = Beta(
    'measurement_coefficient_car_centric_attitude_Mobil09', 0.0, None, None, 0
)
measurement_coefficient_car_centric_attitude_Mobil10 = Beta(
    'measurement_coefficient_car_centric_attitude_Mobil10', 0.0, None, None, 0
)
measurement_intercept_Envir01 = 0.0
measurement_intercept_Envir02 = Beta(
    'measurement_intercept_Envir02', 0.0, None, None, 0
)
measurement_intercept_Envir06 = Beta(
    'measurement_intercept_Envir06', 0.0, None, None, 0
)
measurement_intercept_LifSty07 = Beta(
    'measurement_intercept_LifSty07', 0.0, None, None, 0
)
measurement_intercept_Mobil03 = Beta(
    'measurement_intercept_Mobil03', 0.0, None, None, 0
)
measurement_intercept_Mobil05 = Beta(
    'measurement_intercept_Mobil05', 0.0, None, None, 0
)
measurement_intercept_Mobil08 = Beta(
    'measurement_intercept_Mobil08', 0.0, None, None, 0
)
measurement_intercept_Mobil09 = Beta(
    'measurement_intercept_Mobil09', 0.0, None, None, 0
)
measurement_intercept_Mobil10 = Beta(
    'measurement_intercept_Mobil10', 0.0, None, None, 0
)
struct_car_centric_attitude_car_oriented_parents = Beta(
    'struct_car_centric_attitude_car_oriented_parents', 0.0, None, None, 0
)
struct_car_centric_attitude_high_education = Beta(
    'struct_car_centric_attitude_high_education', 0.0, None, None, 0
)
struct_car_centric_attitude_intercept = Beta(
    'struct_car_centric_attitude_intercept', 0.0, None, None, 0
)
struct_car_centric_attitude_low_education = Beta(
    'struct_car_centric_attitude_low_education', 0.0, None, None, 0
)
struct_car_centric_attitude_sigma_log = Beta(
    'struct_car_centric_attitude_sigma_log', 2.302585092994046, None, None, 0
)
struct_car_centric_attitude_sigma = exp(struct_car_centric_attitude_sigma_log)
struct_car_centric_attitude_top_manager = Beta(
    'struct_car_centric_attitude_top_manager', 0.0, None, None, 0
)
struct_car_centric_attitude_used_to_go_to_school_by_car = Beta(
    'struct_car_centric_attitude_used_to_go_to_school_by_car', 0.0, None, None, 0
)

# %%
# Structural equation for the latent variable
# -------------------------------------------
# The latent variable is represented as a normal random variable. Its mean is a
# linear function of observed socio-economic characteristics, and its standard
# deviation is estimated from the data.
mu_car_centric_attitude = (
    struct_car_centric_attitude_intercept
    + struct_car_centric_attitude_top_manager * Variable('top_manager')
    + struct_car_centric_attitude_car_oriented_parents
    * Variable('car_oriented_parents')
    + struct_car_centric_attitude_high_education * Variable('high_education')
    + struct_car_centric_attitude_low_education * Variable('low_education')
    + struct_car_centric_attitude_used_to_go_to_school_by_car
    * Variable('used_to_go_to_school_by_car')
)
draw_car_centric_attitude = Draws(
    'struct_car_centric_attitude_draws', draw_type='NORMAL_MLHS_ANTI'
)
car_centric_attitude = (
    mu_car_centric_attitude
    + struct_car_centric_attitude_sigma * draw_car_centric_attitude
)

# %%
# Threshold system for the ordered-probit indicators
# --------------------------------------------------
# The attitudinal indicators are coded on a five-point Likert scale. The
# ordered-probit model uses four thresholds to separate the five ordered
# response categories. The logarithmic parameterization of the threshold
# increments guarantees the required ordering.
# Threshold system: Likert scale
likert_tau_1 = -(likert_delta_0 + likert_delta_1)
likert_tau_2 = -likert_delta_0
likert_tau_3 = likert_delta_0
likert_tau_4 = likert_delta_0 + likert_delta_1

# %%
# Measurement equations and ordered-probit likelihood terms
# ---------------------------------------------------------
# Each indicator is linked to the latent variable by a linear measurement
# equation. The probability of the observed ordinal answer is then computed
# with an ordered-probit model.
# Indicator: Envir01
mu_Envir01 = (
    measurement_intercept_Envir01
    + measurement_coefficient_car_centric_attitude_Envir01 * car_centric_attitude
)
y_Envir01 = Variable('Envir01')
term_Envir01 = OrderedProbit(
    eta=mu_Envir01 / measurement_Envir01_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Envir01_sigma,
        likert_tau_2 / measurement_Envir01_sigma,
        likert_tau_3 / measurement_Envir01_sigma,
        likert_tau_4 / measurement_Envir01_sigma,
    ],
    y=y_Envir01,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Envir02
mu_Envir02 = (
    measurement_intercept_Envir02
    + measurement_coefficient_car_centric_attitude_Envir02 * car_centric_attitude
)
y_Envir02 = Variable('Envir02')
term_Envir02 = OrderedProbit(
    eta=mu_Envir02 / measurement_Envir02_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Envir02_sigma,
        likert_tau_2 / measurement_Envir02_sigma,
        likert_tau_3 / measurement_Envir02_sigma,
        likert_tau_4 / measurement_Envir02_sigma,
    ],
    y=y_Envir02,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Envir06
mu_Envir06 = (
    measurement_intercept_Envir06
    + measurement_coefficient_car_centric_attitude_Envir06 * car_centric_attitude
)
y_Envir06 = Variable('Envir06')
term_Envir06 = OrderedProbit(
    eta=mu_Envir06 / measurement_Envir06_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Envir06_sigma,
        likert_tau_2 / measurement_Envir06_sigma,
        likert_tau_3 / measurement_Envir06_sigma,
        likert_tau_4 / measurement_Envir06_sigma,
    ],
    y=y_Envir06,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Mobil03
mu_Mobil03 = (
    measurement_intercept_Mobil03
    + measurement_coefficient_car_centric_attitude_Mobil03 * car_centric_attitude
)
y_Mobil03 = Variable('Mobil03')
term_Mobil03 = OrderedProbit(
    eta=mu_Mobil03 / measurement_Mobil03_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Mobil03_sigma,
        likert_tau_2 / measurement_Mobil03_sigma,
        likert_tau_3 / measurement_Mobil03_sigma,
        likert_tau_4 / measurement_Mobil03_sigma,
    ],
    y=y_Mobil03,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Mobil05
mu_Mobil05 = (
    measurement_intercept_Mobil05
    + measurement_coefficient_car_centric_attitude_Mobil05 * car_centric_attitude
)
y_Mobil05 = Variable('Mobil05')
term_Mobil05 = OrderedProbit(
    eta=mu_Mobil05 / measurement_Mobil05_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Mobil05_sigma,
        likert_tau_2 / measurement_Mobil05_sigma,
        likert_tau_3 / measurement_Mobil05_sigma,
        likert_tau_4 / measurement_Mobil05_sigma,
    ],
    y=y_Mobil05,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Mobil08
mu_Mobil08 = (
    measurement_intercept_Mobil08
    + measurement_coefficient_car_centric_attitude_Mobil08 * car_centric_attitude
)
y_Mobil08 = Variable('Mobil08')
term_Mobil08 = OrderedProbit(
    eta=mu_Mobil08 / measurement_Mobil08_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Mobil08_sigma,
        likert_tau_2 / measurement_Mobil08_sigma,
        likert_tau_3 / measurement_Mobil08_sigma,
        likert_tau_4 / measurement_Mobil08_sigma,
    ],
    y=y_Mobil08,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Mobil09
mu_Mobil09 = (
    measurement_intercept_Mobil09
    + measurement_coefficient_car_centric_attitude_Mobil09 * car_centric_attitude
)
y_Mobil09 = Variable('Mobil09')
term_Mobil09 = OrderedProbit(
    eta=mu_Mobil09 / measurement_Mobil09_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Mobil09_sigma,
        likert_tau_2 / measurement_Mobil09_sigma,
        likert_tau_3 / measurement_Mobil09_sigma,
        likert_tau_4 / measurement_Mobil09_sigma,
    ],
    y=y_Mobil09,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: Mobil10
mu_Mobil10 = (
    measurement_intercept_Mobil10
    + measurement_coefficient_car_centric_attitude_Mobil10 * car_centric_attitude
)
y_Mobil10 = Variable('Mobil10')
term_Mobil10 = OrderedProbit(
    eta=mu_Mobil10 / measurement_Mobil10_sigma,
    cutpoints=[
        likert_tau_1 / measurement_Mobil10_sigma,
        likert_tau_2 / measurement_Mobil10_sigma,
        likert_tau_3 / measurement_Mobil10_sigma,
        likert_tau_4 / measurement_Mobil10_sigma,
    ],
    y=y_Mobil10,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# Indicator: LifSty07
mu_LifSty07 = (
    measurement_intercept_LifSty07
    + measurement_coefficient_car_centric_attitude_LifSty07 * car_centric_attitude
)
y_LifSty07 = Variable('LifSty07')
term_LifSty07 = OrderedProbit(
    eta=mu_LifSty07 / measurement_LifSty07_sigma,
    cutpoints=[
        likert_tau_1 / measurement_LifSty07_sigma,
        likert_tau_2 / measurement_LifSty07_sigma,
        likert_tau_3 / measurement_LifSty07_sigma,
        likert_tau_4 / measurement_LifSty07_sigma,
    ],
    y=y_LifSty07,
    categories=[1, 2, 3, 4, 5],
    neutral_labels=[6, -1],
)

# %%
# Conditional measurement likelihood
# ----------------------------------
# Conditional on a realization of the latent variable, the indicator likelihood
# is the product of the ordered-probit probabilities. The corresponding sum of
# logarithms is also constructed for later combination with the choice model.
conditional_measurement_likelihood = MultipleProduct(
    [
        term_Envir01,
        term_Envir02,
        term_Envir06,
        term_LifSty07,
        term_Mobil03,
        term_Mobil05,
        term_Mobil08,
        term_Mobil09,
        term_Mobil10,
    ]
)
conditional_log_likelihood = MultipleSum(
    [
        log(term)
        for term in [
            term_Envir01,
            term_Envir02,
            term_Envir06,
            term_LifSty07,
            term_Mobil03,
            term_Mobil05,
            term_Mobil08,
            term_Mobil09,
            term_Mobil10,
        ]
    ]
)
integrated_measurement_likelihood = MonteCarlo(conditional_measurement_likelihood)


# %%
# Generate the choice utilities.
# ------------------------------
# The latent variable is passed to the utility specification so that the
# car-centric attitude can directly influence the mode-choice probabilities.
latent_expressions = {'car_centric_attitude': car_centric_attitude}
utilities = generate_utility_functions(latent_expressions)

# %%
# Conditional choice likelihood.
# ------------------------------
# Given a realization of the latent variable, the probability of the observed
# mode choice is computed using the logit model.
availability = generate_availability()
conditional_choice_likelihood = logit(utilities, availability, Choice)

# %%
# Joint conditional likelihood.
# -----------------------------
# The measurement component and the choice component are combined conditional
# on the latent variable. This expression is the joint contribution of one
# observation before integrating out the unobserved latent variable.
combined_conditional_likelihood = (
    conditional_measurement_likelihood * conditional_choice_likelihood
)

# %%
# Integrate over the latent variable.
# -----------------------------------
# The latent variable is unobserved, so the joint conditional likelihood must
# be integrated over its distribution. The integral is approximated by Monte
# Carlo simulation, and the logarithm of the simulated probability defines the
# log-likelihood used for estimation.
integrated_likelihood = MonteCarlo(combined_conditional_likelihood)
log_likelihood = log(integrated_likelihood)


# %%
# Estimate the simultaneous model with Biogeme.
# ---------------------------------------------
# The dictionary groups parameters in the output tables according to their
# statistical role. Biogeme then estimates the model, or reloads existing
# results from the YAML file when available.
group_of_parmeters = {
    'Structural equation': [
        'struct_car_centric_attitude_intercept',
        'struct_car_centric_attitude_top_manager',
        'struct_car_centric_attitude_car_oriented_parents',
        'struct_car_centric_attitude_high_education',
        'struct_car_centric_attitude_low_education',
        'struct_car_centric_attitude_used_to_go_to_school_by_car',
        'struct_car_centric_attitude_sigma_log',
    ],
    'Measurement equation: Envir02': [
        'measurement_intercept_Envir02',
        'measurement_coefficient_car_centric_attitude_Envir02',
        'measurement_Envir02_sigma_log',
    ],
    'Measurement equation: Envir06': [
        'measurement_intercept_Envir06',
        'measurement_coefficient_car_centric_attitude_Envir06',
        'measurement_Envir06_sigma_log',
    ],
    'Measurement equation: Mobil03': [
        'measurement_intercept_Mobil03',
        'measurement_coefficient_car_centric_attitude_Mobil03',
        'measurement_Mobil03_sigma_log',
    ],
    'Measurement equation: Mobil05': [
        'measurement_intercept_Mobil05',
        'measurement_coefficient_car_centric_attitude_Mobil05',
        'measurement_Mobil05_sigma_log',
    ],
    'Measurement equation: Mobil08': [
        'measurement_intercept_Mobil08',
        'measurement_coefficient_car_centric_attitude_Mobil08',
        'measurement_Mobil08_sigma_log',
    ],
    'Measurement equation: Mobil09': [
        'measurement_intercept_Mobil09',
        'measurement_coefficient_car_centric_attitude_Mobil09',
        'measurement_Mobil09_sigma_log',
    ],
    'Measurement equation: Mobil10': [
        'measurement_intercept_Mobil10',
        'measurement_coefficient_car_centric_attitude_Mobil10',
        'measurement_Mobil10_sigma_log',
    ],
    'Measurement equation: LifSty07': [
        'measurement_intercept_LifSty07',
        'measurement_coefficient_car_centric_attitude_LifSty07',
        'measurement_LifSty07_sigma_log',
    ],
    'Thresholds': ['likert_delta_0_log', 'likert_delta_1_log'],
}

biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    calculating_second_derivatives='never',
    max_iterations=5_000,
    group_of_parameters=group_of_parmeters,
)
biogeme.model_name = 'plot_h07_mode_lv_ordprobit_simult_generated'

yaml_file_name = f'saved_results/{biogeme.model_name}.yaml'
results = biogeme.estimate_or_load(yaml_file_name=yaml_file_name)

# %%
# Report the estimation results.
# ------------------------------
# The compact summary gives the main diagnostics. The estimated parameters and
# general statistics are also exported as pandas and LaTeX tables, which is
# useful when preparing reports, slides, or scientific documents.
print(results.short_summary())
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
    group_of_parameters=group_of_parmeters,
)
for group_name, pandas_table in pandas_results.items():
    print(group_name if group_name else 'Estimated parameters')
    print(pandas_table)

general_statistics = get_latex_general_statistics(estimation_results=results)
print(general_statistics)

estimated_parameters = get_latex_estimated_parameters(
    estimation_results=results,
    group_of_parameters=group_of_parmeters,
)
for group_name, latex_table in estimated_parameters.items():
    print(group_name if group_name else 'Estimated parameters')
    print(latex_table)
