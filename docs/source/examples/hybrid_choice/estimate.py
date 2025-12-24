from IPython.core.display_functions import display
from biogeme.bayesian_estimation import (
    get_pandas_estimated_parameters as get_pandas_bayesian_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Expression, MonteCarlo, log
from biogeme.latent_variables import EstimationMode
from biogeme.models import logit, loglogit
from biogeme.results_processing import (
    get_pandas_estimated_parameters as get_pandas_ml_estimated_parameters,
)

from choice_model import generate_choice_model
from config import (
    Config,
)
from latent_variables import car_name, env_name
from mimic import generate_mimic_model
from optima import Choice, read_data
from read_or_estimate import read_or_estimate


def generate_expression(config: Config) -> Expression:
    utilities = generate_choice_model(config=config)

    # If there are no latent variables, return only the choice model.
    if config.latent_variables == "zero":
        return (
            loglogit(utilities, None, Choice)
            if config.estimation == "bayes"
            else log(MonteCarlo(logit(utilities, None, Choice)))
        )

    mimic = generate_mimic_model(config=config)
    car_centric_attitude = mimic.get_latent_variable(name=car_name)
    environmental_attitude = mimic.get_latent_variable(name=env_name)

    # Build the "inside" of the likelihood once
    if config.estimation == "bayes":
        inner = mimic.log_measurement_equations()
        if config.choice_model == "yes":
            inner = loglogit(utilities, None, Choice) + inner
        return inner

    # ML
    inner = mimic.measurement_equations()
    if config.choice_model == "yes":
        inner = logit(utilities, None, Choice) * inner
    return log(MonteCarlo(inner))


def estimate_model(config: Config) -> None:
    the_expression = generate_expression(config=config)
    estimation_mode = (
        EstimationMode.BAYESIAN
        if config.estimation == "bayes"
        else EstimationMode.MAXIMUM_LIKELIHOOD
    )
    # %%
    # Read the data
    database = read_data()

    # %%
    # Create the Biogeme object
    the_biogeme = BIOGEME(
        database,
        the_expression,
        warmup=config.number_of_bayesian_draws_per_chain,
        bayesian_draws=config.number_of_bayesian_draws_per_chain,
        chains=4,
        number_of_draws=config.number_of_monte_carlo_draws,
        calculating_second_derivatives='never',
        numerically_safe=True,
        max_iterations=5000,
    )
    the_biogeme.model_name = config.name

    # %%
    # If estimation results are saved on file, we read them to speed up the process.
    # If not, we estimate the parameters.
    results = read_or_estimate(
        the_biogeme=the_biogeme,
        estimation_mode=estimation_mode,
        directory='saved_results',
    )

    # %%
    print(results.short_summary())

    # %%
    # Get the results in a pandas table
    pandas_results = (
        get_pandas_ml_estimated_parameters(
            estimation_results=results,
        )
        if estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD
        else get_pandas_bayesian_estimated_parameters(estimation_results=results)
    )
    display(pandas_results)
