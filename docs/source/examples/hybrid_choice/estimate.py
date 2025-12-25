"""

Model estimation
================

Estimation pipeline for discrete and hybrid choice models.

This module defines the high-level logic used to construct and estimate
choice models and hybrid choice (MIMIC) models using Biogeme. The behavior
of the pipeline is fully controlled by a :class:`Config` object, allowing
the same code to be reused across multiple experimental configurations.

Depending on the configuration, the module can:

- estimate a standard discrete choice model without latent variables,
- estimate a hybrid choice model with two latent variables and measurement
  equations,
- perform either maximum likelihood or Bayesian estimation,
- optionally combine the choice model likelihood with the measurement
  likelihood.

The module is intentionally declarative: model structure is assembled here,
while data handling, estimation, and result post-processing are delegated
to specialized helper modules.

Michel Bierlaire
Thu Dec 25 2025, 08:13:25
"""

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
from mimic import generate_mimic_model
from optima import Choice, read_data
from read_or_estimate import read_or_estimate


def generate_expression(config: Config) -> Expression:
    """Generate the likelihood expression to be estimated.

    This function constructs the Biogeme expression corresponding to the
    selected model configuration.

    - If ``config.latent_variables == 'zero'``, only the discrete choice
      model likelihood is returned.
    - Otherwise, the likelihood from the latent-variable measurement
      equations is constructed and optionally combined with the choice
      model likelihood.

    The expression is wrapped differently depending on the estimation
    paradigm:

    - Bayesian estimation uses log-likelihood expressions (``loglogit`` and
      log measurement equations).
    - Maximum likelihood estimation uses Monte Carlo integration when
      required.

    :param config: Configuration object controlling model structure and
        estimation mode.
    :return: A Biogeme expression representing the full likelihood to be
        estimated.
    """
    utilities = generate_choice_model(config=config)

    # If there are no latent variables, return only the choice model.
    if config.latent_variables == "zero":
        return (
            loglogit(utilities, None, Choice)
            if config.estimation == "bayes"
            else log(MonteCarlo(logit(utilities, None, Choice)))
        )

    mimic = generate_mimic_model(config=config)

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
    """Estimate the model specified by the given configuration.

    This function:

    1. Builds the likelihood expression using :func:`generate_expression`.
    2. Creates and configures a :class:`BIOGEME` object.
    3. Either reads existing estimation results from disk or runs a new
       estimation.
    4. Prints a short textual summary and displays the estimated parameters
       in a pandas table.

    :param config: Configuration object defining the model specification,
        estimation method, and numerical settings.
    """
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
