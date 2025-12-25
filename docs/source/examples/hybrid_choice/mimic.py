"""

MIMIC model
===========

Construction of the MIMIC (latent-variable) component of the hybrid choice model.

This module defines a helper function that builds and returns an
:class:`OrderedMimic` object, fully configured according to a given
:class:`Config`.

The MIMIC model includes two latent variables:

- a car-centric attitude latent variable, and
- an environmental attitude latent variable.

For each latent variable, the module specifies:

- the structural equation (explanatory variables),
- the associated Likert-type indicators, and
- the normalization used for identification.

The resulting MIMIC model is used by higher-level code to construct
measurement equations and to combine them with the choice model, under
either maximum likelihood or Bayesian estimation.

This file is intentionally limited to model specification; it contains no
data handling or estimation logic.

Michel Bierlaire
Thu Dec 25 2025, 08:15:26
"""

from biogeme.latent_variables import (
    EstimationMode,
    LatentVariable,
    Normalization,
    OrderedMimic,
    StructuralEquation,
)

from config import Config
from latent_variables import (
    car_explanatory_variables,
    car_likert_indicators,
    car_name,
    env_name,
    environment_explanatory_variables,
    environment_likert_indicators,
)
from likert_indicators import likert_indicators, likert_types


def generate_mimic_model(config: Config):
    """Generate and return a configured MIMIC model.

    The estimation mode (Bayesian or maximum likelihood) is selected based on
    ``config.estimation``. The function then:

    1. Creates an :class:`OrderedMimic` container with the appropriate
       estimation mode and Likert indicator definitions.
    2. Defines the car-centric and environmental latent variables, including
       their structural equations, indicators, and normalizations.
    3. Registers both latent variables with the MIMIC model.

    :param config: Global configuration object controlling the estimation
        paradigm.
    :return: A fully specified :class:`OrderedMimic` instance.
    """
    bayesian_estimation = config.estimation == "bayes"
    estimation_mode = (
        EstimationMode.BAYESIAN
        if bayesian_estimation
        else EstimationMode.MAXIMUM_LIKELIHOOD
    )

    mimic_model = OrderedMimic(
        estimation_mode=estimation_mode,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
    )

    car_lv = LatentVariable(
        name=car_name,
        structural_equation=StructuralEquation(
            name=car_name,
            explanatory_variables=car_explanatory_variables,
        ),
        indicators=car_likert_indicators,
        normalization=Normalization(indicator='Envir01', coefficient=-1),
    )
    mimic_model.add_latent_variable(car_lv)

    env_lv = LatentVariable(
        name=env_name,
        structural_equation=StructuralEquation(
            name=env_name,
            explanatory_variables=environment_explanatory_variables,
        ),
        indicators=environment_likert_indicators,
        normalization=Normalization(indicator='Envir02', coefficient=1),
    )
    mimic_model.add_latent_variable(env_lv)
    return mimic_model
