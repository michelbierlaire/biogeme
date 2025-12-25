"""

Choice model
============

Choice model specification with optional latent-variable interactions.

This module defines the utility functions for a discrete choice model that can
optionally incorporate latent variables estimated in a hybrid choice (MIMIC)
framework.

The behavior of the choice model is controlled by the `Config` object, in
particular by the attribute `config.latent_variables`, which can take two
values:

- ``"zero"``: no latent variables are used; the choice model reduces to a
  standard discrete choice model with observed attributes only.
- ``"two"``: two latent variables (car-centric and environmental attitudes)
  enter the utilities through interaction terms.

Latent variables affect the model in two ways:

1. Multiplicative effects on time coefficients, where the base time sensitivity
   is scaled by an exponential function of the latent variable(s).
2. Additive alternative-specific constants that shift utilities depending on
   the latent attitudes.

This module only constructs the systematic utility functions. The likelihood
construction (logit, Monte Carlo integration, Bayesian or maximum-likelihood
wrapping) is handled elsewhere in the codebase.

Michel Bierlaire
Thu Dec 25 2025, 08:07:33
"""

from biogeme.expressions import Beta, Expression, Numeric, exp

from config import Config
from latent_variables import car_name, env_name
from mimic import generate_mimic_model
from optima import (
    CostCarCHF,
    MarginalCostPT,
    PurpHWH,
    TimeCar_hour,
    TimePT_hour,
    WaitingTimePT,
    distance_km,
)


def generate_choice_model(config: Config) -> dict[int, Expression]:
    """Generate the choice model utilities.

    The behavior depends on the number of latent variables requested:

    - ``config.latent_variables == 'zero'``: no latent variables enter the choice model.
    - ``config.latent_variables == 'two'``: both car-centric and environmental latent variables enter.

    Note: this function returns only the utilities. The estimation / likelihood wrapping
    is handled elsewhere.
    """
    if config.latent_variables not in {"zero", "two"}:
        raise ValueError(
            f"config.latent_variables must be 'zero' or 'two', got {config.latent_variables!r}."
        )
    include_latent_variables = config.latent_variables == "two"

    # Latent variables can be: zero (none) or two (car-centric + environmental).
    car_centric_attitude = None
    environmental_attitude = None

    if include_latent_variables:
        mimic = generate_mimic_model(config=config)
        car_centric_attitude = mimic.get_latent_variable(name=car_name)
        environmental_attitude = mimic.get_latent_variable(name=env_name)

    # %%
    # Choice model
    work_trip = PurpHWH == 1
    other_trip_purposes = PurpHWH != 1

    # Choice model: parameters
    choice_beta_cost = Beta('choice_beta_cost', 0, None, 0, 0)

    choice_asc_car = Beta('choice_asc_car', 0.0, None, None, 0)

    choice_asc_pt = Beta('choice_asc_pt', 0, None, None, 0)

    choice_beta_dist_work = Beta('choice_beta_dist_work', 0, None, 0, 0)
    choice_beta_dist_other_purposes = Beta(
        'choice_beta_dist_other_purposes', 0, None, 0, 0
    )
    choice_beta_dist = (
        choice_beta_dist_work * work_trip
        + choice_beta_dist_other_purposes * other_trip_purposes
    )

    # Time coefficients with optional LV interactions
    choice_beta_time_car_ref = Beta('choice_beta_time_car_ref', 0, None, 0, 0)
    choice_beta_time_car = choice_beta_time_car_ref

    if include_latent_variables:
        beta_time_car_lambda_environment = Beta(
            'beta_time_car_lambda_environment', -1, None, 0, 0
        )
        choice_beta_time_car *= exp(
            beta_time_car_lambda_environment
            * environmental_attitude.structural_equation_jax
        )

        beta_time_car_lambda_car_centric = Beta(
            'beta_time_car_lambda_car_centric', -1, None, 0, 0
        )
        choice_beta_time_car *= exp(
            beta_time_car_lambda_car_centric
            * car_centric_attitude.structural_equation_jax
        )

    choice_beta_time_pt_ref = Beta('choice_beta_time_pt_ref', 0, None, 0, 0)
    choice_beta_time_pt = choice_beta_time_pt_ref

    if include_latent_variables:
        beta_time_pt_lambda_environment = Beta(
            'beta_time_pt_lambda_environment', -1, None, 0, 0
        )
        choice_beta_time_pt *= exp(
            beta_time_pt_lambda_environment
            * environmental_attitude.structural_equation_jax
        )

        beta_time_pt_lambda_car_centric = Beta(
            'beta_time_pt_lambda_car_centric', -1, None, 0, 0
        )
        choice_beta_time_pt *= exp(
            beta_time_pt_lambda_car_centric
            * car_centric_attitude.structural_equation_jax
        )

    choice_beta_waiting_time_work = Beta('choice_beta_waiting_time_work', 0, None, 0, 1)
    choice_beta_waiting_time_other_purposes = Beta(
        'choice_beta_waiting_time_other_purposes', 0, None, 0, 1
    )
    choice_beta_waiting_time = (
        choice_beta_waiting_time_work * work_trip
        + choice_beta_waiting_time_other_purposes * other_trip_purposes
    )

    log_scale_choice_model = Beta('log_scale_choice_model', 0, None, None, 1)
    scale_choice_model = exp(log_scale_choice_model)

    # %%
    # Alternative specific constants (kept as-is; they enter only if the LV exists)
    choice_car_centric_car_cte = Beta('choice_car_centric_car_cte', 1, None, None, 0)
    choice_car_centric_pt_cte = Beta('choice_car_centric_pt_cte', 0, None, None, 0)
    choice_environment_car_cte = Beta('choice_environment_car_cte', 0, None, None, 0)
    choice_environment_pt_cte = Beta('choice_environment_pt_cte', 1, None, None, 0)

    # %%
    # Definition of utility functions:
    v_public_transport = scale_choice_model * (
        choice_asc_pt
        + choice_beta_time_pt * TimePT_hour
        + choice_beta_waiting_time * WaitingTimePT / 60
        + choice_beta_cost * MarginalCostPT
        + (
            choice_car_centric_pt_cte * car_centric_attitude.structural_equation_jax
            if car_centric_attitude is not None
            else Numeric(0)
        )
        + (
            choice_environment_pt_cte * environmental_attitude.structural_equation_jax
            if environmental_attitude is not None
            else Numeric(0)
        )
    )

    v_car = scale_choice_model * (
        choice_asc_car
        + choice_beta_time_car * TimeCar_hour
        + choice_beta_cost * CostCarCHF
        + (
            choice_car_centric_car_cte * car_centric_attitude.structural_equation_jax
            if car_centric_attitude is not None
            else Numeric(0)
        )
        + (
            choice_environment_car_cte * environmental_attitude.structural_equation_jax
            if environmental_attitude is not None
            else Numeric(0)
        )
    )

    v_slow_modes = scale_choice_model * (choice_beta_dist * distance_km)

    # %%
    # Associate utility functions with the numbering of alternatives
    v = {0: v_public_transport, 1: v_car, 2: v_slow_modes}

    return v
