"""
Utilities for Likert-type indicators used in Biogeme latent-variable models.

This module defines small helper data structures to work with Likert-scale
survey items in Biogeme measurement equations.

Two concepts are separated:

- :class:`LikertType` describes a category of indicators of the same type (categories, symmetry, and labels).
- :class:`LikertIndicator` describes an *item* (statement / variable name),
  its shared type metadata, and the measurement model used to connect the
  latent response to the observed indicator.

Threshold construction (cut-points) and other expression-building logic are
implemented in builder modules (for example ``likert_builders.py``). This file
contains only the specification objects and naming helpers.

Michel Bierlaire
Tue Dec 23 2025, 15:08:48
"""

from dataclasses import dataclass
from enum import Enum


class MeasurementModel(str, Enum):
    """Measurement-model family used for one indicator.

    :cvar GAUSSIAN:
        Treat the observed indicator as continuous and use a Gaussian
        measurement likelihood.
    :cvar ORDERED_PROBIT:
        Treat the observed indicator as ordinal and use an ordered-probit
        likelihood.
    :cvar ORDERED_LOGIT:
        Treat the observed indicator as ordinal and use an ordered-logit
        likelihood.
    """

    GAUSSIAN = "gaussian"
    ORDERED_PROBIT = "ordered_probit"
    ORDERED_LOGIT = "ordered_logit"


@dataclass
class LikertType:
    """Describe a Likert scale specification.

    A :class:`LikertType` represents the definition of a Likert scale shared by
    one or several indicators (items).

    :param type_name:
        Short label used as prefix for threshold-parameter names.
    :param symmetric:
        If True, thresholds are symmetric around 0. If False, thresholds are
        only constrained to be strictly increasing.
    :param categories:
        Ordered list of distinct category codes used in the data (for example
        ``[-2, -1, 0, 1, 2]``).
    :param neutral_labels:
        Category codes considered neutral for this scale (for example ``[0]``).
        This metadata is not used to build thresholds, but can be used by
        downstream code.
    """

    type_name: str
    symmetric: bool
    categories: list[int]
    neutral_labels: list[int]


@dataclass
class LikertIndicator:
    """Represent a Likert indicator and provide helpers for measurement parameters.

    The class does not store the scale definition itself (categories, thresholds,
    etc.). Those are described by :class:`LikertType`. This class focuses on
    consistent parameter naming for the measurement equation.

    :param name:
        Short identifier of the indicator, used to construct parameter names.
    :param statement:
        Text of the statement that respondents evaluate on the Likert scale.
    :param type_name:
        Optional indicator-type label (for example to implement threshold sharing
        policies by type).
    :param measurement_model:
        Statistical model used for the measurement equation. The default is
        :class:`MeasurementModel.ORDERED_PROBIT`, which preserves the current
        behavior for existing specifications.
    """

    name: str
    statement: str
    type_name: str
    measurement_model: MeasurementModel = MeasurementModel.ORDERED_PROBIT

    @property
    def intercept_parameter_name(self) -> str:
        """
        Return the name of the intercept parameter for this indicator.

        :return:
            The parameter name used for the measurement intercept.
        """
        return f'measurement_intercept_{self.name}'

    def get_lv_coefficient_parameter_name(self, latent_variable_name: str) -> str:
        """
        Build the name of the coefficient linking a latent variable to this indicator.

        :param latent_variable_name:
            Name of the latent variable appearing in the measurement equation.
        :return:
            The parameter name used for the corresponding coefficient.
        """
        return f'measurement_coefficient_{latent_variable_name}_{self.name}'
