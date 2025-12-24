"""High-level wrapper for hybrid choice models (latent variables + Likert measurement).

This module provides :class:`OrderedMimic`, a convenience wrapper to assemble a
hybrid choice model consisting of:

- one or more latent variables (structural equations + normalization + indicator sets),
- a set of Likert indicators (measurement intercepts, measurement scales, loadings),
- a set of Likert types (categories and ordered thresholds).

It then builds the joint ordered-probit measurement likelihood using the
JAX-compatible builders in :mod:`.measurement_equations`.

Defaults depend on the estimation mode:

- Maximum likelihood (``EstimationMode.MAXIMUM_LIKELIHOOD``):
  uses ``draw_type='NORMAL_MLHS_ANTI'`` and a log-sigma parameterization for
  positive scale parameters.
- Bayesian (``EstimationMode.BAYESIAN``):
  uses ``draw_type='Normal'`` and a strictly positive sigma parameterization
  enforced through bounds.

Users can override these defaults by providing ``draw_type`` and/or
``sigma_factory`` at construction time, and can override ``draw_type`` again
when calling :meth:`OrderedMimic.measurement_equations` or
:meth:`OrderedMimic.log_measurement_equations`.

Michel Bierlaire
Sun Dec 14 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from biogeme.expressions import Expression

from .latent_variables import LatentVariable
from .likert_indicators import LikertIndicator, LikertType
from .measurement_equations import (
    log_measurement_equations_jax,
    measurement_equations_jax,
)
from .positive_parameter_factory import (
    PositiveParameterFactory,
    SigmaFactory,
    make_positive_parameter_factory,
    make_sigma_factory,
)

logger = logging.getLogger(__name__)


class EstimationMode(str, Enum):
    """Estimation mode controlling default draw types and positivity constraints."""

    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN = "bayesian"


# ---------------------------------------------------------------------
# Threshold sharing policy enum
# ---------------------------------------------------------------------


@dataclass
class OrderedMimic:
    """High-level wrapper for hybrid choice models with ordered-probit measurement.

    The class registers Likert indicators and Likert types, injects default
    factories when missing, and allows registering latent variables. It then
    builds the joint ordered-probit measurement likelihood (or its log) for all
    registered indicators.

    Defaults depend on :attr:`estimation_mode`:

    - Maximum likelihood: ``draw_type='NORMAL_MLHS_ANTI'`` and a log-sigma
      parameterization for scale parameters.
    - Bayesian: ``draw_type='Normal'`` and a strictly positive sigma
      parameterization (positivity enforced via bounds).

    :param estimation_mode:
        Estimation mode controlling defaults (maximum likelihood vs Bayesian).
    :param likert_indicators:
        List of Likert indicator specifications used by the measurement model.
    :param likert_types:
        List of Likert type specifications defining categories and thresholds.
    :param draw_type:
        Draw type used by latent-variable error terms. If ``None``, a default is
        selected based on ``estimation_mode``.
    :param sigma_factory:
        Factory used to generate strictly positive scale parameters. If ``None``,
        a default is selected based on ``estimation_mode``.
    """

    estimation_mode: EstimationMode
    likert_indicators: list[LikertIndicator]
    likert_types: list[LikertType]

    draw_type: str | None = None
    sigma_factory: SigmaFactory | None = None

    _latent_variables: dict[str, LatentVariable] = field(
        default_factory=dict, init=False
    )
    _likert_by_name: dict[str, LikertIndicator] = field(
        default_factory=dict, init=False
    )
    _likert_type_by_name: dict[str, LikertType] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        # Defaults only if user did not provide overrides.
        if self.draw_type is None:
            self.draw_type = (
                "NORMAL_MLHS_ANTI"
                if self.estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD
                else "Normal"
            )

        if self.sigma_factory is None:
            # ML -> log sigma; Bayes -> sigma (positive constraint handled via bounds)
            self.sigma_factory = make_sigma_factory(
                use_log=(self.estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD)
            )

        # Default positive-parameter factory used by indicators to build ordered thresholds.
        self._positive_factory: PositiveParameterFactory = (
            make_positive_parameter_factory(
                use_log=self.estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD
            )
        )
        self._register_likert_indicators()
        self._register_likert_types()

    def _register_likert_indicators(self) -> None:
        """Register Likert indicators used by the model.

        The method injects model defaults into each indicator when they are not
        already provided:

        - ``sigma_factory`` is set from the model.
        - ``positive_parameter_factory`` is set from the model.

        :raises ValueError:
            If the model sigma factory is undefined.
        :raises ValueError:
            If the positive-parameter factory is undefined.
        """
        if self.sigma_factory is None:
            raise ValueError("Sigma factory is undefined.")
        if self._positive_factory is None:
            raise ValueError("Positive parameter factory is undefined")

        for ind in self.likert_indicators:
            if ind.sigma_factory is None:
                ind.sigma_factory = self.sigma_factory

            if ind.positive_parameter_factory is None:
                ind.positive_parameter_factory = self._positive_factory

            self._likert_by_name[ind.name] = ind

    def _register_likert_types(self) -> None:
        """Register Likert types used by the model.

        The method injects model defaults into each type when they are not
        already provided:

        - ``sigma_factory`` is set from the model.
        - ``positive_parameter_factory`` is set from the model.

        :raises ValueError:
            If the model sigma factory is undefined.
        :raises ValueError:
            If the positive-parameter factory is undefined.
        """
        if self.sigma_factory is None:
            raise ValueError("Sigma factory is undefined.")
        if self._positive_factory is None:
            raise ValueError("Positive parameter factory is undefined")

        for the_type in self.likert_types:
            if the_type.sigma_factory is None:
                the_type.sigma_factory = self.sigma_factory

            if the_type.positive_parameter_factory is None:
                the_type.positive_parameter_factory = self._positive_factory

            self._likert_type_by_name[the_type.type] = the_type

    def add_latent_variable(self, lv: LatentVariable) -> LatentVariable:
        """Register a latent variable.

        The latent variable must be constructed outside of this class.
        This method injects model defaults for ``draw_type_jax`` and
        ``sigma_factory`` when they are not already set.

        :param lv: A fully specified :class:`LatentVariable` instance.
        :return: The registered :class:`LatentVariable` (same object).
        :raises ValueError: If the model ``sigma_factory`` is undefined.
        :raises ValueError: If a latent variable with the same name is already registered.
        """
        if self.sigma_factory is None:
            raise ValueError("Sigma factory is undefined.")

        if lv.name in self._latent_variables:
            raise ValueError(f"Latent variable '{lv.name}' registered twice.")

        # Inject defaults if not already provided by the caller.
        if lv.draw_type_jax is None:
            lv.draw_type_jax = self.draw_type

        if lv.sigma_factory is None:
            lv.sigma_factory = self.sigma_factory

        # Ensure indicators are stored as a set (helps with equality / consistency checks).
        try:
            lv.indicators = set(lv.indicators)
        except TypeError:
            # If lv.indicators is not iterable for some reason, surface a clear error.
            raise ValueError(
                f"Latent variable '{lv.name}' has invalid indicators: {lv.indicators!r}."
            )

        self._latent_variables[lv.name] = lv
        return lv

    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------

    @property
    def latent_variables(self) -> list[LatentVariable]:
        """Return the registered latent variables."""
        return list(self._latent_variables.values())

    def get_likert_indicator(self, name: str) -> LikertIndicator:
        """Return a registered Likert indicator by name.

        :param name: Indicator name.
        :return: The corresponding :class:`LikertIndicator`.
        :raises KeyError: If the indicator is not registered.
        """
        return self._likert_by_name[name]

    def get_likert_type(self, name: str) -> LikertType:
        """Return a registered Likert type by name.

        :param name:
            Type label.
        :return:
            The corresponding :class:`LikertType`.
        :raises KeyError:
            If the type is not registered.
        """
        return self._likert_type_by_name[name]

    def get_latent_variable(self, name: str) -> LatentVariable:
        """Return a registered latent variable by name.

        :param name: Latent variable name.
        :return: The corresponding :class:`LatentVariable`.
        :raises KeyError: If the latent variable is not registered.
        """
        return self._latent_variables[name]

    # ---------------------------------------------------------------------
    # Model construction
    # ---------------------------------------------------------------------

    def measurement_equations(
        self,
        *,
        draw_type: str | None = None,
    ) -> Expression:
        """Build the joint ordered-probit measurement likelihood (product form).

        This returns a single Biogeme expression corresponding to the product of
        ordered-probit likelihood terms for all registered indicators.

        If the model is configured for a non-maximum-likelihood estimation mode,
        a warning is emitted because the likelihood (product) is typically used
        for maximum-likelihood estimation.

        :param draw_type:
            If provided, overrides the instance draw type.
        :return:
            A Biogeme expression for the measurement likelihood.
        :raises ValueError:
            If no latent variables have been registered.
        :raises ValueError:
            If no Likert indicators have been registered.
        """
        if not self._latent_variables:
            raise ValueError("No latent variables have been registered.")
        if not self._likert_by_name:
            raise ValueError("No Likert indicators have been registered.")

        if self.estimation_mode != EstimationMode.MAXIMUM_LIKELIHOOD:
            warning_msg = f'The likelihood function is usually used for maximum likelihood estimation, as it must be integrated. But the hybrid choice model has been defined for {self.estimation_mode} estimation. This may be an undesired inconsistency'
            logger.warning(warning_msg)

        the_draw_type = self.draw_type if draw_type is None else draw_type

        return measurement_equations_jax(
            latent_variables=list(self._latent_variables.values()),
            likert_indicators=self.likert_indicators,
            likert_types=self.likert_types,
            draw_type=the_draw_type,
        )

    def log_measurement_equations(
        self,
        *,
        draw_type: str | None = None,
    ) -> Expression:
        """Build the joint ordered-probit measurement log-likelihood (sum of logs).

        This returns a single Biogeme expression corresponding to the sum of log
        ordered-probit likelihood terms for all registered indicators.

        If the model is configured for a non-Bayesian estimation mode, a warning
        is emitted because the log-likelihood is typically used for Bayesian
        estimation.

        :param draw_type:
            If provided, overrides the instance draw type.
        :return:
            A Biogeme expression for the log measurement likelihood.
        :raises ValueError:
            If no latent variables have been registered.
        :raises ValueError:
            If no Likert indicators have been registered.
        """
        if not self._latent_variables:
            raise ValueError("No latent variables have been registered.")
        if not self._likert_by_name:
            raise ValueError("No Likert indicators have been registered.")

        if self.estimation_mode != EstimationMode.BAYESIAN:
            warning_msg = f'The log likelihood function is usually used for Bayesian estimation, as it must be integrated. But the hybrid choice model has been defined for {self.estimation_mode} estimation. This may be an undesired inconsistency'
            logger.warning(warning_msg)

        the_draw_type = (self.draw_type if draw_type is None else draw_type) or "NORMAL"

        return log_measurement_equations_jax(
            latent_variables=list(self._latent_variables.values()),
            likert_indicators=self.likert_indicators,
            likert_types=self.likert_types,
            draw_type=the_draw_type,
        )
