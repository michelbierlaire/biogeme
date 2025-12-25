"""
Utilities for Likert-type indicators used in Biogeme latent-variable models.

This module defines small helper data structures to work with Likert-scale
survey items in Biogeme measurement equations.

Two concepts are separated:

- :class:`LikertType` describes a category of indicators of the same type  (number of categories, symmetry,
  and how ordered thresholds are parameterized).
- :class:`LikertIndicator` describes an *item* (statement / variable name)
  and provides consistent parameter naming for the measurement equation.

`LikertType.get_thresholds` returns the ordered cut-points (thresholds)
corresponding to the type definition. Symmetric scales build thresholds as
cumulative sums of strictly positive increments on the positive side, mirrored
around zero (and optionally inserting a central 0 for an even number of
categories). Non-symmetric scales build thresholds as a monotone sequence.

The concrete creation of strictly positive parameters (for increments and
measurement scales) is delegated to factories, so the same code can be used in
maximum-likelihood and Bayesian contexts.

Michel Bierlaire
Tue Dec 23 2025, 15:08:48
"""

from dataclasses import dataclass

from biogeme.expressions import Beta, Expression, Numeric

from .positive_parameter_factory import PositiveParameterFactory, SigmaFactory


@dataclass
class LikertType:
    """Describe a Likert scale and build its ordered thresholds.

    A :class:`LikertType` represents the definition of a Likert scale shared by
    one or several indicators (items).

    :param type:
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
    :param sigma_factory:
        Factory creating strictly positive measurement scale parameters.
    :param positive_parameter_factory:
        Factory creating strictly positive parameters used for threshold
        increments.
    :param fix_first_cut_point_for_non_symmetric_thresholds:
        If not None and ``symmetric`` is False, fix the first cut-point to this
        numeric value; otherwise the first cut-point is a free parameter.
    """

    type: str
    symmetric: bool
    categories: list[int]
    neutral_labels: list[int]
    scale_normalization: str
    sigma_factory: SigmaFactory | None = None
    positive_parameter_factory: PositiveParameterFactory | None = None
    fix_first_cut_point_for_non_symmetric_thresholds: float | None = None

    def get_thresholds(self) -> list[Expression]:
        """Construct and return the ordered cut-points (thresholds).

        :return:
            A list of ``K-1`` expressions, where ``K`` is the number of
            categories, representing strictly increasing cut-points.
        """
        number_of_categories = len(self.categories)
        if self.positive_parameter_factory is None:
            raise ValueError("Positive parameter factory is undefined")
        if number_of_categories < 2:
            raise ValueError("Likert scale must have at least 2 categories.")
        n_tau = number_of_categories - 1
        if self.symmetric:
            thresholds = self._build_symmetric(n_tau=n_tau)
        else:
            thresholds = self._build_monotone(n_tau=n_tau)

        if len(thresholds) != n_tau:
            raise RuntimeError(
                f"Internal error: expected {n_tau} cutpoints for K={number_of_categories}, got {len(thresholds)}."
            )
        return thresholds

    def _build_symmetric(self, n_tau: int) -> list[Expression]:
        """Symmetric thresholds around zero (with optional central 0 if needed)."""

        # number of strictly positive increments that define the half-side
        n_deltas = n_tau // 2

        deltas: list[Expression] = [
            self.positive_parameter_factory(
                name=f"delta_{k}",
                prefix=self.type,
                value=-0.86 + 0.43 * k,  # keep your current init scheme
            )
            for k in range(n_deltas)
        ]

        cum: list[Expression] = []
        running: Expression | None = None
        for d in deltas:
            running = d if running is None else running + d
            cum.append(running)

        # n_tau even  <=> K odd  : [-s_h,...,-s_1, s_1,...,s_h]
        # n_tau odd   <=> K even : [-s_h,...,-s_1, 0, s_1,...,s_h]
        thresholds: list[Expression] = []
        for s in reversed(cum):
            thresholds.append(-s)

        if n_tau % 2 == 1:
            thresholds.append(Numeric(0.0))

        for s in cum:
            thresholds.append(s)

        return thresholds

    def _build_monotone(self, n_tau: int) -> list[Expression]:
        """Monotone (non-symmetric) thresholds using cumulative positive increments."""

        thresholds: list[Expression] = []
        if self.fix_first_cut_point_for_non_symmetric_thresholds is not None:
            thresholds.append(
                Numeric(self.fix_first_cut_point_for_non_symmetric_thresholds)
            )
            start_k = 2
            last: Expression = Numeric(
                self.fix_first_cut_point_for_non_symmetric_thresholds
            )
        else:
            # You can keep Beta(...) if you want tau_1 unconstrained:
            tau_1 = Beta(f"{self.type}_tau_1", 0.0, None, None, 0)
            thresholds.append(tau_1)
            start_k = 2
            last = tau_1

        # Build tau_k = tau_{k-1} + delta_{k-1}, with delta>0
        for k in range(start_k, n_tau + 1):
            delta = self.positive_parameter_factory(
                name=f"delta_{k-1}",
                prefix=self.type,
                value=0.3 + 0.5 * (k - 2),  # keep your current init scheme
            )
            last = last + delta
            thresholds.append(last)

        return thresholds


@dataclass
class LikertIndicator:
    """Represent a Likert indicator and provide helpers for measurement parameters.

    The class does not store the scale definition itself (categories, thresholds,
    etc.). Those are described by :class:`LikertType`. This class focuses on
    consistent naming and creation of parameters used in the measurement
    equation.

    :param name:
        Short identifier of the indicator, used to construct parameter names.
    :param statement:
        Text of the statement that respondents evaluate on the Likert scale.
    :param type:
        Optional indicator-type label (for example to implement threshold sharing
        policies by type).
    :param sigma_factory:
        Factory creating strictly positive measurement scale parameters.
    :param positive_parameter_factory:
        Factory creating strictly positive parameters (kept for API symmetry with
        other components; may be unused depending on the model design).
    """

    name: str
    statement: str
    type: str
    sigma_factory: SigmaFactory | None = None
    positive_parameter_factory: PositiveParameterFactory | None = None

    @property
    def intercept_parameter_name(self) -> str:
        """
        Return the name of the intercept parameter for this indicator.

        :return:
            The parameter name used for the measurement intercept.
        """
        return f'measurement_intercept_{self.name}'

    @property
    def intercept_parameter(self) -> Beta:
        """
        Return the Biogeme parameter corresponding to the measurement intercept.

        :return:
            A :class:`Beta` object representing the intercept parameter.
        """
        return Beta(self.intercept_parameter_name, 0, None, None, 0)

    @property
    def scale_parameter(self) -> Expression:
        """Return the measurement scale parameter for this indicator.

        :return:
            An expression representing a strictly positive scale parameter.
        :raises ValueError:
            If ``sigma_factory`` is undefined.
        """
        if self.sigma_factory is None:
            raise ValueError('Sigma factory is undefined')
        return self.sigma_factory(prefix=f'measurement_{self.name}')

    def get_lv_coefficient_parameter_name(self, latent_variable_name: str) -> str:
        """
        Build the name of the coefficient linking a latent variable to this indicator.

        :param latent_variable_name:
            Name of the latent variable appearing in the measurement equation.
        :return:
            The parameter name used for the corresponding coefficient.
        """
        return f'measurement_coefficient_{latent_variable_name}_{self.name}'

    def get_lv_coefficient_parameter(self, latent_variable_name: str) -> Beta:
        """
        Return the Biogeme parameter for the coefficient of a latent variable
        in this indicator's measurement equation.

        :param latent_variable_name:
            Name of the latent variable appearing in the measurement equation.
        :return:
            A :class:`Beta` object representing the corresponding coefficient.
        """
        return Beta(
            self.get_lv_coefficient_parameter_name(
                latent_variable_name=latent_variable_name
            ),
            0,
            None,
            None,
            0,
        )
