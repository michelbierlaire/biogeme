"""Latent-variable definitions and normalization utilities for structural equation models.

This module provides lightweight data structures used to define latent variables
in Biogeme latent-variable models.

- :class:`Normalization` defines the anchor (indicator) and coefficient used to
  identify a latent variable.
- :class:`LatentVariable` combines a latent variable name, its
  :class:`~biogeme.latent_variables.structural_equation.StructuralEquation`, the
  related indicators, and the information required to build a JAX-compatible
  structural expression.

Michel Bierlaire
Thu Dec 11 2025, 15:30:00
"""

from collections.abc import Iterable
from dataclasses import dataclass

from biogeme.expressions import Expression

from .positive_parameter_factory import SigmaFactory
from .structural_equation import StructuralEquation


@dataclass
class Normalization:
    """Normalization information for a latent variable.

    A normalization anchors a latent variable by fixing one measurement loading.

    :param indicator:
        Name of the indicator used as anchor for the normalization.
    :param coefficient:
        Numeric value of the fixed loading associated with the anchor indicator.
    """

    indicator: str
    coefficient: float


@dataclass
class LatentVariable:
    """Define a latent variable with its structural equation and metadata.

    The structural equation is provided through a
    :class:`~biogeme.latent_variables.structural_equation.StructuralEquation`.
    For JAX-based computations, the complete expression (including the stochastic
    error term) can be obtained via :pyattr:`structural_equation_jax`.

    :param name:
        Name of the latent variable.
    :param structural_equation:
        Structural equation specification (deterministic part and stochastic
        error term definition).
    :param indicators:
        Collection of indicator names linked to this latent variable.
    :param normalization:
        Normalization information (anchor indicator and fixed coefficient).
    :param draw_type_jax:
        Identifier of the draw type used when constructing the stochastic error
        term for JAX (passed to ``structural_equation.expression(draw_type=...)``).
        If None, :pyattr:`structural_equation_jax` cannot be evaluated.
    :param sigma_factory:
        Factory used to create the strictly positive scale parameter of the
        stochastic error term. If None, :pyattr:`structural_equation_jax` cannot
        be evaluated.
    """

    name: str
    structural_equation: StructuralEquation
    indicators: Iterable[str]
    normalization: Normalization
    draw_type_jax: str | None = None
    sigma_factory: SigmaFactory | None = None

    @property
    def structural_equation_jax(self) -> Expression:
        """Return the full structural equation expression for JAX.

        This property requires both ``draw_type_jax`` and ``sigma_factory`` to be
        defined. It injects ``sigma_factory`` into the underlying
        :attr:`structural_equation` before calling
        ``structural_equation.expression(draw_type=draw_type_jax)``.

        :return:
            The complete structural equation expression including the stochastic
            error term.
        :raises ValueError:
            If ``draw_type_jax`` is not defined.
        :raises ValueError:
            If ``sigma_factory`` is not defined.
        """
        if self.draw_type_jax is None:
            raise ValueError('The type of draws has not been defined.')
        if self.sigma_factory is None:
            raise ValueError('The sigma factory has not been defined.')
        self.structural_equation.sigma_factory = self.sigma_factory
        return self.structural_equation.expression(draw_type=self.draw_type_jax)
