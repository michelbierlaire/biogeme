"""Structural equations for latent variables.

This module defines :class:`StructuralEquation`, a lightweight specification of
latent-variable structural equations. A structural equation consists of:

- a latent variable name,
- a set of explanatory variables entering the deterministic part,
- a stochastic error term built from simulation draws, scaled by a strictly
  positive parameter (sigma).

The deterministic part is a linear-in-parameters expression built with one
coefficient per explanatory variable.

The full structural equation returned by :meth:`StructuralEquation.expression`
creates a :class:`~biogeme.expressions.DistributedParameter` whose child is the
sum of the deterministic component and a random term
``sigma * Draws(draw_type=...)``.

Michel Bierlaire
Tue Dec 23 2025, 16:00:07
"""

from collections.abc import Iterable
from dataclasses import dataclass

from biogeme.expressions import (
    Beta,
    DistributedParameter,
    Draws,
    Expression,
    LinearTermTuple,
    LinearUtility,
    Numeric,
    Variable,
)

from .positive_parameter_factory import SigmaFactory


@dataclass
class StructuralEquation:
    """Specification of a latent-variable structural equation.

    The structural equation defines the deterministic part as a linear utility
    over the explanatory variables and adds a stochastic term based on draws.

    :param name:
        Name of the latent variable.
    :param explanatory_variables:
        Iterable of variable names entering the deterministic part.
    :param sigma_factory:
        Factory creating a strictly positive scale parameter (sigma) used to
        scale the draw-based stochastic term. Required unless an explicit
        ``scale_parameter`` is provided to :meth:`expression`.
    """

    name: str
    explanatory_variables: Iterable[str]
    sigma_factory: SigmaFactory | None = None

    @property
    def prefix(self) -> str:
        """Prefix used to namespace parameters belonging to this structural equation."""
        return f'struct_{self.name}'

    def get_expression_deterministic_part(self) -> Expression:
        """Construct the deterministic part of the structural equation.

        A coefficient :class:`~biogeme.expressions.Beta` is created for each
        explanatory variable and assembled into a
        :class:`~biogeme.expressions.LinearUtility`.

        :return:
            A linear expression representing the deterministic component.
        """
        coefficients = {
            variable_name: Beta(f'{self.prefix}_{variable_name}', 0.0, None, None, 0)
            for variable_name in self.explanatory_variables
        }

        if not coefficients:
            return Numeric(0)

        the_expression = LinearUtility(
            [
                LinearTermTuple(
                    beta=coefficients[variable_name],
                    x=Variable(variable_name),
                )
                for variable_name in self.explanatory_variables
            ]
        )
        return the_expression

    def expression(
        self, *, draw_type: str, scale_parameter: Expression | None = None
    ) -> Expression:
        """Construct the full structural equation including the stochastic term.

        If ``scale_parameter`` is not provided, the method uses
        :attr:`sigma_factory` to create a strictly positive sigma parameter named
        with :attr:`prefix`.

        The returned expression is a
        :class:`~biogeme.expressions.DistributedParameter` with child expression:

        ``deterministic + scale_parameter * Draws(draw_type=draw_type)``.

        :param draw_type:
            Draw type identifier used by :class:`~biogeme.expressions.Draws`.
        :param scale_parameter:
            Optional scale parameter (sigma). If None, it is created through
            :attr:`sigma_factory`.
        :return:
            Expression representing the full structural equation.
        :raises ValueError:
            If ``scale_parameter`` is None and :attr:`sigma_factory` is undefined.
        """
        if scale_parameter is None:
            if self.sigma_factory is None:
                raise ValueError('Sigma factory is undefined.')
            scale_parameter = self.sigma_factory(prefix=self.prefix)
        draws = Draws(name=f'{self.prefix}_draws', draw_type=draw_type)
        deterministic = self.get_expression_deterministic_part()
        return DistributedParameter(
            self.name, child=deterministic + scale_parameter * draws
        )
