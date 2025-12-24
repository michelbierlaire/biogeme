"""Arithmetic expressions accepted by Biogeme: BoxCox

Michel Bierlaire
Mon Nov 03 2025, 17:16:46
"""

from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt
from biogeme.expressions import Beta, Expression

from .bayesian import PymcModelBuilderType
from .binary_expressions import BinaryOperator
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class BoxCox(BinaryOperator):
    """
    Box–Cox transform with McLaurin expansion near :math:`\\ell = 0`.

    .. math::

        B(x, \\ell) = \\frac{x^{\\ell} - 1}{\\ell}

    with the limit

    .. math::

        \\lim_{\\ell \\to 0} B(x, \\ell) = \\log(x).

    To avoid numerical issues, we use a McLaurin expansion for small
    :math:`\\ell`:

    .. math::

        \\log(x)
        + \\ell \\log(x)^2
        + \\frac{1}{6} \\ell^2 \\log(x)^3
        + \\frac{1}{24} \\ell^3 \\log(x)^4.

    and a special case :math:`B(0, \\ell) = 0`.

    This class reproduces the behaviour of ``boxcox_old`` but implements
    the piecewise logic with JAX / PyTensor control flow instead of
    :class:`Elem`, so it is compatible with JAX and PyMC backends.
    """

    def __init__(self, x: Expression, ell: Expression):
        # Always store the validated/converted children and pass them to the
        # parent constructor. This avoids keeping both raw and converted
        # references and prevents duplicating children in the expression tree.
        x_c = validate_and_convert(x)
        ell_c = validate_and_convert(ell)
        super().__init__(left=x_c, right=ell_c)
        self.x = x_c
        self.ell = ell_c

    def __str__(self) -> str:
        return f'BoxCox({self.left}, {self.right})'

    def __repr__(self) -> str:
        return f'BoxCox({repr(self.left)}, {repr(self.right)})'

    def deep_flat_copy(self) -> BoxCox:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(x=left_copy, ell=right_copy)

    def get_value(self) -> float:
        """
        Evaluate the Box–Cox transform for scalar values.

        - If ``x == 0``, returns 0.0.
        - If ``|ell| < 1e-5``, uses the McLaurin expansion around ``ell = 0``.
        - Otherwise, uses the standard Box–Cox formula.

        :return: Scalar value of the Box–Cox transform.
        """
        # Retrieve scalar values; will raise if not possible
        x_value = float(self.x.get_value())
        ell_value = float(self.ell.get_value())

        # Convention: B(0, ell) = 0 for any ell
        if x_value == 0.0:
            return 0.0

        # McLaurin expansion around ell = 0 for numerical stability
        if abs(ell_value) < 1.0e-5:
            lx = math.log(x_value)
            return (
                lx
                + ell_value * lx**2
                + (ell_value**2) * lx**3 / 6.0
                + (ell_value**3) * lx**4 / 24.0
            )

        # Regular Box–Cox formula
        return (x_value**ell_value - 1.0) / ell_value

    # ------------------------------------------------------------------
    # JAX builder
    # ------------------------------------------------------------------
    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        JAX implementation of the Box–Cox transform.

        Uses:
        - regular formula when ``|ell| >= 1e-5``,
        - McLaurin expansion when ``|ell| < 1e-5``,
        - value 0.0 when ``x == 0``.
        """
        get_x = self.x.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        get_ell = self.ell.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax(theta, one_row, draws, rvars):
            x = get_x(theta, one_row, draws, rvars)
            ell = get_ell(theta, one_row, draws, rvars)

            # regular branch
            def regular_branch(_):
                return (jnp.power(x, ell) - 1.0) / ell

            # McLaurin expansion branch
            def mclaurin_branch(_):
                lx = jnp.log(x)
                return (
                    lx + ell * lx**2 + (ell**2) * lx**3 / 6.0 + (ell**3) * lx**4 / 24.0
                )

            # choose between regular and McLaurin based on |ell|
            def inner(_):
                return jax.lax.cond(
                    jnp.abs(ell) < 1.0e-5,
                    mclaurin_branch,
                    regular_branch,
                    operand=None,
                )

            # top-level: x == 0 -> 0.0, else inner
            val = jax.lax.cond(x == 0.0, lambda _: 0.0, inner, operand=None)
            return val

        return the_jax

    # ------------------------------------------------------------------
    # PyMC / PyTensor builder
    # ------------------------------------------------------------------
    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        PyTensor implementation of the Box–Cox transform, mirroring the
        original ``boxcox_old`` piecewise logic with ``pt.switch``.
        """
        x_b = self.x.recursive_construct_pymc_model_builder()
        ell_b = self.ell.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            x = x_b(dataframe)
            ell = ell_b(dataframe)

            # Warn if ell is a Beta without bounds, as in boxcox_old
            # (we inspect the expression tree statically, so do this outside
            # builder if you prefer; left here for simplicity)
            # NOTE: if ell is not a Beta, this does nothing.
            # You can drop this if you already warn elsewhere.
            if isinstance(self.ell, Beta) and (
                self.ell.upper_bound is None or self.ell.lower_bound is None
            ):
                warning_msg = (
                    f'It is advised to set the bounds on parameter {self.ell.name}. '
                    f'A value of -10 and 10 should be appropriate: '
                    f'Beta("{self.ell.name}", {self.ell.init_value}, -10, 10, '
                    f'{self.ell.status})'
                )
                logger.warning(warning_msg)

            lx = pt.log(x)  # your Expression log, which maps to pt.log

            regular = (x**ell - 1.0) / ell
            mclaurin = lx + ell * lx**2 + ell**2 * lx**3 / 6.0 + ell**3 * lx**4 / 24.0
            close_to_zero = pt.lt(ell, 1.0e-5) & pt.gt(ell, -1.0e-5)

            smooth = pt.switch(close_to_zero, mclaurin, regular)
            result = pt.switch(pt.eq(x, 0.0), 0.0, smooth)
            return result

        return builder
