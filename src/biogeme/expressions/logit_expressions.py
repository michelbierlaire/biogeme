"""Arithmetic expressions accepted by Biogeme: logit

:author: Michel Bierlaire
:date: Sat Sep  9 15:28:39 2023
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from biogeme.floating_point import JAX_FLOAT, MIN_EXP_ARG
from jax.scipy.special import logsumexp

from .base_expressions import Expression, LogitTuple
from .bayesian import PymcModelBuilderType
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType
from ..deprecated import deprecated
from ..exceptions import BiogemeError

if TYPE_CHECKING:
    from . import ExpressionOrNumeric
logger: logging.Logger = logging.getLogger(__name__)


MASKED_UTILITY_PT = pt.as_tensor_variable(float(MIN_EXP_ARG))


def _describe_tensor(t: pt.TensorVariable) -> str:
    """Compact description of a PyTensor variable for error messages."""
    t_type = getattr(t, "type", None)
    shape = getattr(t_type, "shape", None)
    return f"ndim={getattr(t, 'ndim', '?')}, shape={shape}"


def _prepare_pymc_logit_inputs(
    dataframe: pd.DataFrame,
    choice_builder: PymcModelBuilderType,
    util_builders: list[PymcModelBuilderType],
    av_builders: list[PymcModelBuilderType] | None,
    util_keys: list[int],
) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable | None]:
    """
    Validate and construct PyTensor inputs for the LogLogit PyMC builder.

    This function takes care of:
    - evaluating choice / utilities / availabilities,
    - checking their shapes,
    - stacking utilities / availabilities along axis 1,
    - sanitizing NaN/Inf to safe values.

    :param dataframe: DataFrame used as input to builders.
    :param choice_builder: Builder returning the chosen alternative (per obs).
    :param util_builders: Builders returning utilities (one per alternative).
    :param av_builders: Builders returning availabilities (one per alt), or None.
    :param util_keys: List of alternative identifiers (for error messages).
    :return: (choice_i32, utilities, availabilities_or_None)
    :raises BiogemeError: If shapes are inconsistent or not 1-D where expected.
    """
    # ---- Choice vector ----
    choice = choice_builder(dataframe)
    if choice.ndim != 1:
        raise BiogemeError(
            "LogLogit PyMC builder: the choice expression must return a 1-D "
            f"tensor of per-observation choices.\n"
            f"Got: {_describe_tensor(choice)}"
        )
    choice_i32 = pt.cast(choice, "int32")  # (N,)

    # ---- Utilities ----
    util_tensors = [ub(dataframe) for ub in util_builders]
    bad_utils = [
        (alt, _describe_tensor(t))
        for alt, t in zip(util_keys, util_tensors)
        if getattr(t, "ndim", None) != 1
    ]
    if bad_utils:
        details = ", ".join(f"alt={alt}: {desc}" for alt, desc in bad_utils)
        raise BiogemeError(
            "LogLogit PyMC builder: each utility must be a 1-D tensor "
            "with one value per observation.\n"
            f"Offending utilities: {details}"
        )

    try:
        utilities = pt.stack(util_tensors, axis=1)  # (N, K)
    except (TypeError, ValueError, IndexError) as e:
        shapes = [_describe_tensor(t) for t in util_tensors]
        raise BiogemeError(
            "LogLogit PyMC builder: utilities for some alternatives are not "
            "shape-compatible for stacking along axis=1.\n"
            "Utilities (alt_id → tensor): "
            + ", ".join(f"{alt}: {desc}" for alt, desc in zip(util_keys, shapes))
            + "\nExpected all utilities to be 1-D tensors with the same "
            "length N (number of rows in the dataframe)."
        ) from e

    # Sanitize upstream NaN/Inf in utilities to a very small finite value
    utilities = pt.where(_isfinite(utilities), utilities, MASKED_UTILITY_PT)

    # ---- Availabilities ----
    if av_builders is None:
        availabilities = None
    else:
        av_tensors = [ab(dataframe) for ab in av_builders]
        bad_av = [
            (alt, _describe_tensor(t))
            for alt, t in zip(util_keys, av_tensors)
            if getattr(t, "ndim", None) != 1
        ]
        if bad_av:
            details = ", ".join(f"alt={alt}: {desc}" for alt, desc in bad_av)
            raise BiogemeError(
                "LogLogit PyMC builder: each availability must be a 1-D tensor "
                "with one value per observation.\n"
                f"Offending availabilities: {details}"
            )

        try:
            availabilities = pt.stack(av_tensors, axis=1)  # (N, K)
        except (TypeError, ValueError, IndexError) as e:
            shapes = [_describe_tensor(t) for t in av_tensors]
            raise BiogemeError(
                "LogLogit PyMC builder: availabilities for some alternatives "
                "are not shape-compatible for stacking along axis=1.\n"
                "Availabilities (alt_id → tensor): "
                + ", ".join(f"{alt}: {desc}" for alt, desc in zip(util_keys, shapes))
                + "\nExpected all availabilities to be 1-D tensors with the same "
                "length N (number of rows in the dataframe)."
            ) from e

        # Treat non-finite availability as unavailable (0.0)
        availabilities = pt.where(_isfinite(availabilities), availabilities, 0.0)

    return choice_i32, utilities, availabilities


# PyTensor does not have isfinite, so define our own
def _isfinite(x: pt.TensorVariable) -> pt.TensorVariable:
    """Elementwise finite check for PyTensor: True iff not NaN and not Inf."""
    return ~(pt.isnan(x) | pt.isinf(x))


def index_of(key: float, keys: list[int]):
    """Function returning the index of a kex for biogeme_jax"""
    return jnp.argmax(keys == key)


class LogLogit(Expression):
    """Expression capturing the logit formula.

    It contains one formula for the target alternative, a dict of
    formula for the availabilities and a dict of formulas for the
    utilities

    """

    def __init__(
        self,
        util: dict[int, ExpressionOrNumeric],
        av: dict[int, ExpressionOrNumeric] | None,
        choice: ExpressionOrNumeric,
    ):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param av: dictionary where the keys are the identifiers of
                   the alternatives, and the elements are object of
                   type biogeme.expressions.Expression defining the
                   availability conditions. If av is None, all the
                   alternatives are assumed to be always available

        :type av: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)
        self.util: dict[int, Expression] = {
            alt_id: validate_and_convert(util_expression)
            for (alt_id, util_expression) in util.items()
        }

        #: dict of availability formulas
        self.av: dict[int, Expression] | None = None
        if av is not None:
            self.av = {
                alt_id: validate_and_convert(avail_expression)
                for (alt_id, avail_expression) in av.items()
            }
            for i, e in self.av.items():
                self.children.append(e)
            self.av_keys = jnp.array(list(self.av.keys()), dtype=JAX_FLOAT)
            self.av_values = tuple(self.av[k] for k in self.av.keys())

        self.choice: Expression = validate_and_convert(choice)
        """expression for the chosen alternative"""

        self.children.append(self.choice)
        for i, e in self.util.items():
            self.children.append(e)

        # Convert the dict into list for biogeme_jax
        self.util_keys = jnp.array(list(self.util.keys()), dtype=JAX_FLOAT)
        self.util_values = tuple(self.util[k] for k in self.util.keys())

    def deep_flat_copy(self) -> LogLogit:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_util = {key: util.deep_flat_copy() for key, util in self.util.items()}
        copy_av = (
            {
                key: av.deep_flat_copy() if av is not None else None
                for key, av in self.av.items()
            }
            if self.av is not None
            else None
        )
        copy_choice = self.choice.deep_flat_copy()
        return type(self)(util=copy_util, av=copy_av, choice=copy_choice)

    def logit_choice_avail(self) -> list[LogitTuple]:
        result: list[LogitTuple] = list(
            chain.from_iterable(e.logit_choice_avail() for e in self.children)
        )
        if self.av is not None:
            this_tuple: LogitTuple = LogitTuple(
                choice=self.choice, availabilities=self.av
            )
            result.append(this_tuple)
        return result

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the chosen alternative does not correspond
            to any of the utility functions

        :raise BiogemeError: if the chosen alternative does not correspond
            to any of entry in the availability condition

        """
        choice = int(self.choice.get_value())
        if choice not in self.util:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of utility functions: {self.util.keys()}'
            )
            raise BiogemeError(error_msg)
        if choice not in self.av:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of availabilities: {self.av.keys()}'
            )
            raise BiogemeError(error_msg)
        if self.av[choice].get_value() == 0.0:
            return -np.log(0)
        v_chosen = self.util[choice].get_value()
        denom = 0.0
        for i, V in self.util.items():
            if self.av[i].get_value() != 0.0:
                denom += np.exp(V.get_value() - v_chosen)
        return -np.log(denom)

    @deprecated(get_value)
    def getValue(self) -> float:
        pass

    def __str__(self) -> str:
        s = f'{self.get_class_name()}[choice={self.choice}]'
        util_str = ', '.join(f'{int(i)}:{e}' for i, e in self.util.items())
        s += f'U=({util_str})'
        if self.av is None:
            s += '[always available]'
        else:
            av_str = ', '.join(f'{int(i)}:{e}' for i, e in self.av.items())
            s += f'av=({av_str})'
        return s

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a JAX-compatible function. This function computes the logit-based
        probability calculation based on availability and utility values.

        :return: A function that takes parameters, a row of the database,
            and random draws.
        """

        def get_value(
            expression: Expression,
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            """Retrieve the JAX function of an object and evaluate it."""
            jax_fn = expression.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
            return jax_fn(parameters, one_row, the_draws, the_random_variables)

        if self.av is None:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                """JAX-compatible function for logit probability calculation
                with availability."""

                choice_id = get_value(
                    self.choice, parameters, one_row, the_draws, the_random_variables
                )
                choice_index = index_of(choice_id, self.util_keys)

                # Compute v_chosen
                branches = tuple(
                    lambda _, V=V_expr: jnp.asarray(
                        get_value(
                            V, parameters, one_row, the_draws, the_random_variables
                        ),
                        dtype=JAX_FLOAT,
                    )
                    for V_expr in self.util_values
                )
                v_chosen = jax.lax.switch(choice_index, branches, operand=None)

                # Vectorized computation of utilities and availabilities
                all_utils = jnp.array(
                    [
                        get_value(
                            V, parameters, one_row, the_draws, the_random_variables
                        )
                        - v_chosen
                        for V in self.util_values
                    ]
                )

                # Compute the log-sum-exp safely
                result = -logsumexp(all_utils)
                return result

            return the_jax_function

        else:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                """JAX-compatible function for logit probability calculation."""

                choice_id = get_value(
                    self.choice, parameters, one_row, the_draws, the_random_variables
                )
                choice_index = index_of(choice_id, self.util_keys)

                # Get availability of chosen alternative
                av_branches = tuple(
                    lambda _, av=av_expr: get_value(
                        av, parameters, one_row, the_draws, the_random_variables
                    )
                    for av_expr in self.av_values
                )
                chosen_avail = jax.lax.switch(choice_index, av_branches, operand=None)

                def unavailable_branch(_):
                    # If the chosen alternative is unavailable
                    return -jnp.finfo(JAX_FLOAT).max

                def available_branch(_):
                    # Compute v_chosen
                    branches = tuple(
                        lambda _, V=V_expr: jnp.asarray(
                            get_value(
                                V, parameters, one_row, the_draws, the_random_variables
                            ),
                            dtype=JAX_FLOAT,
                        )
                        for V_expr in self.util_values
                    )
                    v_chosen = jax.lax.switch(choice_index, branches, operand=None)

                    # Vectorized computation of utilities and availabilities
                    all_utils = jnp.array(
                        [
                            get_value(
                                V, parameters, one_row, the_draws, the_random_variables
                            )
                            - v_chosen
                            for V in self.util_values
                        ]
                    )
                    all_avail = jnp.array(
                        [
                            get_value(
                                A, parameters, one_row, the_draws, the_random_variables
                            )
                            for A in self.av_values
                        ]
                    )

                    masked_utils = jnp.where(all_avail != 0.0, all_utils, -jnp.inf)
                    return -logsumexp(masked_utils)

                # Conditionally compute result
                result = jax.lax.cond(
                    chosen_avail == 0.0,
                    unavailable_branch,
                    available_branch,
                    operand=None,
                )
                return result

            return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """Return a *vectorized* PyTensor builder computing per-observation
        log-likelihoods for the multinomial logit, compatible with PyMC.

        Shapes (vectorized over observations):
          - util_keys_vec: (K,)
          - choice:        (N,)  arbitrary codes (mapped to column indices)
          - U / AV:        (N, K)
          - output ll:     (N,)
        """
        util_keys = list(self.util.keys())
        util_builders = [
            self.util[k].recursive_construct_pymc_model_builder() for k in util_keys
        ]
        choice_builder = self.choice.recursive_construct_pymc_model_builder()
        av_builders = (
            [self.av[k].recursive_construct_pymc_model_builder() for k in util_keys]
            if self.av is not None
            else None
        )

        util_keys_vec = pt.constant(np.asarray(util_keys, dtype=np.int32))

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            # Delegate all validation + stacking to the helper
            choice_i32, utilities, availabilities = _prepare_pymc_logit_inputs(
                dataframe=dataframe,
                choice_builder=choice_builder,
                util_builders=util_builders,
                av_builders=av_builders,
                util_keys=util_keys,
            )

            if availabilities is None:
                U_masked = utilities
            else:
                U_masked = pt.where(
                    pt.neq(availabilities, 0.0), utilities, MASKED_UTILITY_PT
                )

            # Map choice codes to utility columns
            matches = pt.eq(choice_i32[:, None], util_keys_vec[None, :])  # (N, K)
            idx = pt.argmax(matches, axis=1)  # (N,)
            any_match = pt.any(matches, axis=1)  # (N,)

            n_obs = choice_i32.shape[0]
            row_idx = pt.arange(n_obs)
            # Safe index: use column 0 when there is no match; penalize later
            idx_safe = pt.where(any_match, idx, 0)
            chosen_logits = U_masked[row_idx, idx_safe]  # (N,)
            ll = chosen_logits - pt.logsumexp(U_masked, axis=1)  # (N,)

            neg_large = pt.cast(pt.as_tensor_variable(-1.0e30), utilities.dtype)

            if availabilities is not None:
                chosen_av = availabilities[row_idx, idx_safe]
                ll = pt.where(pt.eq(chosen_av, 0.0), neg_large, ll)

            ll = pt.where(any_match, ll, neg_large)
            return ll

        return builder

    def recursive_construct_pymc_model_builder_old(self) -> PymcModelBuilderType:
        """Return a *vectorized* PyTensor builder computing per-observation
        log-likelihoods for the multinomial logit, compatible with PyMC.

        Shapes (vectorized over observations):
          - util_keys_vec: (K,)
          - choice:        (N,)  arbitrary codes (mapped to column indices)
          - U / AV:        (N, K)
          - output ll:     (N,)
        """
        util_keys = list(self.util.keys())
        util_builders = [
            self.util[k].recursive_construct_pymc_model_builder() for k in util_keys
        ]
        choice_builder = self.choice.recursive_construct_pymc_model_builder()
        av_builders = (
            [self.av[k].recursive_construct_pymc_model_builder() for k in util_keys]
            if self.av is not None
            else None
        )

        util_keys_vec = pt.constant(np.asarray(util_keys, dtype=np.int32))

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            choice_i32 = pt.cast(choice_builder(dataframe), "int32")  # (N,)

            utilities = pt.stack(
                [ub(dataframe) for ub in util_builders], axis=1
            )  # (N, K)
            # Sanitize upstream NaN/Inf in utilities to a very small finite value
            utilities = pt.where(_isfinite(utilities), utilities, MASKED_UTILITY_PT)

            if av_builders is None:
                U_masked = utilities
                availabilities = None
            else:
                availabilities = pt.stack(
                    [ab(dataframe) for ab in av_builders], axis=1
                )  # (N, K)
                # Treat non-finite availability as unavailable (0.0)
                availabilities = pt.where(
                    _isfinite(availabilities), availabilities, 0.0
                )
                U_masked = pt.where(
                    pt.neq(availabilities, 0.0), utilities, MASKED_UTILITY_PT
                )

            matches = pt.eq(choice_i32[:, None], util_keys_vec[None, :])  # (N, K)
            idx = pt.argmax(matches, axis=1)  # (N,)
            any_match = pt.any(matches, axis=1)  # (N,)

            n_obs = choice_i32.shape[0]
            row_idx = pt.arange(n_obs)
            # Safe index: use column 0 when there is no match; penalize later
            idx_safe = pt.where(any_match, idx, 0)
            chosen_logits = U_masked[row_idx, idx_safe]  # (N,)
            ll = chosen_logits - pt.logsumexp(U_masked, axis=1)  # (N,)

            neg_large = pt.cast(pt.as_tensor_variable(-1.0e30), utilities.dtype)

            if availabilities is not None:
                chosen_av = availabilities[row_idx, idx_safe]
                ll = pt.where(pt.eq(chosen_av, 0.0), neg_large, ll)

            ll = pt.where(any_match, ll, neg_large)
            return ll

        return builder
