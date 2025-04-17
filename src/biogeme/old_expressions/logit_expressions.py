""" Arithmetic expressions accepted by Biogeme: logit

:author: Michel Bierlaire
:date: Sat Sep  9 15:28:39 2023
"""

from __future__ import annotations
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from .base_expressions import Expression
from .jax import JAX_FLOAT
from .numeric_expressions import Numeric
from .convert import validate_and_convert
from ..deprecated import deprecated
from ..exceptions import BiogemeError

if TYPE_CHECKING:
    from . import ExpressionOrNumeric
    from ..database import Database
logger: logging.Logger = logging.getLogger(__name__)


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
        self.util = {
            alt_id: validate_and_convert(util_expression)
            for (alt_id, util_expression) in util.items()
        }

        #: dict of availability formulas
        if av is None:
            self.av = {k: Numeric(1) for k, v in util.items()}
        else:
            self.av = {
                alt_id: validate_and_convert(avail_expression)
                for (alt_id, avail_expression) in av.items()
            }

        self.choice = validate_and_convert(choice)
        """expression for the chosen alternative"""

        self.children.append(self.choice)
        for i, e in self.util.items():
            self.children.append(e)
        for i, e in self.av.items():
            self.children.append(e)

        # Convert the dict into list for biogeme_jax
        self.util_keys = jnp.array(list(self.util.keys()))
        self.util_values = tuple(self.util[k] for k in self.util.keys())
        self.av_keys = jnp.array(list(self.av.keys()))
        self.av_values = tuple(self.av[k] for k in self.av.keys())

    def audit(self, database: Database | None = None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors = []
        list_of_warnings = []
        for e in self.children:
            err, war = e.audit(database)
            list_of_errors += err
            list_of_warnings += war

        if self.util.keys() != self.av.keys():
            the_error = 'Incompatible list of alternatives in logit expression. '
            consistent = False
            my_set = self.util.keys() - self.av.keys()
            if my_set:
                my_set_content = ', '.join(f'{str(k)} ' for k in my_set)
                the_error += (
                    'Id(s) used for utilities and not for ' 'availabilities: '
                ) + my_set_content
            my_set = self.av.keys() - self.util.keys()
            if my_set:
                my_set_content = ', '.join(f'{str(k)} ' for k in my_set)
                the_error += (
                    ' Id(s) used for availabilities and not ' 'for utilities: '
                ) + my_set_content
            list_of_errors.append(the_error)
        else:
            consistent = True
        list_of_alternatives = list(self.util)
        if database is None:
            choices = np.array([self.choice.get_value_c()])
        else:
            choices = database.values_from_database(self.choice)
        correct_choices = np.isin(choices, list_of_alternatives)
        index_of_incorrect_choices = np.argwhere(~correct_choices)
        if index_of_incorrect_choices.any():
            incorrect_choices = choices[index_of_incorrect_choices]
            content = '-'.join(
                '{}[{}]'.format(*t)
                for t in zip(index_of_incorrect_choices, incorrect_choices)
            )
            truncate = 100
            if len(content) > truncate:
                content = f'{content[:truncate]}...'
            the_error = (
                f'The choice variable [{self.choice}] does not '
                f'correspond to a valid alternative for the '
                f'following observations (rownumber[choice]): '
            ) + content
            list_of_errors.append(the_error)

        if consistent:
            if database is None:
                value_choice = self.choice.get_value_c()
                logger.debug(f'{value_choice=}')
                if value_choice not in self.av.keys():
                    the_error = (
                        f'The chosen alternative [{value_choice}] ' f'is not available'
                    )
                    list_of_warnings.append(the_error)
            else:
                choice_availability = database.check_availability_of_chosen_alt(
                    self.av, self.choice
                )
                index_of_unavailable_choices = np.where(~choice_availability)[0]
                if index_of_unavailable_choices.size > 0:
                    incorrect_choices = choices[index_of_unavailable_choices]
                    content = '-'.join(
                        '{}[{}]'.format(*t)
                        for t in zip(index_of_unavailable_choices, incorrect_choices)
                    )
                    truncate = 100
                    if len(content) > truncate:
                        content = f'{content[:truncate]}...'
                    the_error = (
                        f'The chosen alternative [{self.choice}] '
                        f'is not available for the following '
                        f'observations (rownumber[choice]): '
                    ) + content
                    list_of_warnings.append(the_error)

        return list_of_errors, list_of_warnings

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
        s = self.get_class_name()
        s += f'[choice={self.choice}]'
        s += 'U=('
        first = True
        for i, e in self.util.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        s += 'av=('
        first = True
        for i, e in self.av.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        return s

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of alternatives between ( )
            5. the id of the expression for the chosen alternative, preceded
               by a comma.
            6. for each alternative, separated by commas:

                 a. the number of the alternative, as defined by the user,
                 b. the id of the expression for the utility,
                 c. the id of the expression for the availability condition.

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
          \\frac{\\exp(-\\beta_2 V_2) }{ \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        list_of_signatures = []
        for e in self.get_children():
            list_of_signatures += e.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.util)})'
        signature += f',{self.choice.get_id()}'
        for i, e in self.util.items():
            signature += f',{i},{e.get_id()},{self.av[i].get_id()}'
        list_of_signatures += [signature.encode()]
        return list_of_signatures

    def _recursive_construct_jax_function(
        self,
    ) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
        """
        Generates a JAX-compatible function. This function computes the logit-based
        probability calculation based on availability and utility values.

        :return: A function that takes parameters, a row of the database, and random draws.
        """

        def get_value(
            expression: Expression,
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
        ):
            """Retrieve the JAX function of an object and evaluate it."""
            jax_fn = expression._recursive_construct_jax_function()
            return jax_fn(parameters, one_row, the_draws)

        def the_jax_function(
            parameters: jnp.ndarray, one_row: jnp.ndarray, the_draws: jnp.ndarray
        ) -> float:
            """JAX-compatible function for logit probability calculation."""

            choice_id = get_value(self.choice, parameters, one_row, the_draws)
            choice_index = index_of(choice_id, self.av_keys)

            # Get availability of chosen alternative
            av_branches = tuple(
                lambda _, av=av_expr: get_value(av, parameters, one_row, the_draws)
                for av_expr in self.av_values
            )
            chosen_avail = jax.lax.switch(choice_index, av_branches, operand=None)

            def unavailable_branch(_):
                # If the chosen alternative is unavailable
                return -jnp.inf

            def available_branch(_):
                # Compute v_chosen
                branches = tuple(
                    lambda _, V=V_expr: jnp.asarray(
                        get_value(V, parameters, one_row, the_draws), dtype=JAX_FLOAT
                    )
                    for V_expr in self.util_values
                )

                v_chosen = jax.lax.switch(choice_index, branches, operand=None)

                # Compute contributions from all available alternatives
                terms = []
                for i, V in enumerate(self.util_values):
                    avail = get_value(self.av_values[i], parameters, one_row, the_draws)
                    util = get_value(V, parameters, one_row, the_draws)
                    contribution = jnp.where(
                        avail != 0.0, jnp.exp(util - v_chosen), 0.0
                    )
                    terms.append(contribution)

                denom = jnp.sum(jnp.stack(terms))
                return -jnp.log(denom)

            # Conditionally compute result
            result = jax.lax.cond(
                chosen_avail == 0.0, unavailable_branch, available_branch, operand=None
            )
            return result

        return the_jax_function


class _bioLogLogit(LogLogit):
    """log of logit formula

    This expression captures the logarithm of the logit formula. It
    contains one formula for the target alternative, a dict of formula
    for the availabilities and a dict of formulas for the utilities It
    uses only the C++ implementation.
    """


class _bioLogLogitFullChoiceSet(LogLogit):
    """This expression captures the logarithm of the logit formula, where
    all alternatives are supposed to be always available.

       It contains one formula for the target alternative and a dict of
       formulas for the utilities. It uses only the C++ implementation.

    """

    def __init__(
        self, util: dict[int, ExpressionOrNumeric], choice: ExpressionOrNumeric
    ):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        super().__init__(util=util, av=None, choice=choice)
