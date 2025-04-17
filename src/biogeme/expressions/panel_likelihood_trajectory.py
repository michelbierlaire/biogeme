"""Arithmetic expressions accepted by Biogeme: PanelLikelihoodTrajectory

Michel Bierlaire
11.04.2025 09:29
"""

from __future__ import annotations

import copy
import logging

from icecream import ic

from .log import log
from .exp import exp

from .conditional_sum import ConditionalSum, ConditionalTermTuple
from .elementary_expressions import Variable
from .base_expressions import ExpressionOrNumeric, Expression
from .jax_utils import JaxFunctionType
from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)


class PanelLikelihoodTrajectory(Expression):
    """
    Likelihood of a sequences of observations for the same individual
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        # In this case, the formula itself is not a child. It will have to be manipulated by the function
        # `set_maximum_number_of_observations_per_individual`. Therefore, the super().__init__() has no argument.
        super().__init__()
        self.child: Expression | None = None
        self.initial_formula: Expression = child
        self.maximum_number_of_observations_per_individual: int | None = None

    def set_maximum_number_of_observations_per_individual(
        self, max_number: int
    ) -> None:
        from biogeme.database import observation_suffix, RELEVANT_PREFIX

        self.maximum_number_of_observations_per_individual = max_number
        if self.initial_formula.get_class_name() == 'exp':
            copies_of_expression = [
                copy.deepcopy(self.initial_formula.child)
                for _ in range(self.maximum_number_of_observations_per_individual)
            ]
        else:
            copies_of_expression = [
                log(copy.deepcopy(self.initial_formula))
                for _ in range(self.maximum_number_of_observations_per_individual)
            ]
        list_of_terms = []
        for index, a_copy in enumerate(copies_of_expression):
            suffix = observation_suffix(index)
            a_copy.add_suffix_to_all_variables(suffix=suffix)
            the_term = ConditionalTermTuple(
                condition=Variable(f'{RELEVANT_PREFIX}{suffix}'), term=a_copy
            )
            list_of_terms.append(the_term)

        self.child = exp(ConditionalSum(list_of_terms=list_of_terms))
        self.children.append(self.child)

    def __str__(self) -> str:
        return f'PanelLikelihoodTrajectory({self.child})'

    def __repr__(self) -> str:
        return f'PanelLikelihoodTrajectory({repr(self.child)})'

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates recursively a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.child is None:
            error_msg = 'The PanelLikelihoodTrajectory has not been prepared before being evaluated.'
            raise BiogemeError(error_msg)
        if self.maximum_number_of_observations_per_individual is None:
            error_msg = (
                'Maximum number of observations per individual has not been defined.'
            )
            raise BiogemeError(error_msg)

        return self.child.recursive_construct_jax_function()
