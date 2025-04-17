"""Handles the segmentation of a single parameter using a discrete variable.

This class manages how a single Beta parameter is segmented according to a
discrete variable. It generates distinct parameters for each category in the
segmentation (except the reference), enabling models to estimate category-specific
effects.

Attributes:
    beta: The base Beta parameter being segmented.
    segmentation_tuple: The DiscreteSegmentationTuple defining the segmentation.
    variable: The discrete variable used for segmentation.
    reference: The reference category for the segmentation.
    mapping: Dictionary of category values to category names, excluding the reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .segmentation_context import DiscreteSegmentationTuple
from biogeme.exceptions import BiogemeError

if TYPE_CHECKING:
    from biogeme.expressions import Beta, Variable, Expression, Numeric


class OneSegmentation:

    def __init__(
        self,
        beta: Beta,
        segmentation_tuple: DiscreteSegmentationTuple,
    ):
        """
        Initialize the segmentation of a Beta parameter.

        Parameters:
            beta: The base Beta parameter to be segmented.
            segmentation_tuple: Describes the discrete segmentation including
                                the mapping of values to category names and the reference category.
        """
        self.beta: Beta = beta
        self.segmentation_tuple = segmentation_tuple
        self.variable: Variable = segmentation_tuple.variable
        self.reference: str = segmentation_tuple.reference
        self.mapping: dict[int, str] = {
            k: v for k, v in segmentation_tuple.mapping.items() if v != self.reference
        }

    def beta_name(self, category: str) -> str:
        """Construct the name of the parameter associated with a specific category

        :param category: name of the category
        :return: name of parameter for the category

        :raise BiogemeError: if the category is not listed in the
            mapping of the segmentation.
        """
        if category != self.reference and category not in self.mapping.values():
            error_msg = (
                f'Unknown category: {category}. List of known categories: '
                f'{self.mapping.values()}'
            )
            raise BiogemeError(error_msg)
        return (
            f'{self.beta.name}_ref'
            if category == self.reference
            else f'{self.beta.name}_diff_{category}'
        )

    def beta_expression(self, category: str | None = None) -> Beta:
        """Constructs the expression for the parameter associated with
            a specific category

        :param category: name of the category. If None, it is the reference category.
        :return: expression of the parameter for the category


        """
        from biogeme.expressions import Beta

        if category is None:
            category = self.reference
        name = self.beta_name(category)
        if category == self.reference:
            lower_bound = self.beta.lower_bound
            upper_bound = self.beta.upper_bound
        else:
            lower_bound = None
            upper_bound = None

        return Beta(
            name,
            self.beta.init_value,
            lower_bound,
            upper_bound,
            self.beta.status,
        )

    def beta_code(self, category: str, assignment: bool) -> str:
        """Constructs the Python code for the expression of the
            parameter associated with a specific category

        :param category: name of the category
        :param assignment: if True, the code includes the assignment to a variable.
        :return: the Python code
        """
        if category == self.reference:
            lower_bound = self.beta.lower_bound
            upper_bound = self.beta.upper_bound
        else:
            lower_bound = None
            upper_bound = None
        name = self.beta_name(category)
        if assignment:
            return (
                f"{name} = Beta('{name}', {self.beta.init_value}, "
                f"{lower_bound}, {upper_bound}, {self.beta.status})"
            )
        return (
            f"Beta('{name}', {self.beta.init_value}, {lower_bound}, "
            f"{upper_bound}, {self.beta.status})"
        )

    def list_of_expressions(self) -> list[Expression]:
        """Create a list of expressions involved in the segmentation of the parameter

        :return: list of expressions
        :rtype: list(biogeme.expressions.Expression)

        """
        from biogeme.expressions import Numeric

        terms = [
            self.beta_expression(category) * (self.variable == Numeric(value))
            for value, category in self.mapping.items()
        ]
        return terms

    def list_of_code(self) -> list[str]:
        """Create a list of Python codes for the expressions involved
            in the segmentation of the parameter

        :return: list of codes
        """
        return [
            (
                f"{self.beta_name(category)} "
                f"* (Variable('{self.variable.name}') == {value})"
            )
            for value, category in self.mapping.items()
        ]
