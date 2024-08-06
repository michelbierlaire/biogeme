"""Class that provides some automatic specification for segmented parameters

:author: Michel Bierlaire
:date: Thu Feb  2 09:42:36 2023

"""

from __future__ import annotations

from typing import Iterable

import biogeme.expressions
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, bioMultSum, Variable, Numeric, Expression
from biogeme.deprecated import deprecated


class DiscreteSegmentationTuple:
    """Characterization of a segmentation"""

    def __init__(
        self,
        variable: Variable | str,
        mapping: dict[int, str],
        reference: str | None = None,
    ):
        """Ctor

        :param variable: socio-economic variable used for the segmentation, or its name
        :type variable: biogeme.expressions.Variable or str

        :param mapping: maps the values of the variable with the name of a category
        :type mapping: dict(int: str)

        :param reference: name of the reference category. If None, an
            arbitrary category is selected as reference.  :type:
        :type reference: str

        :raise BiogemeError: if the name of the reference category
            does not appear in the list.

        """

        self.variable: Variable = (
            variable if isinstance(variable, Variable) else Variable(variable)
        )
        self.mapping: dict[int, str] = mapping
        if reference is None:
            self.reference: str = next(iter(mapping.values()))
        elif reference not in mapping.values():
            error_msg = (
                f'Reference category {reference} does not appear in the list '
                f'of categories: {mapping.values()}'
            )
            raise BiogemeError(error_msg)
        else:
            self.reference: str = reference

        if self.reference is None:
            raise BiogemeError('Reference should not be None')

    def __repr__(self) -> str:
        result = f'{self.variable.name}: [{self.mapping}] ref: {self.reference}'
        return result

    def __str__(self) -> str:
        result = f'{self.variable.name}: [{self.mapping}] ref: {self.reference}'
        return result


class OneSegmentation:
    """Single segmentation of a parameter"""

    def __init__(
        self,
        beta: biogeme.expressions.Beta,
        segmentation_tuple: DiscreteSegmentationTuple,
    ):
        """Ctor

        :param beta: parameter to be segmented
        :type beta: biogeme.expressions.Beta

        :param segmentation_tuple: characterization of the segmentation
        """
        self.beta: biogeme.expressions.Beta = beta
        self.variable: Variable = segmentation_tuple.variable
        self.reference: str = segmentation_tuple.reference
        self.mapping: dict[int, str] = {
            k: v for k, v in segmentation_tuple.mapping.items() if v != self.reference
        }

    def beta_name(self, category: str) -> str:
        """Construct the name of the parameter associated with a specific category

        :param category: name of the category
        :type category: str

        :return: name of parameter for the category
        :rtype: str

        :raise BiogemeError: if the category is not listed in the
            mapping of the segmentation.
        """
        if category not in self.mapping.values():
            error_msg = (
                f'Unknown category: {category}. List of known categories: '
                f'{self.mapping.values()}'
            )
            raise BiogemeError(error_msg)
        return f'{self.beta.name}_{category}'

    def beta_expression(self, category: str) -> biogeme.expressions.Beta:
        """Constructs the expression for the parameter associated with
            a specific category

        :param category: name of the category
        :type category: str

        :return: expression of the parameter for the category


        """
        name = self.beta_name(category)
        if category == self.reference:
            lower_bound = self.beta.lb
            upper_bound = self.beta.ub
        else:
            lower_bound = None
            upper_bound = None

        return Beta(
            name,
            self.beta.initValue,
            lower_bound,
            upper_bound,
            self.beta.status,
        )

    def beta_code(self, category: str, assignment: bool) -> str:
        """Constructs the Python code for the expression of the
            parameter associated with a specific category

        :param category: name of the category
        :type category: str

        :param assignment: if True, the code includes the assignment to a variable.
        :type assignment: bool

        :return: the Python code
        :rtype: str
        """
        if category == self.reference:
            lower_bound = self.beta.lb
            upper_bound = self.beta.ub
        else:
            lower_bound = None
            upper_bound = None
        name = self.beta_name(category)
        if assignment:
            return (
                f"{name} = Beta('{name}', {self.beta.initValue}, "
                f"{lower_bound}, {upper_bound}, {self.beta.status})"
            )
        return (
            f"Beta('{name}', {self.beta.initValue}, {lower_bound}, "
            f"{upper_bound}, {self.beta.status})"
        )

    def list_of_expressions(self) -> list[Expression]:
        """Create a list of expressions involved in the segmentation of the parameter

        :return: list of expressions
        :rtype: list(biogeme.expressions.Expression)

        """
        terms = [
            self.beta_expression(category) * (self.variable == Numeric(value))
            for value, category in self.mapping.items()
        ]
        return terms

    def list_of_code(self) -> list[str]:
        """Create a list of Python codes for the expressions involved
            in the segmentation of the parameter

        :return: list of codes
        :rtype: list(str)

        """
        return [
            (
                f"{self.beta_name(category)} "
                f"* (Variable('{self.variable.name}') == {value})"
            )
            for value, category in self.mapping.items()
        ]


class Segmentation:
    """Segmentation of a parameter, possibly with multiple socio-economic variables"""

    def __init__(
        self,
        beta: biogeme.expressions.Beta,
        segmentation_tuples: Iterable[DiscreteSegmentationTuple],
        prefix: str = 'segmented',
    ):
        """Ctor

        :param beta: parameter to be segmented
        :param segmentation_tuples: characterization of the segmentations
        :param prefix: prefix to be used to generated the name of the
            segmented parameter
        """
        self.beta: biogeme.expressions.Beta = beta
        self.segmentations: tuple[OneSegmentation, ...] = tuple(
            OneSegmentation(beta, s) for s in segmentation_tuples
        )
        self.prefix = prefix

    def beta_code(self) -> str:
        """Constructs the Python code for the parameter

        :return: Python code
        :rtype: str
        """

        beta_name = f"'{self.beta.name}'"
        return (
            f'Beta({beta_name}, {self.beta.initValue}, {self.beta.lb}, '
            f'{self.beta.ub}, {self.beta.status})'
        )

    def segmented_beta(self) -> Expression:
        """Create an expressions that combines all the segments

        :return: combined expression
        :rtype: biogeme.expressions.Expression

        """
        ref_beta = Beta(
            name=self.beta.name,
            value=self.beta.initValue,
            lowerbound=self.beta.lb,
            upperbound=self.beta.ub,
            status=self.beta.status,
        )
        terms = [ref_beta]
        terms += [
            element for s in self.segmentations for element in s.list_of_expressions()
        ]

        return bioMultSum(terms)

    def segmented_code(self) -> str:
        """Create the Python code for an expressions that combines all the segments

        :return: Python code for the combined expression
        :rtype: str
        """
        result = '\n'.join(
            [
                s.beta_code(c, assignment=True)
                for s in self.segmentations
                for c in s.mapping.values()
            ]
        )
        result += '\n'

        terms = [self.beta_code()]
        terms += [element for s in self.segmentations for element in s.list_of_code()]

        if len(terms) == 1:
            result += terms[0]
        else:
            joined_terms = ', '.join(terms)
            result += f'{self.prefix}_{self.beta.name} = bioMultSum([{joined_terms}])'
        return result


def segmented_beta(
    beta: biogeme.expressions.Beta,
    segmentation_tuples: Iterable[DiscreteSegmentationTuple],
    prefix: str = 'segmented',
):
    """Obtain the segmented Beta from a unique function call

    :param beta: parameter to be segmented
    :param segmentation_tuples: characterization of the segmentations
    :param prefix: prefix to be used to generated the name of the
        segmented parameter
    :return: expression of the segmented Beta
    """
    the_segmentation = Segmentation(
        beta=beta, segmentation_tuples=segmentation_tuples, prefix=prefix
    )
    return the_segmentation.segmented_beta()


@deprecated(new_func=segmented_beta)
def segment_parameter(
    beta: biogeme.expressions.Beta,
    segmentation_tuples: Iterable[DiscreteSegmentationTuple],
    prefix: str = 'segmented',
):
    pass
