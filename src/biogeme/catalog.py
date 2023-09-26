"""Defines  a catalog of expressions that may be considered in a specification

:author: Michel Bierlaire
:date: Fri Mar 31 08:54:07 2023

"""
import logging
from itertools import product
from biogeme.expressions import Beta
from biogeme.expressions import MultipleExpression, NamedExpression
import biogeme.exceptions as excep
from biogeme.controller import Controller
import biogeme.expressions as ex
import biogeme.segmentation as seg
from biogeme.configuration import (
    SEPARATOR,
    SELECTION_SEPARATOR,
)

logger = logging.getLogger(__name__)


class Catalog(MultipleExpression):
    """Catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically by a controller.
    """

    def __init__(self, catalog_name, named_expressions, controlled_by=None):
        """Ctor

        :param name: name of the catalog of expressions
        :type name: str

        :param list_of_named_expressions: list of NamedExpression,
            each containing a name and an expression.
        :type list_of_named_expressions: list(NamedExpression)

        :param controlled_by: Object controlling the selection of the specifications.
        :type controlled_by: Controller

        :raise BiogemeError: if list_of_named_expressions is empty
        :raise BiogemeError: if incompatible Controller

        """
        super().__init__(catalog_name)

        if not named_expressions:
            raise excep.BiogemeError(
                f'{catalog_name}: cannot create a catalog from an empty list.'
            )

        if controlled_by and not isinstance(controlled_by, Controller):
            error_msg = (
                f'The controller must be of type Controller and not '
                f'{type(controlled_by)}'
            )
            raise excep.BiogemeError(error_msg)

        self.named_expressions = [
            NamedExpression(
                name=named.name, expression=ex.process_numeric(named.expression)
            )
            for named in named_expressions
        ]

        # Check if the name of the catalog was not already used.
        if any(
            named.expression.contains_catalog(self.name)
            for named in self.named_expressions
        ):
            error_msg = (
                f'Catalog {self.name} cannot contain itself. Use different names'
            )
            raise excep.BiogemeError(error_msg)

        # Declare the expressions as children of the catalog
        for _, expression in self.named_expressions:
            self.children.append(expression)

        names = [named_expr.name for named_expr in self.named_expressions]
        if controlled_by is None:
            controller_name = catalog_name
            self.controlled_by = Controller(
                controller_name=controller_name, specification_names=names
            )
        else:
            self.controlled_by = controlled_by
            controller_names = list(controlled_by.specification_names)
            if names != controller_names:
                error_msg = (
                    f'Incompatible IDs between catalog [{names}] and controller '
                    f'[{controller_names}]'
                )
                raise excep.BiogemeError(error_msg)

    def get_all_controllers(self):
        """Provides all controllers controlling the specifications of
            a multiple expression

        :return: a set of controllers
        :rtype: set(biogeme.controller.Controller)

        """
        all_controllers = {self.controlled_by}
        for e in self.children:
            all_controllers |= e.get_all_controllers()
        return all_controllers

    @classmethod
    def from_dict(cls, catalog_name, dict_of_expressions, controlled_by=None):
        """Ctor using a dict instead of a list.

        Python does not guarantee the order of elements of a dict,
        although, in practice, it is always preserved. If the order is
        critical, it is better to use the main constructor. If not,
        this constructor provides a more readable code.

        :param catalog_name: name of the catalog
        :type catalog_name: str

        :param dict_of_expressions: dict associating the name of an
            expression and the expression itself.
        :type dict_of_expressions: dict(str:biogeme.expressions.Expression)

        :param controlled_by: Object controlling the selection of the specifications.
        :type controlled_by: Controller

        """
        named_expressions = [
            NamedExpression(name=name, expression=expression)
            for name, expression in dict_of_expressions.items()
        ]
        return cls(
            catalog_name=catalog_name,
            named_expressions=named_expressions,
            controlled_by=controlled_by,
        )

    def catalog_size(self) -> int:
        """Return the size of the catalog."""
        return len(self.named_expressions)

    def get_iterator(self):
        """Obtain an iterator on the named expressions"""
        return iter(self.named_expressions)

    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: NamedExpression
        """
        return self.named_expressions[self.controlled_by.current_index]

    def selected_name(self):
        """Return the name of the selected expression

        :return: the name of the selected expression
        :rtype: str
        """
        return self.named_expressions[self.controlled_by.current_index].name


def segmentation_catalogs(
    generic_name, beta_parameters, potential_segmentations, maximum_number
):
    """Generate catalogs for potential segmentations of a parameter

    :param generic_name: name used for the definition of the group of catalogs
    :type generic_name: str

    :param beta_parameters: list of parameters to be segmented
    :type beta_parameters: list(biogeme.expressions.Beta)

    :param potential_segmentations: tuple of potential segmentations
    :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

    :param maximum_number: maximum number of segmentations to consider
    :type maximum_number: int

    """

    for segmentation in potential_segmentations:
        for key, value in segmentation.mapping.items():
            if SEPARATOR in value or SELECTION_SEPARATOR in value:
                error_msg = (
                    f'Invalid segment name for variable {segmentation.variable.name}='
                    f'{key}: [{value}]. Characters [{SEPARATOR}] and '
                    f'[{SELECTION_SEPARATOR}] are reserved for specification coding.'
                )
                raise excep.BiogemeError(error_msg)

    def get_name_from_combination(combination):
        """Assign a name to a combination"""
        if sum(combination) == 0:
            return 'no_seg'

        return '-'.join(
            [
                segment.variable.name
                for keep, segment in zip(combination, potential_segmentations)
                if keep
            ]
        )

    def get_expression_from_combination(beta_parameter, combination):
        """Assign an expression to a combination"""
        selected_expressions = (
            segment
            for keep, segment in zip(combination, potential_segmentations)
            if keep
        )
        the_segmentation = seg.Segmentation(beta_parameter, selected_expressions)
        return the_segmentation.segmented_beta()

    if not isinstance(beta_parameters, list):
        error_msg = (
            f'A list is expected for beta_parameters, and not an object of type '
            f'{type(beta_parameters)}'
        )
        raise excep.BiogemeError(error_msg)

    list_of_possibilities = [
        combination
        for combination in product([False, True], repeat=len(potential_segmentations))
        if sum(combination) <= maximum_number
    ]
    catalogs = []
    names = [
        get_name_from_combination(combination) for combination in list_of_possibilities
    ]
    the_controller = Controller(controller_name=generic_name, specification_names=names)
    for beta_parameter in beta_parameters:
        named_expressions = [
            NamedExpression(
                name=get_name_from_combination(combination),
                expression=get_expression_from_combination(beta_parameter, combination),
            )
            for combination in list_of_possibilities
        ]
        name = f'segmented_{beta_parameter.name}'
        catalog = Catalog(
            catalog_name=name,
            named_expressions=named_expressions,
            controlled_by=the_controller,
        )
        catalogs.append(catalog)
    return catalogs


class SegmentedParameters:
    """Class managing the names of segmented and alternative specific parameters"""

    def __init__(self, beta_parameters, alternatives):
        """Constructor"""

        # The parameters are organized as follows:
        #   - all generic parameters,
        #   - all parameters associated with the first alternative,
        #   - all parameters associated with the second alternative,
        #   - etc.
        self.beta_parameters = beta_parameters
        self.all_parameters = beta_parameters.copy()
        self.alternatives = alternatives
        for alternative in self.alternatives:
            self.all_parameters += [
                Beta(
                    name=f'{beta.name}_{alternative}',
                    value=beta.initValue,
                    lowerbound=beta.lb,
                    upperbound=beta.ub,
                    status=beta.status,
                )
                for beta in beta_parameters
            ]

    def get_index(self, beta_index, alternative):
        """Returns the index in the list of the beta parameter with
            the given index specific to the given alternative

        :param beta_index: index of the beta in the generic list
        :type beta_index: int

        :param alternative: name of the alternative, or None for the generic parameter
        :type alternative: str or None

        """
        if alternative is None:
            return beta_index
        alt_index = self.alternatives.index(alternative)
        return beta_index + (alt_index + 1) * len(self.beta_parameters)

    def get_beta(self, beta_index, alternative):
        """Return the beta parameters for the given index and given alternative

        :param beta_index: index of the beta in the generic list
        :type beta_index: int

        :param alternative: name of the alternative, or None for the generic parameter
        :type alternative: str or None

        """
        return self.all_parameters[self.get_index(beta_index, alternative)]


def generic_alt_specific_catalogs(
    generic_name,
    beta_parameters,
    alternatives,
    potential_segmentations=None,
    maximum_number=5,
):
    """Generate catalogs selecting generic or alternative specific coefficients

    :param generic_name: name associated with all the parameters in the catalog
    :type generic_name: str

    :param beta_parameters: coefficients of interest
    :type beta_parameters: list(biogeme.expressions.Beta)

    :param alternatives: names of the alternatives
    :type alternatives: tuple(str)

    :param potential_segmentations: tuple of potential segmentations, or None
    :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

    :param maximum_number: maximum number of segmentations to consider
    :type maximum_number: int

    :return: a list of catalogs for each alternative
    :rtype: list(dict(str: biogeme.catalog.Catalog))
    """
    if len(alternatives) < 2:
        error_msg = (
            f'An alternative specific specification requires at least 2 '
            f'alternatives, and not {len(alternatives)}'
        )
        raise excep.BiogemeError(error_msg)

    if not isinstance(beta_parameters, list):
        error_msg = (
            f'Argument "beta_parameters" of function '
            f'"{generic_alt_specific_catalogs.__name__}" must be a list.'
        )
        raise excep.BiogemeError(error_msg)

    wrong_indices = []
    for index, beta in enumerate(beta_parameters):
        if not isinstance(beta, Beta):
            wrong_indices.append(index)

    if wrong_indices:
        error_msg = (
            f'The entries at the following indices are not Beta expressions: '
            f'{wrong_indices}'
        )
        raise excep.BiogemeError(error_msg)

    # We first generate the alternative specific versions of the parameters
    generic_parameters = beta_parameters
    the_segmented_parameters = SegmentedParameters(
        beta_parameters=generic_parameters,
        alternatives=alternatives,
    )

    # If applicable, we apply the potential segmentations
    if potential_segmentations:
        segmented_catalogs = segmentation_catalogs(
            generic_name=generic_name,
            beta_parameters=the_segmented_parameters.all_parameters,
            potential_segmentations=potential_segmentations,
            maximum_number=maximum_number,
        )

    def get_expression(param_index, alternative):
        """Returns either the parameter, or the segmented version if applicable"""

        if potential_segmentations:
            index = the_segmented_parameters.get_index(param_index, alternative)
            return segmented_catalogs[index]
        return the_segmented_parameters.get_beta(param_index, alternative)

    # We now control for generic or alternative specific with a single
    # controller for all catalogs
    the_controller = Controller(
        controller_name=f'{generic_name}_gen_altspec',
        specification_names=('generic', 'altspec'),
    )

    # We organize the catalogs as a list of dict
    results = []
    for index, beta in enumerate(beta_parameters):
        the_dict = {
            alternative: Catalog.from_dict(
                catalog_name=f'{beta.name}_{alternative}_gen_altspec',
                dict_of_expressions={
                    'generic': get_expression(index, None),
                    'altspec': get_expression(index, alternative),
                },
                controlled_by=the_controller,
            )
            for alternative in alternatives
        }
        results.append(the_dict)
    return results
