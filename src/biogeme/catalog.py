"""Defines  a catalog of expressions that may be considered in a specification

:author: Michel Bierlaire
:date: Fri Mar 31 08:54:07 2023

"""
import logging
from itertools import product
from biogeme.expressions import Beta
from biogeme.multiple_expressions import MultipleExpression, NamedExpression
import biogeme.exceptions as excep
import biogeme.expressions as ex
import biogeme.segmentation as seg
from biogeme.configuration import (
    SelectionTuple,
    Configuration,
)

logger = logging.getLogger(__name__)


class Catalog(MultipleExpression):
    """Catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically.
    """

    def __init__(self, name, list_of_named_expressions):
        """Ctor

        :param name: name of the catalog of expressions
        :type name: str

        :param list_of_named_expressions: list of NamedExpression,
            each containing a name and an expression.
        :type list_of_named_expressions: list(NamedExpression)

        :raise BiogemeError: if list_of_named_expressions is empty

        """
        super().__init__(name)

        if not list_of_named_expressions:
            raise excep.BiogemeError(
                f'{name}: cannot create a catalog from an empty list.'
            )

        # Transform numeric values into Biogeme expressions
        self.list_of_named_expressions = [
            NamedExpression(
                name=named.name, expression=ex.process_numeric(named.expression)
            )
            for named in list_of_named_expressions
        ]

        # Check if the name of the catalog was not already used.
        if any(
            named.expression.contains_catalog(self.name)
            for named in self.list_of_named_expressions
        ):
            error_msg = (
                f'Catalog {self.name} cannot contain itself. Use different names'
            )
            raise excep.BiogemeError(error_msg)

        self.current_index = 0
        self.dict_of_index = {
            expression.name: index
            for index, expression in enumerate(self.list_of_named_expressions)
        }
        self.synchronized_catalogs = []

    @classmethod
    def from_dict(cls, catalog_name, dict_of_expressions):
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

        """
        list_of_named_expressions = list(
            NamedExpression(name=name, expression=expression)
            for name, expression in dict_of_expressions.items()
        )
        return cls(catalog_name, list_of_named_expressions)

    def get_iterator(self):
        """Obtain an iterator on the named expressions"""
        return iter(self.list_of_named_expressions)

    def set_index(self, index):
        """Set the index of the selected expression, and update the
            synchronized catalogs

        :param index: value of the index
        :type index: int

        :raises BiogemeError: if index is out of range

        """
        if index >= self.catalog_size():
            error_msg = (
                f'Wrong index {index}. ' f'Must be in [0, {self.catalog_size()}]'
            )
            raise excep.BiogemeError(error_msg)
        self.current_index = index
        for sync_catalog in self.synchronized_catalogs:
            sync_catalog.current_index = index

    def set_of_configurations(self):
        """Set of possible configurations for a multiple
        expression. If the expression is simple, an empty set is
        returned. If the number of configurations exceeds the maximum
        length, None is returned.

        :param maximum_length: maximum length for complete enumeration
        :type maximum_length: int

        :return: set of configurations, or None
        :rtype: set(biogeme.configure.Configuration)
        """
        results = set()
        for name, expression in self.get_iterator():
            this_selection = Configuration(
                [SelectionTuple(catalog=self.name, selection=name)]
            )
            this_set = expression.set_of_configurations()
            if this_set is None:
                return None
            if this_set:
                the_updated_set = {
                    Configuration.from_tuple_of_configurations((conf, this_selection))
                    for conf in this_set
                }
                results |= the_updated_set
            else:
                results.add(this_selection)

            if len(results) > self.maximum_number_of_configurations:
                return None
        return results

    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: NamedExpression
        """
        return self.list_of_named_expressions[self.current_index]

    def selected_name(self):
        """Return the name of the selected expression

        :return: the name of the selected expression
        :rtype: str
        """
        return self.list_of_named_expressions[self.current_index].name

    def current_configuration(self, includes_controlled_catalogs=False):
        """Obtain the current configuration of an expression with Catalog

        :param includes_controlled_catalogs: if True, the controlled
            catalogs are included in the configuration. This is used
            mainly for debugging purposes.
        :type includes_controlled_catalogs: bool

        :return: configuration
        :rtype: biogeme.configuration.Configuration
        """
        the_result = self.selected_expression().current_configuration(
            includes_controlled_catalogs
        )
        current_selection = Configuration(
            [SelectionTuple(catalog=self.name, selection=self.selected_name())]
        )
        if the_result is None:
            return Configuration.from_tuple_of_configurations((current_selection,))
        return Configuration.from_tuple_of_configurations(
            (the_result, current_selection)
        )

    def configure_catalogs(self, configuration):
        """Select the items in each catalog corresponding to the
        requested configuration

        :param configuration: description of the configuration
        :type configuration: biogeme.configuration.Configuration

        """
        selection = configuration.get_selection(self.name)
        if selection is not None:
            self.set_index(self.dict_of_index[selection])

        for _, expr in self.list_of_named_expressions:
            expr.configure_catalogs(configuration)

    def modify_catalogs(self, set_of_catalogs, step, circular):
        """Modify the specification of several catalogs

        :param set_of_catalogs: set of catalogs to modify
        :type set_of_catalogs: set(str)

        :param step: increment of the modifications. Can be negative.
        :type step: int

        :param circular: If True, the modificiation is always made. If
            the selection needs to move past the last one, it comes
            back to the first one. For instance, if the catalog is
            currently at its last value, and the step is 1, it is set
            to its first value. If circular is False, and the
            selection needs to move past the last one, the selection
            is set to the last one. It works symmetrically if the step
            is negative
        :type circular: bool

        :return: number of actual modifications
        :rtype: int

        """
        total_modif = 0
        for _, e in self.list_of_named_expressions:
            total_modif += e.modify_catalogs(set_of_catalogs, step, circular)

        if self.name not in set_of_catalogs:
            return total_modif

        the_size = self.catalog_size()
        new_index = self.current_index + step
        if circular:
            self.set_index(new_index % the_size)
            return total_modif + step

        if new_index < 0:
            total_modif += self.current_index
            self.set_index(0)
            return total_modif

        if new_index >= the_size:
            total_modif += the_size - 1 - self.current_index
            self.set_index(the_size - 1)
            return total_modif

        self.set_index(new_index)
        return total_modif + step

    def reset_this_expression_selection(self):
        """In each group of expressions, select the first one"""
        self.set_index(0)

    def select_expression(self, group_name, index):
        """Select a specific expression in a group

        :param group_name: name of the group of expressions
        :type group_name: str

        :param index: index of the expression in the group
        :type index: int

        """
        total = 0
        if self.name == group_name:
            total = 1
            self.set_index(index)
        for _, expression in self.list_of_named_expressions:
            total += expression.select_expression(group_name, index)
        return total


class SynchronizedCatalog(Catalog):
    """A catalog is synchronized when the selection of its expression
    is controlled by another catalog of the same length.

    """

    def __init__(self, name, tuple_of_named_expressions, controller):
        super().__init__(name, tuple_of_named_expressions)
        if not isinstance(controller, Catalog):
            error_msg = (
                f'The controller of a synchronized catalog must be of type '
                f'Catalog, and not {type(controller)}'
            )
            raise excep.BiogemeError(error_msg)
        self.controller = controller
        if self.catalog_size() != controller.catalog_size():
            error_msg = (
                f'Catalog {name} contains {self.catalog_size()} expressions. '
                f'It must contain the same number of expressions '
                f'({controller.catalog_size()}) as its controller '
                f'({controller.name})'
            )
            raise excep.BiogemeError(error_msg)
        self.controller.synchronized_catalogs.append(self)

    @classmethod
    def from_dict(cls, catalog_name, dict_of_expressions, controller):
        """Ctor using a dict instead of a tuple.

        Python does not guarantee the order of elements of a dict,
        although, in practice, it is always preserved. If the order is
        critical, it is better to use the main constructor. If not,
        this constructor provides a more readable code.

        :param catalog_name: name of the catalog
        :type catalog_name: str

        :param dict_of_expressions: dict associating the name of an
            expression and the expression itself.
        :type dict_of_expressions: dict(str:biogeme.expressions.Expression)

        :param controller: catalog that controls this one.
        :type controller: Catalog
        """
        the_tuple = tuple(
            NamedExpression(name=name, expression=expression)
            for name, expression in dict_of_expressions.items()
        )
        return cls(catalog_name, the_tuple, controller)

    def set_of_configurations(self):
        """Set of possible configurations for a multiple
        expression. If the expression is simple, an empty set is
        returned.
        """
        results = set()
        for _, expression in self.get_iterator():
            the_set = expression.set_of_configurations()
            results |= the_set
        return results

    def set_index(self, index):
        """Set the index of the selected expression, and update the
        synchronized catalogs

        :param index: value of the index
        :type index: int

        :raises BiogemeError: the index of the catalog cannot be
            changed directly. It must be changed by its controller

        """
        error_msg = (
            f'The index of catalog {self.name} cannot be changed directly. '
            f'It must be changed by its controller {self.controller.name}'
        )
        raise excep.BiogemeError(error_msg)

    def current_configuration(self, includes_controlled_catalogs=False):
        """Obtain the current configuration of an expression with Catalog

        :param includes_controlled_catalogs: if True, the controlled
            catalogs are included in the configuration. This is used
            mainly for debugging purposes.
        :type includes_controlled_catalogs: bool

        :return: configuration
        :rtype: biogeme.configuration.Configuration
        """
        the_result = self.selected_expression().current_configuration(
            includes_controlled_catalogs
        )
        if not includes_controlled_catalogs:
            return the_result
        current_selection = Configuration(
            [SelectionTuple(catalog=self.name, selection=self.selected_name())]
        )
        if the_result is None:
            return Configuration.from_tuple_of_configurations((current_selection,))
        return Configuration.from_tuple_of_configurations(
            (the_result, current_selection)
        )

    def select_expression(self, group_name, index):
        """Select a specific expression in a group

        :param group_name: name of the group of expressions
        :type group_name: str

        :param index: index of the expression in the group
        :type index: int

        :raise BiogemeError: if the group_name is a synchronized group.
        """
        if self.name == group_name:
            error_msg = (
                f'Group {self.name} is controlled by group {self.controller}. '
                f'It is not possible to select its expression independently. '
            )
            raise excep.BiogemeError(error_msg)
        total = 0
        for _, expression in self.list_of_named_expressions:
            total += expression.select_expression(group_name, index)
        return total


def segmentation_catalog(
    beta_parameter,
    potential_segmentations,
    maximum_number,
    synchronized_with=None
):
    """Generate a catalog (possibly synchronized) for potential
        segmentations of a parameter

    :param beta_parameter: parameter to be segmented
    :type beta_parameter: biogeme.expressions.Beta

    :param potential_segmentations: tuple of potential segmentations
    :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

    :param maximum_number: maximum number of segmentations to consider
    :type maximum_number: int

    :param synchronized_with: Catalog that controls this one. If None, no control.
    :type synchronized_with: Catalog

    """

    def get_name_from_combination(combination):
        """Assign a name to a combination"""
        if sum(combination) == 0:
            return f'{beta_parameter.name} (no seg.)'

        return '-'.join(
            [
                segment.variable.name
                for keep, segment in zip(combination, potential_segmentations)
                if keep
            ]
        )

    def get_expression_from_combination(combination):
        """Assign an expression to a combination"""
        selected_expressions = (
            segment
            for keep, segment in zip(combination, potential_segmentations)
            if keep
        )
        the_segmentation = seg.Segmentation(beta_parameter, selected_expressions)
        return the_segmentation.segmented_beta()

    list_of_possibilities = [
        combination
        for combination in product([False, True], repeat=len(potential_segmentations))
        if sum(combination) <= maximum_number
    ]
    the_tuple_of_named_expressions = (
        NamedExpression(
            name=get_name_from_combination(combination),
            expression=get_expression_from_combination(combination),
        )
        for combination in list_of_possibilities
    )
    name = f'segmented_{beta_parameter.name}'
    if synchronized_with is None:
        return Catalog(name, the_tuple_of_named_expressions)

    return SynchronizedCatalog(name, the_tuple_of_named_expressions, synchronized_with)


def generic_alt_specific_catalog(
    coefficient,
    alternatives,
    potential_segmentations=None,
    maximum_number=5,
):
    """Generate catalogs selecting generic or alternative specific coefficients

    :param coefficient: coefficient of interest
    :type coefficient: biogeme.expressions.Beta

    :param alternatives: names of the alternatives
    :type alternatives: tuple(str)

    :param potential_segmentations: tuple of potential segmentations, or None
    :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

    :param maximum_number: maximum number of segmentations to consider
    :type maximum_number: int

    :return: a catalog for each alternative
    :rtype: dict(str: biogeme.catalog.Catalog)
    """
    if not isinstance(coefficient, Beta):
        error_msg = 'This function must be called with a Beta object'
        raise excep.BiogemeError(error_msg)

    if len(alternatives) < 2:
        error_msg = (
            f'An alternative specific specification requires at least 2 '
            f'alternatives, and not {len(alternatives)}'
        )
        raise excep.BiogemeError(error_msg)

    generic_coeff = (
        coefficient
        if potential_segmentations is None
        else segmentation_catalog(
            beta_parameter=coefficient,
            potential_segmentations=potential_segmentations,
            maximum_number=maximum_number,
            synchronized_with=None,
        )
    )

    altspec_coeffs = {}
    controller = None
    for alternative in alternatives:
        the_beta = Beta(
            name=f'{coefficient.name}_{alternative}',
            value=coefficient.initValue,
            lowerbound=coefficient.lb,
            upperbound=coefficient.ub,
            status=coefficient.status,
        )
        if controller is None:
            the_coeff = (
                the_beta
                if potential_segmentations is None
                else segmentation_catalog(
                        beta_parameter=the_beta,
                        potential_segmentations=potential_segmentations,
                        maximum_number=maximum_number,
                )
            )
            controller = the_coeff
        else:
            the_coeff = (
                the_beta
                if potential_segmentations is None
                else segmentation_catalog(
                        beta_parameter=the_beta,
                        potential_segmentations=potential_segmentations,
                        maximum_number=maximum_number,
                        synchronized_with=controller,
                )
            )
        altspec_coeffs[alternative] = the_coeff

    results = {}
    controller = None
    for alternative in alternatives:
        catalog_name = f'_{coefficient.name}_{alternative}_gen_alt_spec'
        expressions = {
            'generic': coefficient,
            'altspec': altspec_coeffs[alternative],
        }
        if controller is None:
            results[alternative] = Catalog.from_dict(
                catalog_name=catalog_name,
                dict_of_expressions=expressions,
            )
            controller = results[alternative]
        else:
            results[alternative] = SynchronizedCatalog.from_dict(
                catalog_name=catalog_name,
                dict_of_expressions=expressions,
                controller=controller,
            )

    return results
