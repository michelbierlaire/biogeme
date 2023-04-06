"""Defines  a catalog of expressions that may be considered in a specification

:author: Michel Bierlaire
:date: Fri Mar 31 08:54:07 2023

"""
import logging
from itertools import product
from biogeme.multiple_expressions import MultipleExpression, NamedExpression
import biogeme.exceptions as excep
import biogeme.expressions as ex
import biogeme.segmentation as seg
from biogeme.configuration import SelectionTuple, Configuration

logger = logging.getLogger(__name__)


class Catalog(MultipleExpression):
    """Catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically.
    """

    def __init__(self, name, tuple_of_named_expressions):
        """Ctor

        :param name: name of the catalog of expressions
        :type name: str

        :param tuple_of_named_expressions: tuple of NamedExpression,
            each containing a name and an expression.
        :type tuple_of_named_expressions: tuple(NamedExpression)

        :raise biogemeError: if tuple_of_named_expressions is empty

        """
        super().__init__(name)

        if not tuple_of_named_expressions:
            raise excep.biogemeError(
                f'{name}: cannot create a catalog from an empty tuple.'
            )

        self.tuple_of_named_expressions = tuple(
            NamedExpression(
                name=named.name, expression=ex.process_numeric(named.expression)
            )
            for named in tuple_of_named_expressions
        )

        # Check if the name of the catalog was not already used.
        for named_expression in self.tuple_of_named_expressions:
            if named_expression.expression.contains_catalog(self.name):
                error_msg = (
                    f'Catalog {self.name} cannot contain itself. Use different names'
                )
                raise excep.biogemeError(error_msg)

        self.current_index = 0
        self.dict_of_index = {
            expression.name: index
            for index, expression in enumerate(self.tuple_of_named_expressions)
        }
        self.synchronized_catalogs = []

    @classmethod
    def from_dict(cls, catalog_name, dict_of_expressions):
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

        """
        the_tuple = tuple(
            NamedExpression(name=name, expression=expression)
            for name, expression in dict_of_expressions.items()
        )
        return cls(catalog_name, the_tuple)

    def get_iterator(self):
        """Obtain an iterator on the named expressions"""
        return iter(self.tuple_of_named_expressions)

    def set_index(self, index):
        """Set the index of the selected expression, and update the
            synchronized catalogs

        :param index: value of the index
        :type index: int

        :raises biogemeError: if index is out of range

        """
        if index >= self.catalog_size():
            error_msg = (
                f'Wrong index {index}. ' f'Must be in [0, {self.catalog_size()}]'
            )
            raise excep.biogemeError(error_msg)
        self.current_index = index
        for sync_catalog in self.synchronized_catalogs:
            sync_catalog.current_index = index

    def set_of_configurations(self):
        """Set of possible configurations for a multiple
        expression. If the expression is simple, an empty set is
        returned.
        """
        results = set()
        for name, expression in self.get_iterator():
            this_selection = Configuration(
                [SelectionTuple(catalog=self.name, selection=name)]
            )
            this_set = expression.set_of_configurations()
            if this_set:
                the_updated_set = {
                    Configuration.from_tuple_of_configurations((conf, this_selection))
                    for conf in this_set
                }
                results |= the_updated_set
            else:
                results.add(this_selection)

        return results

    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression)
        """
        return self.tuple_of_named_expressions[self.current_index]

    def current_configuration(self):
        """Obtain the current configuration of an expression with Catalog

        :return: configuration

        :rtype: biogeme.configuration.Configuration
        """
        the_result = self.selected_expression().current_configuration()
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

        for _, expr in self.tuple_of_named_expressions:
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
        """
        total_modif = 0
        for _, e in self.tuple_of_named_expressions:
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
        for _, expression in self.tuple_of_named_expressions:
            total += expression.select_expression(group_name, index)
        return total


class SynchronizedCatalog(Catalog):
    """A catalog is synchronized when the selection of its expression
    is controlled by another catalog of the same length.

    """

    def __init__(self, name, tuple_of_named_expressions, controller):
        super().__init__(name, tuple_of_named_expressions)
        self.controller = controller
        if self.catalog_size() != controller.catalog_size():
            error_msg = (
                f'Catalog {name} contains {self.catalog_size()} expressions. '
                f'It must contain the same number of expressions '
                f'({controller.catalog_size()}) as its controller '
                f'({controller.name})'
            )
            raise excep.biogemeError(error_msg)
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

        :raises biogemeError: the index of the catalog cannot be
            changed directly. It must be changed by its controller

        """
        error_msg = (
            f'The index of catalog {self.name} cannot be changed directly. '
            f'It must be changed by its controller {self.controller.name}'
        )
        excep.biogemeError(error_msg)

    def current_configuration(self):
        """Obtain the current configuration of an expression with Catalog's

        :return: a dictionary such that the keys are the catalog
            names, and the values are the selected specification in
            the Catalog

        :rtype: dict(str: str)
        """
        the_result = self.selected_expression().current_configuration()
        return the_result

    def select_expression(self, group_name, index):
        """Select a specific expression in a group

        :param group_name: name of the group of expressions
        :type group_name: str

        :param index: index of the expression in the group
        :type index: int

        :raise biogemeError: if the group_name is a synchronized group.
        """
        if self.name == group_name:
            error_msg = (
                f'Group {self.name} is controlled by group {self.controller}. '
                f'It is not possible to select its expression independently. '
            )
            raise excep.biogemeError(error_msg)
        total = 0
        for _, expression in self.tuple_of_named_expressions:
            total += expression.select_expression(group_name, index)
        return total


def segmentation_catalog(
    beta_parameter, potential_segmentations, maximum_number, synchronized_with=None
):
    """Generate a catalog (possibly synchronized) for potential segmentations of a parameter

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

    list_of_possiblities = [
        p
        for p in product([False, True], repeat=len(potential_segmentations))
        if sum(p) <= maximum_number
    ]
    the_tuple_of_named_expressions = (
        NamedExpression(
            name=get_name_from_combination(combination),
            expression=get_expression_from_combination(combination),
        )
        for combination in list_of_possiblities
    )
    name = f'segmented_{beta_parameter.name}'
    if synchronized_with is None:
        return Catalog(name, the_tuple_of_named_expressions)

    return SynchronizedCatalog(name, the_tuple_of_named_expressions, synchronized_with)
