"""Defines  a catalog of expressions that may be considered in a specification

:author: Michel Bierlaire
:date: Fri Mar 31 08:54:07 2023

"""
from itertools import product
from biogeme.multiple_expressions import MultipleExpression, NamedExpression
import biogeme.exceptions as excep
import biogeme.expressions as ex
import biogeme.segmentation as seg

SEPARATOR = ';'
SELECTION_SEPARATOR = ':'

def configuration_to_string_id(configuration):
    """Transforms a configuration into a string ID

    :param configuration: dict where the keys are the catalog names,
       and the values are the selected configuration.
    :type configuration: dict(str: str)

    :return: string ID
    :rtype: str
    """
    terms = [f'{k}{SELECTION_SEPARATOR}{v}' for k, v in configuration.items()]
    return SEPARATOR.join(terms)


def string_id_to_configuration(string_id):
    """Transforms a string ID into a configuration

    :param string_id: string ID
    :type string_id: str

    :return: dict where the keys are the catalog names,
       and the values are the selected configuration.
    :rtype: dict(str: str)
    """
    terms = string_id.split(SEPARATOR)
    the_config = {}
    for term in terms:
        catalog, selection = term.split(SELECTION_SEPARATOR)
        the_config[catalog] = selection
    return the_config


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
        """ Obtain an iterator on the names expressions """
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

    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression)
        """
        return self.tuple_of_named_expressions[self.current_index]

    def increment_selection(self):
        """Increment recursively the selection of multiple
        expressions.

        :return: True if the increment has been implemented
        :rtype: bool
        """
        if self.selected_expression().increment_selection():
            return True
        if self.current_index == self.catalog_size() - 1:
            return False
        self.set_index(self.current_index + 1)
        self.selected_expression().reset_expression_selection()
        for sync_catalog in self.synchronized_catalogs:
            sync_catalog.selected_expression().reset_expression_selection()

        return True

    def current_configuration(self):
        """Obtain the current configuration of an expression with Catalog's

        :return: a dictionary such that the keys are the catalog
            names, and the values are the selected specification in
            the Catalog

        :rtype: dict(str: str)
        """
        the_result = self.selected_expression().current_configuration()
        the_result[self.name] = self.selected_name()
        return the_result
    
    def increment_catalog(self, catalog_name, step):
        """Increment the selection of a specific catalog

        :param catalog_name: name of the catalog to change
        :type catalog_name: str

        :param step: number of increments to apply. Can be negative.
        :type step: int
        """
        if step == 0:
            return
        if self.name == catalog_name:
            if (
                self.current_index + step >= self.catalog_size()
                or self.current_index + step < 0
            ):
                raise excep.valueOutOfRange
            self.current_index += step
            return

        for _, expr in self.tuple_of_named_expressions:
            expr.increment_catalog(catalog_name, step)

    def configure_catalogs(self, configuration):
        """Select the items in each catalog corresponding to the
            requested configuration

        :param configuration: a dictionary such that the keys are the catalog
            names, and the values are the selected specification in
            the Catalog
        :type configuration: dict(str: str)

        """
        selection = configuration.get(self.name)
        if selection is not None:
            self.current_index = self.dict_of_index[selection]

        for _, expr in self.tuple_of_named_expressions:
            expr.configure_catalogs(configuration)


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

    def increment_selection(self):
        """Increment recursively the selection of multiple
        expressions.

        :return: True if the increment has been implemented
        :rtype: bool
        """
        return self.selected_expression().increment_selection()

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

class SegmentationCatalog(Catalog):
    """ Possible segmentations of a parameter """

    def __init__(self, beta_parameter, potential_segmentations, maximum_number=None):
        """ Ctor
        
        :param beta_parameter: parameter to be segmented
        :type beta_parameter: biogeme.expressions.Beta

        :param potential_segmentations: tuple of potential segmentations
        :type potential_segmentations: tuple(biogeme.segmentation.DiscreteSegmentationTuple)

        :param maximum_number: maximum number of segmentations to consider
        :type maximum_number: int
        """
        self.beta_parameter = beta_parameter
        self.potential_segmentations = potential_segmentations
        if maximum_number is None:
            maximum_number = len(potential_segmentations)
        self.list_of_possiblities = [
            p
            for p in product([False, True], repeat=len(potential_segmentations))
            if sum(p) <= maximum_number
        ]
        the_tuple_of_named_expressions = (
            NamedExpression(
                name=self.get_name_from_config(config),
                expression=self.get_expression_from_config(config)
            )
            for config in self.list_of_possiblities
        )
        name = f'segmented_{beta_parameter.name}'
        super().__init__(name, the_tuple_of_named_expressions)

    def get_name_from_config(self, config):
        """Assign a name to a config
        """
        if sum(config) == 0:
            return f'{self.beta_parameter.name} (no seg.)'

        return '-'.join(
            [
                segment.variable.name
                for keep, segment in zip(config, self.potential_segmentations)
                if keep
            ]
        )

    def get_expression_from_config(self, config):
        """Assign an expression to a config
        """
        selected_expressions = (
                segment
                for keep, segment in zip(config, self.potential_segmentations)
                if keep
        )
        the_segmentation = seg.Segmentation(self.beta_parameter, selected_expressions)
        return the_segmentation.segmented_beta()
