"""Represents the configuration of a multiple expression

:author: Michel Bierlaire
:date: Sun Apr  2 14:15:17 2023

"""

import logging
from collections import namedtuple
import biogeme.exceptions as excep

logger = logging.getLogger(__name__)

SelectionTuple = namedtuple('SelectionTuple', 'catalog selection')

SEPARATOR = ';'
SELECTION_SEPARATOR = ':'


class Configuration:
    """Represents the configuration of a multiple expression. It is
    internally represented as a sorted list of tuples.

    """

    def __init__(self, list_of_selections=None):
        """Ctor.

        :param list_of_selections: list of tuples, where each of
            them associates a catalog name with the selected configuration
        :type list_of_selections: list(SelectionTuple)

        """
        if list_of_selections is None:
            self.__selections = None
        else:
            self.selections = list_of_selections

    @property
    def selections(self):
        return self.__selections

    @selections.setter
    def selections(self, value):
        self.__selections = sorted(value)
        self.__check_list_validity()
        self.string_id = self.get_string_id()

    @classmethod
    def from_string(cls, string_id):
        """Ctor from a string representation

        :param string_id: string ID
        :type string_id: str

        """
        terms = string_id.split(SEPARATOR)
        the_config = {}
        for term in terms:
            catalog, selection = term.split(SELECTION_SEPARATOR)
            the_config[catalog] = selection
        return cls.from_dict(the_config)

    @classmethod
    def from_dict(cls, dict_of_selections):
        """Ctor from dict

        :param dict_of_selections: dict associating a catalog name
            with the selected configuration
        :type dict_of_selections: dict(str: str)

        """
        the_list = (
            SelectionTuple(catalog=catalog, selection=selection)
            for catalog, selection in dict_of_selections.items()
        )
        return cls(list_of_selections=the_list)

    @classmethod
    def from_tuple_of_configurations(cls, tuple_of_configurations):
        """Ctor from tuple of configurations that are merged together.

        In the presence of two different selections for the same
        catalog, the first one is selected,
        and the others ignored.

        :param tuple_of_configurations: tuple of configurations to merge
        :type tuple_of_configurations: tuple(Configuration)

        """
        known_catalogs = set()
        list_of_selections = []
        for configuration in tuple_of_configurations:
            if configuration.selections is not None:
                for selection in configuration.selections:
                    if selection.catalog not in known_catalogs:
                        list_of_selections.append(selection)
                        known_catalogs.add(selection.catalog)
        return cls(list_of_selections)

    def set_of_catalogs(self):
        return {selection.catalog for selection in self.selections}

    def __eq__(self, other):
        return self.string_id == other.string_id

    def __hash__(self):
        return hash(self.string_id)

    def __repr__(self):
        return repr(self.string_id)

    def __str__(self):
        return str(self.string_id)

    def get_string_id(self):
        """The string ID is a unique string representation of the configuration

        :return: string ID
        :rtype: str
        """
        terms = [
            f'{selection.catalog}{SELECTION_SEPARATOR}{selection.selection}'
            for selection in self.selections
        ]
        return SEPARATOR.join(terms)

    def get_html(self):
        html = '<p>Specification</p><p><ul>\n'
        for selection_tuple in self.selections:
            html += (
                f'<li>{selection_tuple.catalog}: ' f'{selection_tuple.selection}</li>\n'
            )
        html += '</ul></p>\n'
        return html

    def get_selection(self, catalog_name):
        """Retrieve the selection of a given catalog

        :param catalog_name: name of the catalog
        :type catalog_name: str

        :return: name of the selected config, or None if catalog is not known
        :rtype: str
        """
        for selection in self.selections:
            if selection.catalog == catalog_name:
                return selection.selection
        return None

    def __check_list_validity(self):
        """Check the validity of the list.

        :raise biogemeError: if the same catalog appears more than once in the list
        """
        unique_items = set()
        for item in self.__selections:
            if item.catalog in unique_items:
                error_msg = (
                    f'Catalog {item.catalog} appears more than once in the '
                    f'configuration: {self.__selections}'
                )
                raise excep.biogemeError(error_msg)
            unique_items.add(item.catalog)
