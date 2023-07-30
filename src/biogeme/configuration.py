"""Represents the configuration of a multiple expression

:author: Michel Bierlaire
:date: Sun Apr  2 14:15:17 2023

"""

import logging
from typing import NamedTuple
import biogeme.exceptions as excep

logger = logging.getLogger(__name__)


class SelectionTuple(NamedTuple):
    controller: str
    selection: str


SEPARATOR = ';'
SELECTION_SEPARATOR = ':'


class Configuration:
    """Represents the configuration of a multiple expression. It is
    internally represented as a sorted list of tuples.

    """

    def __init__(self, selections=None):
        """Ctor.

        :param selections: list of tuples, where each of
            them associates a controller name with the selected configuration
        :type list_of_selections: list(SelectionTuple)

        """
        if selections is None:
            self.__selections = None
        else:
            self.selections = selections

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
            try:
                controller, selection = term.split(SELECTION_SEPARATOR)
            except ValueError as exc:
                error_msg = (
                    f'{exc}: Invalid syntax for ID {term}. Expecting a separator '
                    f'[{SELECTION_SEPARATOR}]'
                )
                raise excep.BiogemeError(error_msg)

            the_config[controller] = selection
        return cls.from_dict(the_config)

    @classmethod
    def from_dict(cls, dict_of_selections):
        """Ctor from dict

        :param dict_of_selections: dict associating a catalog name
            with the selected configuration
        :type dict_of_selections: dict(str: str)

        """
        the_list = (
            SelectionTuple(controller=controller, selection=selection)
            for controller, selection in dict_of_selections.items()
        )
        return cls(selections=the_list)

    @classmethod
    def from_tuple_of_configurations(cls, tuple_of_configurations):
        """Ctor from tuple of configurations that are merged together.

        In the presence of two different selections for the same
        catalog, the first one is selected,
        and the others ignored.

        :param tuple_of_configurations: tuple of configurations to merge
        :type tuple_of_configurations: tuple(Configuration)

        """
        known_controllers = set()
        selections = []
        for configuration in tuple_of_configurations:
            if configuration.selections is not None:
                for selection in configuration.selections:
                    if selection.controller not in known_controllers:
                        selections.append(selection)
                        known_controllers.add(selection.controller)
        return cls(selections)

    def set_of_controllers(self):
        return {selection.controller for selection in self.selections}

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
            f'{selection.controller}{SELECTION_SEPARATOR}{selection.selection}'
            for selection in self.selections
        ]
        return SEPARATOR.join(terms)

    def get_html(self):
        html = '<p>Specification</p><p><ul>\n'
        for selection_tuple in self.selections:
            html += (
                f'<li>{selection_tuple.controller}: '
                f'{selection_tuple.selection}</li>\n'
            )
        html += '</ul></p>\n'
        return html

    def get_selection(self, controller_name):
        """Retrieve the selection of a given controller

        :param controller_name: name of the controller
        :type controller_name: str

        :return: name of the selected config, or None if controller is not known
        :rtype: str
        """
        for selection in self.selections:
            if selection.controller == controller_name:
                return selection.selection
        return None

    def __check_list_validity(self):
        """Check the validity of the list.

        :raise BiogemeError: if the same catalog appears more than once in the list
        """
        unique_items = set()
        for item in self.__selections:
            if item.controller in unique_items:
                error_msg = (
                    f'Controller {item.controller} appears more than once in the '
                    f'configuration: {self.__selections}'
                )
                raise excep.BiogemeError(error_msg)
            unique_items.add(item.controller)
