"""Defines controllers for multiple expressions.

:author: Michel Bierlaire
:date: Sun Jul 16 15:23:46 2023

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Callable

from biogeme.expressions import SELECTION_SEPARATOR
from biogeme.exceptions import BiogemeError
from .configuration import (
    Configuration,
)

if TYPE_CHECKING:
    from biogeme.catalog import Catalog

ControllerOperator = Callable[
    [Configuration, int],
    tuple[Configuration, int],
]

logger = logging.getLogger(__name__)


class Controller:
    """Class controlling the specification of the Catalogs"""

    def __init__(self, controller_name: str, specification_names: Iterable[str]):
        """Constructor

        :param controller_name: name of the controller
        :type controller_name: str

        :param specification_names: list or tuple of the names of the
            specification controlled by the controller
        :type specification_names: list(str) or tuple(str)
        """
        self.controller_name: str = controller_name
        self.specification_names: tuple[str, ...] = tuple(specification_names)
        self.current_index: int = 0
        self.dict_of_index: dict[str, int] = {
            name: index for index, name in enumerate(self.specification_names)
        }
        self.controlled_catalogs: list[Catalog] = []

    def all_configurations(self) -> set[str]:
        """Return the code of all configurations

        :return: set of codes
        :rtype: set(str)
        """
        return {
            f"{self.controller_name}{SELECTION_SEPARATOR}{specification}"
            for specification in self.specification_names
        }

    def __eq__(self, other: Controller) -> bool:
        return self.controller_name == other.controller_name

    def __lt__(self, other: Controller) -> bool:
        return self.controller_name < other.controller_name

    def __hash__(self) -> int:
        return hash(self.controller_name)

    def __repr__(self) -> str:
        return repr(self.controller_name)

    def __str__(self) -> str:
        return f"{self.controller_name}: {self.specification_names}"

    def controller_size(self) -> int:
        """Number of specifications managed by this controller"""
        return len(self.specification_names)

    def current_name(self) -> str:
        """Name of the currently selected expression"""
        return self.specification_names[self.current_index]

    def set_name(self, name: str) -> None:
        """Set the index of the controller based on the name of the specification

        :param name: name of the specification
        :type name: str
        """
        the_index = self.dict_of_index.get(name)
        if the_index is None:
            error_msg = (
                f"{name}: unknown specification for controller {self.controller_name}"
            )
            raise BiogemeError(error_msg)
        self.set_index(the_index)

    def set_index(self, index: int) -> None:
        """Set the index of the controller, and update the controlled catalogs

        :param index: value of the index
        :type index: int

        :raises BiogemeError: if index is out of range
        """
        if index < 0 or index >= self.controller_size():
            error_msg = (
                f"Wrong index {index} for controller {self.controller_name}. "
                f"Must be in [0, {self.controller_size()}]"
            )
            raise BiogemeError(error_msg)

        self.current_index = index
        for catalog in self.controlled_catalogs:
            catalog.current_index = index

    def reset_selection(self) -> None:
        """Select the first specification"""
        self.set_index(0)

    def modify_controller(self, step: int, circular: bool) -> int:
        """Modify the specification of the controller

        :param step: increment of the modifications. Can be negative.
        :type step: int

        :param circular: If True, the modification is always made. If
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
        the_size = self.controller_size()
        new_index = self.current_index + step
        if circular:
            self.set_index(new_index % the_size)
            return step

        if new_index < 0:
            total_modif = self.current_index
            self.set_index(0)
            return total_modif

        if new_index >= the_size:
            total_modif = the_size - 1 - self.current_index
            self.set_index(the_size - 1)
            return total_modif

        self.set_index(new_index)
        return step
