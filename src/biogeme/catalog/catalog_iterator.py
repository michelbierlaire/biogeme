"""Iterator on arithmetic expressions in a catalog

Michel Bierlaire
Thu Apr 17 2025, 08:30:58
"""

from __future__ import annotations

from biogeme.exceptions import BiogemeError

from .central_controller import CentralController
from .configuration import Configuration


class SelectedConfigurationsIterator:
    """A multiple expression is an expression that contains
    Catalog. This iterator loops on pre-specified configurations
    """

    def __init__(
        self,
        the_central_controller: CentralController,
        selected_configurations: set[Configuration] | None = None,
    ):
        """Ctor.

        :param the_central_controller: expression containing Catalogs
        :param selected_configurations: selected configurations to iterate on. If None, all configurations are considered.
        """
        self.the_central_controller = the_central_controller
        self.configurations = (
            selected_configurations or the_central_controller.all_configurations
        )
        if not isinstance(self.configurations, set):
            error_msg = f'The selected configurations must be a set, and not an object of type {type(self.configurations)}'
            raise BiogemeError(error_msg)
        self.set_iterator = iter(self.configurations)
        self.current_configuration = next(self.set_iterator)
        self.the_central_controller.set_configuration(self.current_configuration)
        self.first = True
        self.number = 0

    def __iter__(self) -> SelectedConfigurationsIterator:
        return self

    def __next__(self) -> Configuration:
        self.number += 1
        if self.first:
            self.first = False
            return self.current_configuration

        self.current_configuration = next(self.set_iterator)
        self.the_central_controller.set_configuration(self.current_configuration)
        return self.current_configuration
