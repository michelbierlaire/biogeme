"""Defines central controller for multiple expressions.

Michel Bierlaire
15.04.2025 09:43
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from functools import reduce
from itertools import product
from typing import Any, Callable

import biogeme
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, ExpressionCollector, NamedExpression
from biogeme.parameters import get_default_value
from .catalog import Catalog
from .configuration import Configuration, SEPARATOR, SelectionTuple
from .controller import Controller

ControllerOperator = Callable[
    [Configuration, int],
    tuple[Configuration, int],
]

logger = logging.getLogger(__name__)


def extract_multiple_expressions_controllers(
    the_expression: Expression,
) -> list[Controller]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Catalog)
    def retrieve_controllers(
        expr: Catalog, context: Any | None = None
    ) -> list[Controller]:
        return [expr.controlled_by]

    # Now use it
    return walker.walk(the_expression)


def count_number_of_specifications(expression: Expression):
    central_controller = CentralController(expression=expression)
    return central_controller.number_of_configurations()


class CentralController:
    """Class controlling the complete multiple expression"""

    def __init__(
        self,
        expression: Expression,
        maximum_number_of_configurations: int | None = None,
    ):
        """Constructor

        :param expression: controllers expression
        :type expression: biogeme.expressions.Expression

        """
        set_of_controllers = set(
            extract_multiple_expressions_controllers(the_expression=expression)
        )

        self.controllers: tuple[Controller, ...] = tuple(sorted(set_of_controllers))
        self.dict_of_controllers = {
            controller.controller_name: controller for controller in self.controllers
        }
        self.expression = expression

        all_controllers_states = [
            the_controller.all_configurations() for the_controller in self.controllers
        ]

        # The total number of configurations is the product of the
        # length of each controller.

        if all_controllers_states:
            self._number_of_configurations = reduce(
                lambda x, y: x * y, map(len, all_controllers_states)
            )
        else:
            self._number_of_configurations = 0

        if maximum_number_of_configurations is None:
            maximum_number_of_configurations = get_default_value(
                name="maximum_number_catalog_expressions", section="Estimation"
            )
        if self.number_of_configurations() > maximum_number_of_configurations:
            self.all_configurations_ids = None
            self.all_configurations = None
            return

        self.all_configurations_ids = {
            SEPARATOR.join(combination)
            for combination in product(*all_controllers_states)
        }
        self.all_configurations = {
            Configuration.from_string(conf_id)
            for conf_id in self.all_configurations_ids
        }

    def reset_selection(self) -> None:
        for controller in self.controllers:
            controller.reset_selection()

    def number_of_configurations(self) -> int:
        """Total number of configurations"""
        return self._number_of_configurations

    def get_configuration(self) -> Configuration:
        """Obtain the current configuration of the controllers"""
        selections = (
            SelectionTuple(
                controller=controller.controller_name,
                selection=controller.current_name(),
            )
            for controller in self.controllers
        )
        return Configuration(selections)

    def set_configuration(self, configuration: Configuration) -> None:
        """Apply a configuration to the controllers

        :param configuration: the configuration to be applied
        :type configuration: Configuration
        """
        properly_set = {
            controller.controller_name: False for controller in self.controllers
        }
        for selection in configuration.selections:
            controller: Controller = self.dict_of_controllers.get(selection.controller)
            if controller is None:
                error_msg = (
                    f"Wrong configuration: {configuration}. Controller "
                    f"{selection.controller} is unknown"
                )
                raise BiogemeError(error_msg)
            controller.set_name(selection.selection)
            properly_set[selection.controller] = True

        missing_controllers = [name for name, done in properly_set.items() if not done]
        if missing_controllers:
            error_msg = (
                f"Incomplete configuration {configuration}. The following controllers "
                f"are not defined: {missing_controllers}"
            )
            raise BiogemeError(error_msg)

    def set_configuration_from_id(self, configuration_id: str) -> None:
        """Apply a configuration to the controllers

        :param configuration_id: the ID of the configuration to be applied
        :type configuration_id: str
        """
        the_configuration = Configuration.from_string(configuration_id)
        self.set_configuration(the_configuration)

    def set_controller(self, controller_name: str, index: int) -> None:
        """Set the given controller to the specified index

        :param controller_name: name of the controller
        :type controller_name: string

        :param index: index of the selection
        :type index: int
        """
        the_controller = self.dict_of_controllers.get(controller_name)

        if the_controller is None:
            error_msg = f'Unknown controller: {controller_name}'
            raise BiogemeError(error_msg)

        the_controller.set_index(index)

    def expression_iterator(self) -> Iterator[NamedExpression]:
        """Iterate over the expression corresponding to each configuration."""
        if self.all_configurations is None:
            return
        for config in self.all_configurations:
            self.set_configuration(config)
            yield NamedExpression(name=config.string_id, expression=self.expression)

    def expression_configuration_iterator(self) -> Iterator[str]:
        """Iterate over the expression corresponding to each configuration."""
        if self.all_configurations is None:
            return
        for config in self.all_configurations:
            self.set_configuration(config)
            yield config

    def increased_controller(
        self,
        controller_name: str,
        current_config: Configuration,
        step: int,
    ) -> tuple[Configuration, int]:
        """Increase the selection of one controller by "step"

        :param controller_name: name of the controller to modify
        :type controller_name: str

        :param current_config: current configuration
        :type current_config: str

        :param step: number of steps to perform
        :type step: int
        """
        self.set_configuration(current_config)
        the_controller = self.dict_of_controllers.get(controller_name)
        if the_controller is None:
            error_msg = f'Unknown controller {the_controller}'
            raise BiogemeError(error_msg)
        the_controller.modify_controller(step=step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def get_operator_increased_controllers(
        self, controller_name: str
    ) -> ControllerOperator:

        def operator_increased_controller(
            current_config: Configuration, step: int
        ) -> tuple[Configuration, int]:
            return self.increased_controller(
                controller_name=controller_name,
                current_config=current_config,
                step=step,
            )

        return operator_increased_controller

    def decreased_controller(
        self,
        controller_name: str,
        current_config: Configuration,
        step: int,
    ) -> tuple[Configuration, int]:
        """Decrease the selection of one controller by "step"

        :param controller_name: name of the controller to modify
        :param current_config: current configuration
        :param step: number of steps to perform
        :return: ID of the new configuration and number of steps performed.
        """
        self.set_configuration(current_config)
        the_controller = self.dict_of_controllers.get(controller_name)
        if the_controller is None:
            error_msg = f'Unknown controller {the_controller}'
            raise BiogemeError(error_msg)
        the_controller.modify_controller(step=-step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def get_operator_decreased_controllers(
        self, controller_name: str
    ) -> ControllerOperator:

        def operator_decreased_controller(
            current_config: Configuration, step: int
        ) -> tuple[Configuration, int]:
            return self.decreased_controller(
                controller_name=controller_name,
                current_config=current_config,
                step=step,
            )

        return operator_decreased_controller

    def two_controllers(
        self,
        first_controller_name: str,
        second_controller_name: str,
        direction: str,
        current_config: Configuration,
        step: int,
    ) -> tuple[Configuration, int]:
        """Modification of two controllers. Meaning of direction:

        - NE (North-East): first controller increased by "step", second
              controller increased by "step"
        - NW (North-West): first controller decreased by "step", second
              controller increased by "step"
        - SE (South-East): first controller increased by "step", second
              controller decreased by "step"
        - SW (South-West): first controller decreased by "step", second
              controller decreased by "step"

        :param first_controller_name: name of the first_controller to modify
        :type first_controller_name: str

        :param second_controller_name: name of the second_controller to modify
        :param direction: direction based on the compass rose. Must be NE, NW, SE or SW
        :param current_config: current configuration
        :param step: number of steps to perform
        :return: ID of the new configuration and number of steps performed.

        """
        if direction not in ["NE", "NW", "SE", "SW"]:
            error_msg = f'Incorrect direction {direction}. Must be one of {direction}'
            raise BiogemeError(error_msg)
        self.set_configuration(current_config)
        the_first_controller = self.dict_of_controllers.get(first_controller_name)
        if the_first_controller is None:
            error_msg = f'Unknown controller {the_first_controller}'
            raise BiogemeError(error_msg)
        the_second_controller = self.dict_of_controllers.get(second_controller_name)
        if the_second_controller is None:
            error_msg = f'Unknown controller {the_second_controller}'
            raise BiogemeError(error_msg)
        # The direction for the first controller is W (decrease) or E (increase)
        # The direction for the second controller is N (increase) or S (decrease)
        actual_step = step if direction[1] == "E" else -step
        the_first_controller.modify_controller(step=actual_step, circular=True)
        actual_step = step if direction[0] == "N" else -step
        the_second_controller.modify_controller(step=actual_step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def get_operator_two_controllers(
        self, first_controller_name: str, second_controller_name: str, direction: str
    ) -> ControllerOperator:

        def operator_two_controller(
            current_config: Configuration, step: int
        ) -> tuple[Configuration, int]:
            return self.two_controllers(
                first_controller_name=first_controller_name,
                second_controller_name=second_controller_name,
                direction=direction,
                current_config=current_config,
                step=step,
            )

        return operator_two_controller

    def modify_random_controllers(
        self,
        increase: bool,
        current_config: Configuration,
        step: int,
    ) -> tuple[Configuration, int]:
        """Increase the selection of "step" controllers by 1

        :param increase: If True, the indices are increased . If
            False, they are decreased.
        :param current_config: current configuration
        :param step: number of steps to perform

        """
        self.set_configuration(current_config)
        number_of_controllers = len(self.controllers)
        actual_size = min(step, number_of_controllers)
        selected_controllers = random.choices(
            list(self.dict_of_controllers.keys()), k=actual_size
        )
        the_modification = 1 if increase else 1
        for the_controller in selected_controllers:
            self.dict_of_controllers[the_controller].modify_controller(
                step=the_modification, circular=True
            )
        new_config = self.get_configuration()
        return new_config, actual_size

    def get_operator_modify_random_controllers(
        self, increase: bool
    ) -> ControllerOperator:

        def operator_modify_random_controller(
            current_config: Configuration, step: int
        ) -> tuple[Configuration, int]:
            return self.modify_random_controllers(
                increase=increase,
                current_config=current_config,
                step=step,
            )

        return operator_modify_random_controller

    def prepare_operators(
        self,
    ) -> dict[str, ControllerOperator]:
        """Operators are functions that take a configuration and a
        size as arguments, and return a new configuration, and the
        actual number of modifications that have been
        implemented.

        """

        dict_of_operators = {}
        # Increase and decrease controllers
        for name in self.dict_of_controllers.keys():
            dict_of_operators[f"Increase {name}"] = (
                self.get_operator_increased_controllers(controller_name=name)
            )
            dict_of_operators[f"Decrease {name}"] = (
                self.get_operator_decreased_controllers(controller_name=name)
            )
        # Pair of controllers
        directions = ["NE", "NW", "SE", "SW"]
        # Using nested list comprehensions to generate all possible pairs
        all_pairs = [
            (name1, name2, direction)
            for name1 in self.dict_of_controllers.keys()
            for name2 in self.dict_of_controllers.keys()
            for direction in directions
        ]

        # Filter out pairs with the same name
        filtered_pairs = [
            (name1, name2, direction)
            for name1, name2, direction in all_pairs
            if name1 != name2
        ]

        for name1, name2, direction in filtered_pairs:
            dict_of_operators[f"Pair_{name1}_{name2}_{direction}"] = (
                self.get_operator_two_controllers(
                    first_controller_name=name1,
                    second_controller_name=name2,
                    direction=direction,
                )
            )

        # Several controllers
        dict_of_operators["Increase_several"] = (
            self.get_operator_modify_random_controllers(increase=True)
        )
        dict_of_operators["Decrease_several"] = (
            self.get_operator_modify_random_controllers(increase=False)
        )
        return dict_of_operators
