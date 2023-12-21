"""Defines controllers for multiple expressions.

:author: Michel Bierlaire
:date: Sun Jul 16 15:23:46 2023

"""
from itertools import product
from functools import partial, reduce
import random
import biogeme.exceptions as excep
from biogeme.parameters import biogeme_parameters
from biogeme.configuration import (
    SelectionTuple,
    Configuration,
    SELECTION_SEPARATOR,
    SEPARATOR,
)


class Controller:
    """Class controlling the specification of the Catalogs"""

    def __init__(self, controller_name, specification_names):
        """Constructor

        :param controller_name: name of the controller
        :type controller_name: str

        :param specification_names: list or tuple of the names of the
            specification controlled by the controller
        :type specification_names: list(str) or tuple(str)
        """
        self.controller_name = controller_name
        self.specification_names = specification_names
        self.current_index = 0
        self.dict_of_index = {
            name: index for index, name in enumerate(self.specification_names)
        }
        self.controlled_catalogs = []

    def all_configurations(self):
        """Return the code of all configurations

        :return: list of codes
        :rtype: list(str)
        """
        return [
            f"{self.controller_name}{SELECTION_SEPARATOR}{specification}"
            for specification in self.specification_names
        ]

    def __eq__(self, other):
        return self.controller_name == other.controller_name

    def __lt__(self, other):
        return self.controller_name < other.controller_name

    def __hash__(self):
        return hash(self.controller_name)

    def __repr__(self):
        return repr(self.controller_name)

    def __str__(self):
        return f"{self.controller_name}: {self.specification_names}"

    def controller_size(self):
        """Number of specifications managed by this controller"""
        return len(self.specification_names)

    def current_name(self):
        """Name of the currently selected expression"""
        return self.specification_names[self.current_index]

    def set_name(self, name):
        """Set the index of the controller based on the name of the specification

        :param name: name of the specification
        :type name: str
        """
        the_index = self.dict_of_index.get(name)
        if the_index is None:
            error_msg = (
                f"{name}: unknown specification for controller {self.controller_name}"
            )
            raise excep.BiogemeError(error_msg)
        self.set_index(the_index)

    def set_index(self, index):
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
            raise excep.BiogemeError(error_msg)

        self.current_index = index
        for catalog in self.controlled_catalogs:
            catalog.current_index = index

    def reset_selection(self):
        """Select the first specification"""
        self.set_index(0)

    def modify_controller(self, step, circular):
        """Modify the specification of the controller

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


class CentralController:
    """Class controlling the complete multiple expression"""

    def __init__(self, expression):
        """Constructor

        :param controllers: all controllers of the expression
        :type controllers: tuple(controllers)

        :param expression: controllers expression
        :type expression: biogeme.expressions.Expression

        """

        set_of_controllers = expression.get_all_controllers()

        self.controllers = tuple(sorted(set_of_controllers))
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

        maximum_number_of_configurations = biogeme_parameters.get_value(
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
            Configuration.from_string(id) for id in self.all_configurations_ids
        }

    def number_of_configurations(self):
        """Total number of configurations"""
        return self._number_of_configurations

    def get_configuration(self):
        """Obtain the current configuration of the controllers"""
        selections = (
            SelectionTuple(
                controller=controller.controller_name,
                selection=controller.current_name(),
            )
            for controller in self.controllers
        )
        return Configuration(selections)

    def set_configuration(self, configuration):
        """Apply a configuration to the controllers

        :param configuration: the configuration to be applied
        :type configuration: Configuration
        """
        properly_set = {
            controller.controller_name: False for controller in self.controllers
        }
        for selection in configuration.selections:
            controller = self.dict_of_controllers.get(selection.controller)
            if controller is None:
                error_msg = (
                    f"Wrong configuration: {configuration}. Controller "
                    f"{selection.controller} is unknown"
                )
                raise excep.BiogemeError(error_msg)
            controller.set_name(selection.selection)
            properly_set[selection.controller] = True

        missing_controllers = [name for name, done in properly_set.items() if not done]
        if missing_controllers:
            error_msg = (
                f"Incomplete configuration {configuration}. The following controllers "
                f"are not defined: {missing_controllers}"
            )
            raise excep.BiogemeError(error_msg)

    def set_configuration_from_id(self, configuration_id):
        """Apply a configuration to the controllers

        :param configuration_id: the ID of the configuration to be applied
        :type configuration_id: str
        """
        the_configuration = Configuration.from_string(configuration_id)
        self.set_configuration(the_configuration)

    def set_controller(self, controller_name, index):
        """Set the given controller to the specified index

        :param controller_name: name of the controller
        :type controller_name: string

        :param index: index of the selection
        :type index: int
        """
        the_controller = self.dict_of_controllers.get(controller_name)

        if the_controller is None:
            error_msg = f"Unknown controller: {controller_name}"
            raise excep.BiogemeError(error_msg)

        the_controller.set_index(index)

    def increased_controller(self, controller_name, current_config, step):
        """Increase the selection of one controller by "step"

        :param controller_name: name of the controller to modify
        :type controller_name: str

        :param current_config: current configuration
        :type current_config: str

        :param step: number of steps to perform
        :type step: int
        """
        self.set_configuration_from_id(current_config)
        the_controller = self.dict_of_controllers.get(controller_name)
        if the_controller is None:
            error_msg = f"Unknown controller {the_controller}"
            raise excep.BiogemeError(error_msg)
        the_controller.modify_controller(step=step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def decreased_controller(self, controller_name, current_config, step):
        """Decrease the selection of one controller by "step"

        :param controller_name: name of the controller to modify
        :type controller_name: str

        :param current_config: current configuration
        :type current_config: str

        :param step: number of steps to perform
        :type step: int

        :return: ID of the new configuration and number of steps performed.
        :rtype: tuple(str, int)
        """
        self.set_configuration_from_id(current_config)
        the_controller = self.dict_of_controllers.get(controller_name)
        if the_controller is None:
            error_msg = f"Unknown controller {the_controller}"
            raise excep.BiogemeError(error_msg)
        the_controller.modify_controller(step=-step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def two_controllers(
        self,
        first_controller_name,
        second_controller_name,
        direction,
        current_config,
        step,
    ):
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
        :type second_controller_name: str

        :param direction: direction based on the compass rose. Must be NE, NW, SE or SW
        :type direction: str

        :param current_config: current configuration
        :type current_config: str

        :param step: number of steps to perform
        :type step: int

        :return: ID of the new configuration and number of steps performed.
        :rtype: tuple(str, int)

        """
        if direction not in ["NE", "NW", "SE", "SW"]:
            error_msg = f"Incorrect direction {direction}. Must be NE, NW, SE or SW"
            raise excep.BiogemeError(error_msg)
        self.set_configuration_from_id(current_config)
        the_first_controller = self.dict_of_controllers.get(first_controller_name)
        if the_first_controller is None:
            error_msg = f"Unknown controller {the_first_controller}"
            raise excep.BiogemeError(error_msg)
        the_second_controller = self.dict_of_controllers.get(second_controller_name)
        if the_second_controller is None:
            error_msg = f"Unknown controller {the_second_controller}"
            raise excep.BiogemeError(error_msg)
        # The direction for the first controller is W (decrease) or E (increase)
        # The direction for the second controller is N (increase) or S (decrease)
        actual_step = step if direction[1] == "E" else -step
        the_first_controller.modify_controller(step=actual_step, circular=True)
        actual_step = step if direction[0] == "N" else -step
        the_second_controller.modify_controller(step=actual_step, circular=True)
        new_config = self.get_configuration()
        return new_config, step

    def modify_random_controllers(
        self,
        increase,
        current_config,
        step,
    ):
        """Increase the selection of "step" controllers by 1

        :param increase: If True, the indices are increased . If
            False, they are decreased.
        :type increase: bool

        :param current_config: current configuration
        :type current_config: str

        :param step: number of steps to perform
        :type step: int

        """
        self.set_configuration_from_id(current_config)
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

    def prepare_operators(self):
        """Operators are functions that take a configuration and a
        size as arguments, and return a new configuration, and the
        actual number of modifications that have been
        implemented.

        """

        dict_of_operators = {}
        # Increase and decrease controllers
        for name in self.dict_of_controllers.keys():
            dict_of_operators[f"Increase {name}"] = partial(
                self.increased_controller, controller_name=name
            )
            dict_of_operators[f"Decrease {name}"] = partial(
                self.decreased_controller, controller_name=name
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
            dict_of_operators[f"Pair_{name1}_{name2}_{direction}"] = partial(
                self.two_controllers,
                first_controller_name=name1,
                second_controller_name=name2,
                direction=direction,
            )

        # Several controllers
        dict_of_operators["Increase_several"] = partial(
            self.modify_random_controllers, increase=True
        )
        dict_of_operators["Decrease_several"] = partial(
            self.modify_random_controllers, increase=False
        )
        return dict_of_operators
