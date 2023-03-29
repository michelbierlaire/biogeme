"""Centralized management of the parameters.

Parameters are organized by section. Parameters within the same
section must have different names

:author: Michel Bierlaire
:date: Fri Dec  2 12:17:38 2022

"""

import logging
from collections import defaultdict
import biogeme.exceptions as excep
import biogeme.default_parameters as dp

logger = logging.getLogger(__name__)


class Parameters:
    """Parameters management"""

    def __init__(self, default=None):
        # Sort parameters by name
        self.default_parameters = (
            dp.all_parameters_tuple if default is None else default
        )
        self.all_parameters_dict = defaultdict(set)

        for p in self.default_parameters:
            self.all_parameters_dict[p.name] |= set({p})

        # Store values in a dict
        self.values = {p: p.default for p in self.default_parameters}
        self.sections = {p.section for p in self.default_parameters}

    def get_param_tuple(self, name, section=None):
        """Obtain a tuple containing the name, section and value of the parameter"""
        tuple_set = self.all_parameters_dict.get(name)
        if tuple_set is None:
            if section is None:
                error_msg = f'Unknown parameter {name}'
            else:
                error_msg = f'Unknown parameter {name} in section {section} '
            raise excep.biogemeError(error_msg)

        if len(tuple_set) == 1:
            # The name is sufficient to identify the parameter
            the_tuple = next(iter(tuple_set))
            if section is not None and section != the_tuple.section:
                error_msg = (
                    f'Parameter {name} is in section {the_tuple.section} '
                    f'and not in section {section}'
                )
                raise excep.biogemeError(error_msg)
            return the_tuple
        # The section is necessary to identify the parameter.
        if section is None:
            error_msg = (
                f'Parameter {name} appears in the following '
                f'sections: {(p.section for p in tuple_set)}'
                f' The section must be mentioned explicitly.'
            )
            raise excep.biogemeError(error_msg)
        for the_tuple in tuple_set:
            if the_tuple.section == section:
                return the_tuple

    @staticmethod
    def check_parameter_value(the_tuple, value):
        """Check if the value of the parameter is valid

        :param the_tuple: tuple to check
        :type the_tuple: biogeme.default_parameters.ParameterTuple

        :param value: value of the parameter
        :type value: float, int or str

        :return: a boolean that is True if the value is valid. If not,
            diagnostics are provided.
        :rtype: tuple(bool, list(str))
        """
        ok = True
        messages = []
        for check in the_tuple.check:
            is_ok, diag = check(value)
            if not is_ok:
                ok = False
                messages.append(
                    f'Error for parameter {the_tuple.name} in '
                    f'section {the_tuple.section}. {diag}'
                )
        return ok, messages

    def set_value_from_tuple(self, the_tuple, value):
        """Set a value of a parameter."""
        ok, messages = self.check_parameter_value(the_tuple, value)
        if not ok:
            raise excep.biogemeError(messages)
        self.values[the_tuple] = value

    def set_value(self, name, value, section=None):
        """Set a value of a parameter. If the parameter appears in
        only one section, the name of the section can be ommitted.

        :param name: name of the parameter
        :type param: str

        :param value: value of the parameter
        :type value: str

        :param section: section containing the parameter
        :type section: str
        """
        the_tuple = self.get_param_tuple(name, section)
        self.set_value_from_tuple(the_tuple, value)

    def get_value(self, name, section=None):
        """Get the value of a parameter. If the parameter appears in
        only one section, the name of the section can be ommitted.

        :param name: name of the parameter
        :type param: str

        :param section: section containing the parameter
        :type section: str
        """
        the_tuple = self.get_param_tuple(name, section)
        return self.values[the_tuple]
