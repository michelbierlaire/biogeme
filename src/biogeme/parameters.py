"""Centralized management of the parameters.

Parameters are organized by section. Parameters within the same
section must have different names

:author: Michel Bierlaire
:date: Fri Dec  2 12:17:38 2022

"""

import logging
import os
from datetime import datetime
from typing import NamedTuple
import tomlkit as tk
import biogeme.exceptions as excep
from biogeme.tools.files import is_valid_filename
from biogeme.version import get_version
from biogeme.default_parameters import (
    all_parameters_tuple,
    ParameterTuple,
    ParameterValue,
)

logger = logging.getLogger(__name__)

TRUE_STR = ('True', 'true', 'Yes', 'yes')
FALSE_STR = ('False', 'false', 'No', 'no')
DEFAULT_FILE_NAME = 'biogeme.toml'
ROW_LENGTH = 80


class NameSectionTuple(NamedTuple):
    name: str
    section: str


def format_comment(the_param: ParameterTuple) -> str:
    """Format the content of the file, in particular the description of the parameter

    :param the_param: parmeter to format
    :type the_param: biogeme.default_parameters.ParameterTuple

    :return: formatted text
    :rtype: str
    """
    the_str = f'{the_param.name} = {the_param.value}'
    multiplier = len(the_str) + 1
    words = the_param.description.split(' ')
    current_row = '#'
    rows = []
    for word in words:
        the_length = len(current_row)
        if not rows:
            the_length += multiplier
        if the_length + len(word) > ROW_LENGTH:
            # Start a new line
            rows.append(current_row)
            blanks = ' ' * multiplier
            if the_param.type in (bool, str):
                blanks += '  '
            current_row = f'{blanks}#'
        current_row += ' ' + word
    if current_row != '':
        rows.append(current_row)
    formatted = '\n'.join(rows)
    return formatted


def parse_boolean(value: str) -> bool:
    """Transforms one of the string representing a boolean into an actual boolean

    :param value: value to be transformed
    :type value: str

    :return: equivalent boolean
    :rtype: bool

    """
    if value in TRUE_STR:
        return True
    if value in FALSE_STR:
        return False

    error_msg = f'{value} is not a valid boolean. Use "True" of "False"'
    raise excep.BiogemeError(error_msg)


class Parameters:
    """Parameters management"""

    def __init__(self) -> None:
        # Store values in a dict
        self.document = None  # TOML document
        self.file_name = None
        self.all_parameters_dict: dict[NameSectionTuple, ParameterTuple] = {}
        # Add the default parameters
        for param in all_parameters_tuple():
            self.add_parameter(param)

    def __str__(self) -> str:
        output = ''
        for section in self.sections:
            output += f'[{section}]\n'
            report = [
                f'\t{param.name} = {param.value} '
                for param in self.parameters_in_section(section)
            ]
            output += '\n'.join(report)
            output += '\n'
        return output

    @property
    def sections(self) -> list[str]:
        return sorted(
            {name_section.section for name_section in self.all_parameters_dict.keys()}
        )

    @property
    def parameter_names(self) -> list[str]:
        return sorted(
            {name_section.name for name_section in self.all_parameters_dict.keys()}
        )

    def add_parameter(self, parameter_tuple: ParameterTuple):
        """Add one parameter

        :param parameter_tuple: tuple containing the data associated with the parameter
        :type parameter_tuple: biogeme.default_parameters.ParameterTuple
        """
        key = NameSectionTuple(
            name=parameter_tuple.name, section=parameter_tuple.section
        )
        ok, messages = self.check_parameter_value(parameter_tuple)
        if not ok:
            raise excep.BiogemeError(messages)

        already_there = self.all_parameters_dict.get(key)
        self.all_parameters_dict[key] = parameter_tuple

    def read_file(self, file_name: str):
        """Read TOML file. If the file name is invalid (typically, an empty string),
        the default value sof the parameters are used"""
        logger.debug(
            f'READ FILE {file_name} : {self.get_value(name="optimization_algorithm")}'
        )
        if file_name is not None:
            is_valid, why = is_valid_filename(os.path.basename(file_name))
            if not is_valid:
                self.file_name = None
                logger.warning(
                    f'The parameter file name provided is invalid :[{file_name}] ({why}).'
                )
                return

        self.file_name = DEFAULT_FILE_NAME if file_name is None else file_name

        try:
            with open(self.file_name, 'r', encoding='utf-8') as f:

                logger.debug(f'Parameter file: {os.path.abspath(self.file_name)}')
                content = f.read()
                self.document = tk.parse(content)
                self.import_document()
                logger.info(f'Biogeme parameters read from {self.file_name}.')

        except FileNotFoundError:
            info_msg = f'Default values of the Biogeme parameters are used.'
            logger.info(info_msg)
            self.dump_file(self.file_name)

    def parameters_in_section(self, section_name: str) -> list[ParameterTuple]:
        """Returns the parameters in a section

        :param section_name: name of the section
        :type section_name: str

        """
        results = [
            p for p in self.all_parameters_dict.values() if p.section == section_name
        ]
        return results

    def sections_from_name(self, name: str) -> set[str]:
        """Obtain the list of sections containing the given entry

        :param name: name of the parameter
        :type name: str
        """
        return {p.section for p in self.all_parameters_dict if p.name == name}

    def dump_file(self, file_name: str):
        """Dump the values of the parameters in the TOML file"""
        self.document = self.generate_document()
        with open(file_name, 'w', encoding='utf-8') as f:
            print(tk.dumps(self.document), file=f)
        logger.warning(f'File {file_name} has been created')

    def import_document(self) -> None:
        """Record the values of the parameters in the TOML document"""
        for section_name, entries in self.document.items():
            for entry_name, entry_value in entries.items():
                key = NameSectionTuple(name=entry_name, section=section_name)
                default = self.all_parameters_dict.get(key)
                if default is None:
                    warning_msg = (
                        f'Entry {entry_name} in Section {section_name} is '
                        f'ignored by Biogeme.'
                    )
                    logger.warning(warning_msg)
                    continue
                if entry_value is None:
                    value = default.value
                elif default.type is bool:
                    try:
                        value = parse_boolean(entry_value)
                    except excep.BiogemeError as e:
                        error_msg = (
                            f'Error with entry {entry_name} in Section '
                            f'{section_name}: {e}'
                        )
                        raise error_msg from e
                else:
                    value = entry_value

                the_parameter = ParameterTuple(
                    name=entry_name,
                    value=value,
                    type=default.type,
                    section=section_name,
                    description=default.description,
                    check=default.check,
                )
                self.add_parameter(the_parameter)

    def generate_document(self) -> tk.TOMLDocument:
        """Generate the  TOML document"""
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%B %d, %Y. %H:%M:%S")
        doc = tk.document()

        doc.add(tk.comment(f'Default parameter file for Biogeme {get_version()}'))
        doc.add(tk.comment(f'Automatically created on {formatted_datetime}'))

        sections = {p.section for p in self.all_parameters_dict}

        tables = {section: tk.table() for section in sections}

        for parameter in self.all_parameters_dict.values():
            if isinstance(parameter.value, bool):
                value = 'True' if parameter.value else 'False'
            else:
                value = parameter.value
            tables[parameter.section].add(parameter.name, value)
            tables[parameter.section][parameter.name].comment(format_comment(parameter))

        for s, t in tables.items():
            doc[s] = t
        return doc

    def get_param_tuple(self, name: str, section: str | None = None) -> ParameterTuple:
        """Obtain a tuple describing the parameter

        :param name: name of the parameter
        :type name: str

        :param section: section where the parameter is relevant
        :type section: str

        :return: tuple containing the name, section and value of the
            parameter, or None if not found
        :rtype: biogeme.default_parameters.ParameterTuple

        """
        if section is None:
            sections = self.sections_from_name(name)
            if len(sections) > 1:
                error_msg = (
                    f'Ambiguity: parameter {name} belongs to multiple '
                    f'sections: {sections}.'
                )
                raise excep.BiogemeError(error_msg)
            if not sections:
                error_msg = f'Parameter {name} does not belong to any section.'
                raise excep.BiogemeError(error_msg)
            section = sections.pop()

        name_section = NameSectionTuple(name=name, section=section)
        the_tuple = self.all_parameters_dict.get(name_section)
        if the_tuple is None:
            if section is None:
                error_msg = f'Unknown parameter {name}'
            else:
                known = self.parameters_in_section(section)
                if known:
                    known_msg = f'Known parameter(s): {", ".join([param.name for param in known])}.'
                else:
                    known_msg = 'The section does not exist.'
                error_msg = (
                    f'Unknown parameter {name} in section {section}. {known_msg}'
                )
            raise excep.BiogemeError(error_msg)

        return the_tuple

    @staticmethod
    def check_parameter_value(the_tuple: ParameterTuple):
        """Check if the value of the parameter is valid

        :param the_tuple: tuple to check
        :type the_tuple: biogeme.default_parameters.ParameterTuple

        :return: a boolean that is True if the value is valid. If not,
            diagnostics are provided.
        :rtype: tuple(bool, list(str))
        """
        ok = True
        messages = []
        if the_tuple.check is None:
            return ok, messages
        for check in the_tuple.check:
            is_ok, diag = check(the_tuple.value)
            if not is_ok:
                ok = False
                messages.append(
                    f'Error for parameter {the_tuple.name} in '
                    f'section {the_tuple.section}. {diag}'
                )
        return ok, messages

    def set_value(self, name: str, value: ParameterValue, section: str | None = None):
        """Set a value of a parameter. If the parameter appears in
        only one section, the name of the section can be ommitted.

        :param name: name of the parameter
        :type name: str

        :param value: value of the parameter
        :type value: str

        :param section: section containing the parameter
        :type section: str
        """
        the_tuple = self.get_param_tuple(name, section)
        the_parameter = ParameterTuple(
            name=the_tuple.name,
            value=value,
            type=the_tuple.type,
            section=the_tuple.section,
            description=the_tuple.description,
            check=the_tuple.check,
        )

        self.add_parameter(the_parameter)

    def get_value(self, name: str, section: str | None = None) -> ParameterValue:
        """Get the value of a parameter. If the parameter appears in
        only one section, the name of the section can be omitted.

        :param name: name of the parameter
        :type name: str

        :param section: section containing the parameter
        :type section: str
        """
        the_tuple = self.get_param_tuple(name, section)
        # logger.info(f'{name} = {the_tuple.value}')
        return the_tuple.value


def get_default_value(name: str, section: str | None = None) -> ParameterValue:
    the_parameters = Parameters()
    return the_parameters.get_value(name=name, section=section)
