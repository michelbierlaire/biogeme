"""Management of the parameters with a TOML file

:author: Michel Bierlaire
:date: Fri Dec  2 12:36:05 2022

"""
import logging
from datetime import date
import tomlkit as tk
import biogeme.parameters as param
import biogeme.exceptions as excep

from biogeme.version import getVersion

logger = logging.getLogger(__name__)


TRUE_STR = ('True', 'true', 'Yes', 'yes')
FALSE_STR = ('False', 'false', 'No', 'no')
DEFAULT_FILE_NAME = 'biogeme.toml'
ROW_LENGTH = 80


def format_comment(the_param):
    """Format the content of the file, in particular the description of the parameter

    :param the_param: parmeter to format
    :type the_param: biogeme.default_parameters.ParameterTuple

    :return: formatted text
    :rtype: str
    """
    the_str = f'{the_param.name} = {the_param.default}'
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


class Toml:

    """TOML file for parameters"""

    def __init__(self, parameter_file=None):
        self.parameters = param.Parameters()
        self.parameter_file = (
            DEFAULT_FILE_NAME if parameter_file is None else parameter_file
        )

        logger.info(f'Parameters read from {self.parameter_file}')

        self.read_file()
        self.record_values()

    def read_file(self):
        """Read TOML file
        """
        try:
            with open(self.parameter_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.document = tk.parse(content)
        except FileNotFoundError:
            # If the file does not exist, defaults values are used and
            # the file is created.
            self.dump_file()

    def dump_file(self):
        """Dump the values of the parameters in the TOML file
        """
        self.document = self.generate_document()
        with open(self.parameter_file, 'w', encoding='utf-8') as f:
            print(tk.dumps(self.document), file=f)
        logger.warning(f'File {self.parameter_file} has been created')

    def record_values(self):
        """Record the values of the parameters in the TOML document
        """
        for the_param in self.parameters.default_parameters:
            the_table = self.document.get(the_param.section)
            new_table = False
            if the_table is None:
                the_table = tk.table()
                new_table = True
            value = the_table.get(the_param.name)
            if value is None:
                value = the_param.default
                the_table.add(the_param.name, value)
            elif the_param.type is bool:
                if value in TRUE_STR:
                    self.parameters.set_value_from_tuple(the_param, True)
                elif value in FALSE_STR:
                    self.parameters.set_value_from_tuple(the_param, False)
                else:
                    error_msg = (
                        f'Error with parameter {the_param.name}: '
                        f'{value} is not a valid boolean. Use "True" of "False"'
                    )
                    raise excep.BiogemeError(error_msg)
            else:
                self.parameters.set_value_from_tuple(the_param, the_param.type(value))
            if new_table:
                self.document.add(the_param.section, the_table)

    def generate_document(self):
        """Generate the  TOML document"""
        doc = tk.document()

        doc.add(tk.comment(f'Default parameter file for Biogeme {getVersion()}'))
        doc.add(tk.comment(f'Automatically created on {date.today()}'))

        tables = {section: tk.table() for section in self.parameters.sections}

        for p, value in self.parameters.values.items():
            if isinstance(value, bool):
                value = 'True' if value else 'False'
            tables[p.section].add(p.name, value)
            tables[p.section][p.name].comment(format_comment(p))

        for s, t in tables.items():
            doc[s] = t

        return doc
