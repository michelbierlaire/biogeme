"""
Test the parameters module

:author: Michel Bierlaire
:date: Tue Nov 29 10:26:32 2022

"""

import shutil
from os import path
import tempfile
import unittest
import numpy as np
from biogeme.parameters import Parameters
import biogeme.exceptions as excep
from biogeme.default_parameters import ParameterTuple
import biogeme.check_parameters as cp
import biogeme.optimization as opt

BLANKS = 25 * ' '

FILE_CONTENT = """
[Specification]
suggest_scales = "True" # bool: If True, Biogeme suggests the scaling of
                         # the variables in the database.
missing_data = 99999 # number: If one variable has this value, it is assumed
                         # that a data is missing and an exception will
                         # be triggered.

[Output]
generate_html = "True" # bool: "True" if the HTML file
                         # with the results must be generated.
generate_pickle = "True" # bool: "True" if the pickle file
                         # with the results must be generated.

[Estimation]
second_derivatives = 1.0 # float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated
tolerance = 6.06273418136464e-06 # float: the algorithm stops when this precision is reached
max_iterations = 100 # int: maximum number of iterations
optimization_algorithm = "simple_bounds" # str: optimization algorithm to be used for estimation.
                         # Valid values: ['scipy', 'LS-newton', 'TR-newton', 'LS-BFGS', 'TR-BFGS', 'simple_bounds']

save_iterations = "True" # bool: If True, the current iterate is saved after
                         # each iteration, in a file named
                         # ``__[modelName].iter``, where
                         # ``[modelName]`` is the name given to the
                         # model. If such a file exists, the starting
                         # values for the estimation are replaced by
                         # the values saved in the file.

a_param = "lopsum"

[MonteCarlo]
number_of_draws = 1000 # int: Number of draws for Monte-Carlo integration.
seed = 0 # int: Seed used for the pseudo-random number generation. It is
                         # useful only when each run should generate the exact same
                         # result. If 0, a new seed is used at each run.

[MultiThreading]
number_of_threads = 0 # int: Number of threads/processors to be used. If
                         # the parameter is 0, the number of
                         # available threads is calculated using
                         # cpu_count().

"""


default_parameters = (
    ParameterTuple(
        name='second_derivatives',
        value=1.0,
        type=float,
        section='Estimation',
        description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
        check=(cp.zero_one, cp.is_number),
    ),
    ParameterTuple(
        name='tolerance',
        value=np.finfo(np.float64).eps ** 0.3333,
        type=float,
        section='Estimation',
        description='# float: the algorithm stops when this precision is reached',
        check=(cp.zero_one, cp.is_number),
    ),
    ParameterTuple(
        name='max_iterations',
        value=100,
        type=int,
        section='Estimation',
        description='# int: maximum number of iterations',
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='optimization_algorithm',
        value='simple_bounds',
        type=str,
        section='Estimation',
        description=(
            f'# str: optimization algorithm to be used for estimation.\n'
            f'{BLANKS}# Valid values: {list(opt.algorithms.keys())}'
        ),
        check=(cp.check_algo_name,),
    ),
    ParameterTuple(
        name='generate_html',
        value=True,
        type=bool,
        section='Output',
        description=(
            '# bool: "True" if the HTML file\n'
            '{BLANKS}# with the results must be generated.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='generate_pickle',
        value=True,
        type=bool,
        section='Output',
        description=(
            f'# bool: "True" if the pickle file\n'
            f'{BLANKS}# with the results must be generated.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='number_of_threads',
        value=0,
        type=int,
        section='MultiThreading',
        description=(
            f'# int: Number of threads/processors to be used. If\n'
            f'{BLANKS}# the parameter is 0, the number of\n'
            f'{BLANKS}# available threads is calculated using\n'
            f'{BLANKS}# cpu_count().'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='number_of_draws',
        value=1000,
        type=int,
        section='MonteCarlo',
        description=('int: Number of draws for Monte-Carlo integration.'),
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='skip_audit',
        value=False,
        type=bool,
        section='Specification',
        description=(
            f'# bool: If True, does not check the validity of the\n'
            f'{BLANKS}# formulas. It may save significant amount of\n'
            f'{BLANKS}# time for large models and large data sets.\n'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='suggest_scales',
        value=True,
        type=bool,
        section='Specification',
        description=(
            f'# bool: If True, Biogeme suggests the scaling of\n'
            f'{BLANKS}# the variables in the database.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='missing_data',
        value=99999,
        type=int,
        section='Specification',
        description=(
            f'# number: If one variable has this value, it is assumed\n'
            f'{BLANKS}# that a data is missing and an exception will\n'
            f'{BLANKS}# be triggered.'
        ),
        check=(cp.is_number,),
    ),
    ParameterTuple(
        name='seed',
        value=0,
        type=int,
        section='MonteCarlo',
        description=(
            f'# int: Seed used for the pseudo-random number generation. It is\n'
            f'{BLANKS}# useful only when each run should generate the exact same\n'
            f'{BLANKS}# result. If 0, a new seed is used at each run.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='save_iterations',
        value=True,
        type=bool,
        section='Estimation',
        description=(
            f'# bool: If True, the current iterate is saved after\n'
            f'{BLANKS}# each iteration, in a file named\n'
            f'{BLANKS}# ``__[modelName].iter``, where\n'
            f'{BLANKS}# ``[modelName]`` is the name given to the\n'
            f'{BLANKS}# model. If such a file exists, the starting\n'
            f'{BLANKS}# values for the estimation are replaced by\n'
            f'{BLANKS}# the values saved in the file.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='a_param',
        value='lopsum',
        type=str,
        section='Estimation',
        description='',
        check=None,
    ),
    ParameterTuple(
        name='a_param',
        value='anything',
        type=str,
        section='Specification',
        description='',
        check=None,
    ),
)


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.biogeme_parameters = Parameters()
        for p in default_parameters:
            self.biogeme_parameters.add_parameter(p)

    def test_get_param_tuple_1(self):
        """Test the retrieval of parameter without the section name"""
        the_tuple = self.biogeme_parameters.get_param_tuple(name='missing_data')
        self.assertEqual(the_tuple.name, 'missing_data')
        self.assertEqual(the_tuple.value, 99999)
        self.assertEqual(the_tuple.section, 'Specification')

    def test_get_param_tuple_2(self):
        """Test the retrieval of parameter with the section name"""
        the_tuple = self.biogeme_parameters.get_param_tuple(
            name='missing_data', section='Specification'
        )
        self.assertEqual(the_tuple.name, 'missing_data')
        self.assertEqual(the_tuple.value, 99999)
        self.assertEqual(the_tuple.section, 'Specification')

    def test_get_param_tuple_3(self):
        """Test the error if the parameter does not appear in the
        mentioned section
        """
        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_param_tuple(
                name='missing_data', section='Estimation'
            )

        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_param_tuple(name='unknown_parameter')

        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_param_tuple(
                name='unknown_parameter', section='Estimation'
            )

    def test_get_param_tuple_4(self):
        """Test the error if the parameter does appears in two
        sections, and the section is not specified.
        """
        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_param_tuple(
                name='a_param',
            )

    def test_get_param_tuple_5(self):
        """Test the retrieval of a parameter that appears in two
        sections, with the section name

        """
        the_tuple = self.biogeme_parameters.get_param_tuple(
            name='a_param', section='Estimation'
        )
        self.assertEqual(the_tuple.name, 'a_param')
        self.assertEqual(the_tuple.value, 'lopsum')
        self.assertEqual(the_tuple.section, 'Estimation')

    def test_get_param_tuple_6(self):
        """Test the retrieval of a parameter that appears in two
        section, with the section name

        """
        the_tuple = self.biogeme_parameters.get_param_tuple(
            name='a_param', section='Specification'
        )
        self.assertEqual(the_tuple.name, 'a_param')
        self.assertEqual(the_tuple.value, 'anything')
        self.assertEqual(the_tuple.section, 'Specification')

    def test_check_parameter_value(self):
        a_tuple = ParameterTuple(
            name='param_name',
            value=2.0,
            type=float,
            section='Estimation',
            description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
            check=(cp.zero_one, cp.is_number),
        )
        ok, _ = self.biogeme_parameters.check_parameter_value(a_tuple)
        self.assertFalse(ok)

    def test_add_parameter(self):
        a_tuple = ParameterTuple(
            name='param_name',
            value=0.5,
            type=float,
            section='Estimation',
            description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
            check=(cp.zero_one, cp.is_number),
        )
        self.biogeme_parameters.add_parameter(a_tuple)
        self.assertEqual(
            self.biogeme_parameters.get_value(
                name=a_tuple.name, section=a_tuple.section
            ),
            0.5,
        )
        another_tuple = ParameterTuple(
            name='param_name',
            value=1.5,
            type=float,
            section='Estimation',
            description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
            check=(cp.zero_one, cp.is_number),
        )
        with self.assertRaises(excep.BiogemeError):
            self.biogeme_parameters.add_parameter(another_tuple)

    def test_set_get_value(self):
        value = 1.0e-3
        self.biogeme_parameters.set_value('tolerance', value, section='Estimation')
        check = self.biogeme_parameters.get_value('tolerance', section='Estimation')
        self.assertEqual(value, check)


class TestToml(unittest.TestCase):
    """Tests for the biogeme.toml module"""

    def setUp(self):
        self.biogeme_parameters = Parameters()
        for p in default_parameters:
            self.biogeme_parameters.add_parameter(p)

    def test_from_file(self):
        """Read the parameters from file"""
        # Create a temporary directory
        test_dir = tempfile.mkdtemp()
        test_file = path.join(test_dir, 'biogeme.toml')
        with open(test_file, 'w', encoding='utf-8') as f:
            print(FILE_CONTENT, file=f)
        self.biogeme_parameters.read_file(file_name=test_file)

        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_value('a_wrong_param', section='Estimation')

        check_int = self.biogeme_parameters.get_value('missing_data')
        self.assertEqual(check_int, 99999)

        check_float = self.biogeme_parameters.get_value(
            'second_derivatives', section='Estimation'
        )
        self.assertEqual(check_float, 1.0)

        check_str = self.biogeme_parameters.get_value('optimization_algorithm')
        self.assertEqual(check_str, 'simple_bounds')

        # Remove the directory after the test
        shutil.rmtree(test_dir)

    def test_from_default(self):
        """Test with default parameters"""
        test_dir = tempfile.mkdtemp()
        test_file = path.join(test_dir, 'any_file_name.toml')
        self.biogeme_parameters.read_file(file_name=test_file)

        with self.assertRaises(excep.BiogemeError):
            _ = self.biogeme_parameters.get_value('a_wrong_param', section='Estimation')

        check_int = self.biogeme_parameters.get_value('missing_data')
        self.assertEqual(check_int, 99999)

        check_float = self.biogeme_parameters.get_value(
            'second_derivatives', section='Estimation'
        )
        self.assertEqual(check_float, 1.0)

        check_str = self.biogeme_parameters.get_value('optimization_algorithm')
        self.assertEqual(check_str, 'simple_bounds')
        # Remove the directory after the test
        shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()
