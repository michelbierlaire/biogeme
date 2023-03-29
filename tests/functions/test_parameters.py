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
import biogeme.parameters as param
import biogeme.exceptions as excep
from biogeme.default_parameters import ParameterTuple
import biogeme.check_parameters as cp
import biogeme.optimization as opt

BLANKS = 25 * ' '

default_parameters = (
    ParameterTuple(
        name='second_derivatives',
        default=1.0,
        type=float,
        section='Estimation',
        description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
        check=(cp.zero_one, cp.is_number),
    ),
    ParameterTuple(
        name='tolerance',
        default=np.finfo(np.float64).eps ** 0.3333,
        type=float,
        section='Estimation',
        description='# float: the algorithm stops when this precision is reached',
        check=(cp.zero_one, cp.is_number),
    ),
    ParameterTuple(
        name='max_iterations',
        default=100,
        type=int,
        section='Estimation',
        description='# int: maximum number of iterations',
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='optimization_algorithm',
        default='simple_bounds',
        type=str,
        section='Estimation',
        description=(
            f'# str: optimization algorithm to be used for estimation.\n'
            f'{BLANKS}# Valid values: {list(opt.algorithms.keys())}'
        ),
        check=(cp.check_algo_name),
    ),
    ParameterTuple(
        name='generate_html',
        default='True',
        type=bool,
        section='Output',
        description=(
            '# bool: "True" if the HTML file\n'
            '{BLANKS}# with the results must be generated.'
        ),
        check=(cp.is_boolean),
    ),
    ParameterTuple(
        name='generate_pickle',
        default='True',
        type=bool,
        section='Output',
        description=(
            f'# bool: "True" if the pickle file\n'
            f'{BLANKS}# with the results must be generated.'
        ),
        check=(cp.is_boolean),
    ),
    ParameterTuple(
        name='number_of_threads',
        default=0,
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
        default=1000,
        type=int,
        section='MonteCarlo',
        description=('int: Number of draws for Monte-Carlo integration.'),
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='skip_audit',
        default='False',
        type=bool,
        section='Specification',
        description=(
            f'# bool: If True, does not check the validity of the\n'
            f'{BLANKS}# formulas. It may save significant amount of\n'
            f'{BLANKS}# time for large models and large data sets.\n'
        ),
        check=(cp.is_boolean),
    ),
    ParameterTuple(
        name='suggest_scales',
        default='True',
        type=bool,
        section='Specification',
        description=(
            f'# bool: If True, Biogeme suggests the scaling of\n'
            f'{BLANKS}# the variables in the database.'
        ),
        check=(cp.is_boolean),
    ),
    ParameterTuple(
        name='missing_data',
        default=99999,
        type=int,
        section='Specification',
        description=(
            f'# number: If one variable has this value, it is assumed\n'
            f'{BLANKS}# that a data is missing and an exception will\n'
            f'{BLANKS}# be triggered.'
        ),
        check=(cp.is_number),
    ),
    ParameterTuple(
        name='seed',
        default=0,
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
        default='True',
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
        check=(cp.is_boolean),
    ),
    ParameterTuple(
        name='a_param',
        default='lopsum',
        type=str,
        section='Estimation',
        description='',
        check=None,
    ),
    ParameterTuple(
        name='a_param',
        default='anything',
        type=str,
        section='Specification',
        description='',
        check=None,
    ),
)


class TestParameters(unittest.TestCase):
    def setUp(self):
        """Prepare for the tests"""
        self.the_parameters = param.Parameters(default_parameters)

    def test_get_param_tuple_1(self):
        """Test the retrieval of parameter without the section name"""
        the_tuple = self.the_parameters.get_param_tuple(name='missing_data')
        self.assertEqual(the_tuple.name, 'missing_data')
        self.assertEqual(the_tuple.default, 99999)
        self.assertEqual(the_tuple.section, 'Specification')

    def test_get_param_tuple_2(self):
        """Test the retrieval of parameter with the section name"""
        the_tuple = self.the_parameters.get_param_tuple(
            name='missing_data', section='Specification'
        )
        self.assertEqual(the_tuple.name, 'missing_data')
        self.assertEqual(the_tuple.default, 99999)
        self.assertEqual(the_tuple.section, 'Specification')

    def test_get_param_tuple_3(self):
        """Test the error if the parameter does not appear in the
        mentioned section
        """
        with self.assertRaises(excep.biogemeError):
            _ = self.the_parameters.get_param_tuple(
                name='missing_data', section='Estimation'
            )

        with self.assertRaises(excep.biogemeError):
            _ = self.the_parameters.get_param_tuple(name='unknown_parameter')

        with self.assertRaises(excep.biogemeError):
            _ = self.the_parameters.get_param_tuple(
                name='unknown_parameter', section='Estimation'
            )

    def test_get_param_tuple_4(self):
        """Test the error if the parameter does appears in two
        sections, and the section is not specified.
        """
        with self.assertRaises(excep.biogemeError):
            _ = self.the_parameters.get_param_tuple(
                name='a_param',
            )

    def test_get_param_tuple_5(self):
        """Test the retrieval of a parameter that appears in two
        sections, with the section name

        """
        the_tuple = self.the_parameters.get_param_tuple(
            name='a_param', section='Estimation'
        )
        self.assertEqual(the_tuple.name, 'a_param')
        self.assertEqual(the_tuple.default, 'lopsum')
        self.assertEqual(the_tuple.section, 'Estimation')

    def test_get_param_tuple_6(self):
        """Test the retrieval of a parameter that appears in two
        section, with the section name

        """
        the_tuple = self.the_parameters.get_param_tuple(
            name='a_param', section='Specification'
        )
        self.assertEqual(the_tuple.name, 'a_param')
        self.assertEqual(the_tuple.default, 'anything')
        self.assertEqual(the_tuple.section, 'Specification')

    def test_check_parameter_value(self):
        a_tuple = ParameterTuple(
            name='param_name',
            default=2.0,
            type=float,
            section='Estimation',
            description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
            check=(cp.zero_one, cp.is_number),
        )
        ok, msg = self.the_parameters.check_parameter_value(a_tuple, 5)
        self.assertFalse(ok)

    def test_set_value_from_tuple(self):
        a_tuple = ParameterTuple(
            name='param_name',
            default=2.0,
            type=float,
            section='Estimation',
            description='# float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated',
            check=(cp.zero_one, cp.is_number),
        )
        self.the_parameters.set_value_from_tuple(a_tuple, 1)
        self.assertEqual(self.the_parameters.values[a_tuple], 1)
        with self.assertRaises(excep.biogemeError):
            self.the_parameters.set_value_from_tuple(a_tuple, 2)

    def test_set_get_value(self):
        value = 1.0e-3
        self.the_parameters.set_value('tolerance', value)
        check = self.the_parameters.get_value('tolerance')
        self.assertEqual(value, check)


if __name__ == '__main__':
    unittest.main()
