"""
Test the parameters module

:author: Michel Bierlaire
:date: Tue Nov 29 10:26:32 2022

"""
from os import path
import shutil
import tempfile
import unittest
from biogeme import toml
import biogeme.exceptions as excep

FILE_CONTENT = """
[Specification]
skip_audit = "False" # bool: If True, does not check the validity of the
                         # formulas. It may save significant amount of
                         # time for large models and large data sets.

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


class TestToml(unittest.TestCase):
    """Tests for the biogeme.toml module"""

    def test_from_file(self):
        """Read the parameters from file"""
        # Create a temporary directory
        test_dir = tempfile.mkdtemp()
        test_file = path.join(test_dir, 'biogeme.toml')
        with open(test_file, 'w', encoding='utf-8') as f:
            print(FILE_CONTENT, file=f)
        the_toml = toml.Toml(parameter_file=test_file)
        check_bool = the_toml.parameters.get_value('skip_audit')
        self.assertEqual(check_bool, False)

        with self.assertRaises(excep.biogemeError):
            _ = the_toml.parameters.get_value('a_param', section='Estimation')

        check_int = the_toml.parameters.get_value('missing_data')
        self.assertEqual(check_int, 99999)

        check_float = the_toml.parameters.get_value('second_derivatives')
        self.assertEqual(check_float, 1.0)

        check_str = the_toml.parameters.get_value('optimization_algorithm')
        self.assertEqual(check_str, 'simple_bounds')

        # Remove the directory after the test
        shutil.rmtree(test_dir)

    def test_from_default(self):
        """Test with default parameters"""
        test_dir = tempfile.mkdtemp()
        test_file = path.join(test_dir, 'any_file_name.toml')
        the_toml = toml.Toml(parameter_file=test_file)
        check_bool = the_toml.parameters.get_value('skip_audit')
        self.assertEqual(check_bool, False)

        with self.assertRaises(excep.biogemeError):
            _ = the_toml.parameters.get_value('a_param', section='Estimation')

        check_int = the_toml.parameters.get_value('missing_data')
        self.assertEqual(check_int, 99999)

        check_float = the_toml.parameters.get_value('second_derivatives')
        self.assertEqual(check_float, 1.0)

        check_str = the_toml.parameters.get_value('optimization_algorithm')
        self.assertEqual(check_str, 'simple_bounds')
        # Remove the directory after the test
        shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()
