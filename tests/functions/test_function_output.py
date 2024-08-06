import unittest
import numpy as np

from biogeme.function_output import (
    convert_to_dict,
    FunctionOutput,
    NamedFunctionOutput,
    BiogemeFunctionOutput,
    NamedBiogemeFunctionOutput,
)


# Assuming all classes and functions are imported here


class TestConvertToDict(unittest.TestCase):
    def test_normal_conversion(self):
        sequence = ['apple', 'banana', 'cherry']
        the_map = {'first': 0, 'second': 1, 'third': 2}
        expected = {'first': 'apple', 'second': 'banana', 'third': 'cherry'}
        self.assertEqual(convert_to_dict(sequence, the_map), expected)

    def test_index_out_of_bounds(self):
        sequence = ['apple', 'banana', 'cherry']
        the_map = {'first': 0, 'third': 10}  # Invalid index
        with self.assertRaises(IndexError):
            convert_to_dict(sequence, the_map)

    def test_empty_inputs(self):
        self.assertEqual(convert_to_dict([], {}), {})


class TestFunctionOutput(unittest.TestCase):
    def test_initialization(self):
        output = FunctionOutput(2.0, np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        self.assertEqual(output.function, 2.0)
        self.assertTrue((output.gradient == np.array([1, 2])).all())
        self.assertTrue((output.hessian == np.array([[1, 0], [0, 1]])).all())


class TestNamedFunctionOutput(unittest.TestCase):
    def test_initialization(self):
        f_output = FunctionOutput(3.5, np.array([0.5, 1.5]), np.array([[2, 0], [0, 3]]))
        mapping = {'zero': 0, 'one': 1}
        named_output = NamedFunctionOutput(f_output, mapping)
        expected_gradient = {'zero': 0.5, 'one': 1.5}
        expected_hessian = {'zero': {'zero': 2, 'one': 0}, 'one': {'zero': 0, 'one': 3}}
        self.assertEqual(named_output.function, 3.5)
        self.assertEqual(named_output.gradient, expected_gradient)
        self.assertEqual(named_output.hessian, expected_hessian)

    def test_with_none_gradients_and_hessians(self):
        f_output = FunctionOutput(1.0)
        mapping = {'zero': 0, 'one': 1}
        named_output = NamedFunctionOutput(f_output, mapping)
        self.assertIsNone(named_output.gradient)
        self.assertIsNone(named_output.hessian)


class TestBiogemeFunctionOutput(unittest.TestCase):
    def test_initialization(self):
        b_output = BiogemeFunctionOutput(
            4.0, np.array([1, 1]), np.array([[1, 0], [0, 1]]), np.array([[1]])
        )
        self.assertEqual(b_output.function, 4.0)
        self.assertTrue((b_output.bhhh == np.array([[1]])).all())


class TestNamedBiogemeFunctionOutput(unittest.TestCase):
    def test_initialization(self):
        b_output = BiogemeFunctionOutput(
            5.0,
            np.array([2, 3]),
            np.array([[4, 0], [0, 4]]),
            np.array([[2, 2], [2, 2]]),
        )
        mapping = {'zero': 0, 'one': 1}
        named_b_output = NamedBiogemeFunctionOutput(b_output, mapping)
        expected_bhhh = {'zero': {'zero': 2, 'one': 2}, 'one': {'zero': 2, 'one': 2}}
        self.assertEqual(named_b_output.bhhh, expected_bhhh)


if __name__ == '__main__':
    unittest.main()
