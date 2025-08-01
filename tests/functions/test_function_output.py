import unittest
import numpy as np

from biogeme.function_output import (
    convert_to_dict,
    FunctionOutput,
    NamedFunctionOutput,
    DisaggregateFunctionOutput,
)


class TestConvertToDict(unittest.TestCase):
    def test_normal_conversion(self):
        sequence = ['apple', 'banana', 'cherry']
        the_map = {'first': 0, 'second': 1, 'third': 2}
        expected = {'first': 'apple', 'second': 'banana', 'third': 'cherry'}
        self.assertEqual(convert_to_dict(sequence, the_map), expected)

    def test_index_out_of_bounds(self):
        sequence = ['apple', 'banana', 'cherry']
        the_map = {'first': 0, 'third': 10}
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
        named_output = NamedFunctionOutput(function_output=f_output, mapping=mapping)
        expected_gradient = {'zero': 0.5, 'one': 1.5}
        expected_hessian = {'zero': {'zero': 2, 'one': 0}, 'one': {'zero': 0, 'one': 3}}
        self.assertEqual(named_output.function, 3.5)
        self.assertEqual(named_output.gradient, expected_gradient)
        self.assertEqual(named_output.hessian, expected_hessian)

    def test_with_none_gradients_and_hessians(self):
        f_output = FunctionOutput(1.0)
        mapping = {'zero': 0, 'one': 1}
        named_output = NamedFunctionOutput(function_output=f_output, mapping=mapping)
        self.assertIsNone(named_output.gradient)
        self.assertIsNone(named_output.hessian)


class TestNamedFunctionOutputFromDisaggregate(unittest.TestCase):
    def test_initialization(self):
        output = DisaggregateFunctionOutput(
            functions=np.array([3.0]),
            gradients=np.array([[1.0, 1.5]]),
            hessians=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            bhhhs=np.array([[[2.0, 2.0], [2.0, 2.0]]]),
        )
        mapping = {'zero': 0, 'one': 1}
        unique = output.unique_entry()
        named_output = NamedFunctionOutput(function_output=unique, mapping=mapping)
        expected_bhhh = {
            'zero': {'zero': 2.0, 'one': 2.0},
            'one': {'zero': 2.0, 'one': 2.0},
        }
        self.assertEqual(named_output.bhhh, expected_bhhh)


class TestDisaggregateFunctionOutput(unittest.TestCase):
    def test_unique_entry_single(self):
        output = DisaggregateFunctionOutput(
            functions=np.array([1.0]),
            gradients=np.array([[1.0, 2.0]]),
            hessians=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            bhhhs=np.array([[[2.0, 2.0], [2.0, 2.0]]]),
        )
        unique = output.unique_entry()
        self.assertIsInstance(unique, FunctionOutput)
        self.assertEqual(unique.function, 1.0)
        self.assertTrue((unique.gradient == np.array([1.0, 2.0])).all())
        self.assertTrue((unique.hessian == np.array([[1.0, 0.0], [0.0, 1.0]])).all())
        self.assertTrue((unique.bhhh == np.array([[2.0, 2.0], [2.0, 2.0]])).all())

    def test_unique_entry_multiple(self):
        output = DisaggregateFunctionOutput(functions=np.array([1.0, 2.0]))
        self.assertIsNone(output.unique_entry())

    def test_length(self):
        output = DisaggregateFunctionOutput(functions=np.array([1.0, 2.0, 3.0]))
        self.assertEqual(len(output), 3)


if __name__ == '__main__':
    unittest.main()
