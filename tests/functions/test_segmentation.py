"""
Test the segmentation module

:author: Michel Bierlaire
:date: Fri Dec 31 11:01:24 2021

"""

import unittest
from biogeme.expressions import Beta, Variable
import biogeme.segmentation as seg
import biogeme.exceptions as excep


class test_segmentation(unittest.TestCase):
    def test_one_segmentation(self):
        x = Variable('x')
        mapping = {1: '1st', 2: '2nd', 3: '3rd'}

        first_tuple = seg.DiscreteSegmentationTuple(variable=x, mapping=mapping)
        second_tuple = seg.DiscreteSegmentationTuple(
            variable=x, mapping=mapping, reference='3rd'
        )
        with self.assertRaises(excep.BiogemeError):
            third_tuple = seg.DiscreteSegmentationTuple(
                variable=x, mapping=mapping, reference='anything'
            )

        beta = Beta('beta', 1, None, None, 0)

        one_segmentation = seg.OneSegmentation(beta, first_tuple)

        beta_name = one_segmentation.beta_name('2nd')
        self.assertEqual(beta_name, 'beta_2nd')

        beta_expression = one_segmentation.beta_expression('2nd')
        expected_start = "Beta('beta_2nd',"
        actual_start = str(beta_expression)[: len(expected_start)]
        self.assertEqual(actual_start, expected_start)

        beta_code = one_segmentation.beta_code('2nd', assignment=False)
        expected_code = "Beta('beta_2nd', 1, None, None, 0)"
        self.assertEqual(beta_code, expected_code)

        beta_code_assignment = one_segmentation.beta_code('2nd', assignment=True)
        expected_code_assignment = "beta_2nd = Beta('beta_2nd', 1, None, None, 0)"
        self.assertEqual(beta_code_assignment, expected_code_assignment)

        list_of_expressions = [str(e) for e in one_segmentation.list_of_expressions()]
        expected_start_list = [
            "(Beta('beta_2nd',",
            "(Beta('beta_3rd',",
        ]
        for start, expr in zip(expected_start_list, list_of_expressions):
            self.assertTrue(expr.startswith(start))

        list_of_code = one_segmentation.list_of_code()
        expected_list = [
            "beta_2nd * (Variable('x') == 2)",
            "beta_3rd * (Variable('x') == 3)",
        ]
        self.assertCountEqual(list_of_code, expected_list)

    def test_segmentation(self):
        variable_1 = Variable('Variable1')
        variable_2 = Variable('Variable2')
        parameter = Beta('test', 0, -10, 10, 0)
        mapping_1 = {
            1: 'var1',
            2: 'var2',
            3: 'var3',
            4: 'var4',
            5: 'var5',
        }
        mapping_2 = {
            10: 'my1',
            20: 'my2',
            30: 'my3',
            40: 'my4',
            50: 'my5',
        }
        first_segmentation = seg.DiscreteSegmentationTuple(
            variable=variable_1, mapping=mapping_1, reference='var1'
        )
        second_segmentation = seg.DiscreteSegmentationTuple(
            variable=variable_2, mapping=mapping_2, reference='my1'
        )

        the_segmentation = seg.Segmentation(
            parameter, (first_segmentation, second_segmentation)
        )

        beta_code = the_segmentation.beta_code()
        expected_code = "Beta('test', 0, -10, 10, 0)"
        self.assertEqual(beta_code, expected_code)

        segmented_expression = the_segmentation.segmented_beta()
        expected_start = "bioMultSum(Beta('test', 0, -10, 10, 0),"
        actual_start = str(segmented_expression)[: len(expected_start)]
        self.assertEqual(actual_start, expected_start)

        segmented_code = the_segmentation.segmented_code()
        result_code = (
            "test_var2 = Beta('test_var2', 0, None, None, 0)\n"
            "test_var3 = Beta('test_var3', 0, None, None, 0)\n"
            "test_var4 = Beta('test_var4', 0, None, None, 0)\n"
            "test_var5 = Beta('test_var5', 0, None, None, 0)\n"
            "test_my2 = Beta('test_my2', 0, None, None, 0)\n"
            "test_my3 = Beta('test_my3', 0, None, None, 0)\n"
            "test_my4 = Beta('test_my4', 0, None, None, 0)\n"
            "test_my5 = Beta('test_my5', 0, None, None, 0)\n"
            "segmented_test = bioMultSum(["
            "Beta('test', 0, -10, 10, 0), "
            "test_var2 * (Variable('Variable1') == 2), "
            "test_var3 * (Variable('Variable1') == 3), "
            "test_var4 * (Variable('Variable1') == 4), "
            "test_var5 * (Variable('Variable1') == 5), "
            "test_my2 * (Variable('Variable2') == 20), "
            "test_my3 * (Variable('Variable2') == 30), "
            "test_my4 * (Variable('Variable2') == 40), "
            "test_my5 * (Variable('Variable2') == 50)])"
        )
        self.assertEqual(str(segmented_code), result_code)


if __name__ == '__main__':
    unittest.main()
