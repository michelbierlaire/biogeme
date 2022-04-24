"""
Test the segmentation module

:author: Michel Bierlaire
:date: Fri Dec 31 11:01:24 2021

"""

import unittest
from biogeme.expressions import Beta, Variable
import biogeme.segmentation as seg

class test_segmentation(unittest.TestCase):
    def test_segmentation(self):
        variable_1 = Variable('Variable1')
        variable_2 = Variable('Variable2')
        parameter = Beta('test', 0, None, None, 0)
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
        expr_mapping = seg.create_segmented_parameter(parameter, mapping_1)
        result_mapping = (
            '{1: test_var1(0), '
            '2: test_var2(0), '
            '3: test_var3(0), '
            '4: test_var4(0), '
            '5: test_var5(0)}')
        self.assertEqual(str(expr_mapping), result_mapping)
        combined_expr = seg.combine_segmented_expressions(variable_1, expr_mapping)

        result_combined = (
            'bioMultSum((test_var1(0) * (Variable1 == `1`)), '
            '(test_var2(0) * (Variable1 == `2`)), '
            '(test_var3(0) * (Variable1 == `3`)), '
            '(test_var4(0) * (Variable1 == `4`)), '
            '(test_var5(0) * (Variable1 == `5`)))'
        )
        self.assertEqual(str(combined_expr), result_combined)
        first_segmentation = seg.DiscreteSegmentationTuple(
            variable=variable_1,
            mapping=mapping_1
        )

        second_segmentation = seg.DiscreteSegmentationTuple(
            variable=variable_2,
            mapping=mapping_2
        )

        tuple_of_segmentations = [
            first_segmentation,
            second_segmentation,
        ]
                                                      
        segmented_expression = seg.segment_parameter(parameter, tuple_of_segmentations)
        result_segment = (
            'bioMultSum('
            '(bioMultSum((test_my1_var1(0) * (Variable1 == `1`)), '
            '(test_my1_var2(0) * (Variable1 == `2`)), '
            '(test_my1_var3(0) * (Variable1 == `3`)), '
            '(test_my1_var4(0) * (Variable1 == `4`)), '
            '(test_my1_var5(0) * (Variable1 == `5`))) * (Variable2 == `10`)), '
            '(bioMultSum((test_my2_var1(0) * (Variable1 == `1`)), '
            '(test_my2_var2(0) * (Variable1 == `2`)), '
            '(test_my2_var3(0) * (Variable1 == `3`)), '
            '(test_my2_var4(0) * (Variable1 == `4`)), '
            '(test_my2_var5(0) * (Variable1 == `5`))) * (Variable2 == `20`)), '
            '(bioMultSum((test_my3_var1(0) * (Variable1 == `1`)), '
            '(test_my3_var2(0) * (Variable1 == `2`)), '
            '(test_my3_var3(0) * (Variable1 == `3`)), '
            '(test_my3_var4(0) * (Variable1 == `4`)), '
            '(test_my3_var5(0) * (Variable1 == `5`))) * (Variable2 == `30`)), '
            '(bioMultSum((test_my4_var1(0) * (Variable1 == `1`)), '
            '(test_my4_var2(0) * (Variable1 == `2`)), '
            '(test_my4_var3(0) * (Variable1 == `3`)), '
            '(test_my4_var4(0) * (Variable1 == `4`)), '
            '(test_my4_var5(0) * (Variable1 == `5`))) * (Variable2 == `40`)), '
            '(bioMultSum((test_my5_var1(0) * (Variable1 == `1`)), '
            '(test_my5_var2(0) * (Variable1 == `2`)), '
            '(test_my5_var3(0) * (Variable1 == `3`)), '
            '(test_my5_var4(0) * (Variable1 == `4`)), '
            '(test_my5_var5(0) * (Variable1 == `5`))) * (Variable2 == `50`)))'
            
        )
        self.assertEqual(str(segmented_expression), result_segment)

        combined_code = seg.code_to_combine_segmented_expressions(
            variable_1,
            expr_mapping,
            'beta'
        )
        print(combined_code)
        
        segmented_code = seg.code_to_segment_parameter(parameter, tuple_of_segmentations)
        print(segmented_code)
        
if __name__ == '__main__':
    unittest.main()
