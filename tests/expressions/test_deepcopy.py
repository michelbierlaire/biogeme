import copy
import unittest

from biogeme.expressions import (
    MultipleSum,
    PanelLikelihoodTrajectory,
    Variable,
    add_prefix_suffix_to_all_variables,
)


class TestDeepCopyInPanelLikelihoodTrajectory(unittest.TestCase):
    def test_deepcopy_children(self):
        # Original expression tree
        original_var = Variable('x')
        expression = MultipleSum([original_var, original_var + 1])
        panel_expr = PanelLikelihoodTrajectory(expression)
        panel_expr.set_maximum_number_of_observations_per_individual(3)

        # Generate copies
        copies = [
            copy.deepcopy(panel_expr.child)
            for _ in range(panel_expr.maximum_number_of_observations_per_individual)
        ]

        # Ensure copies are not the same object
        self.assertTrue(all(c is not panel_expr.child for c in copies))

        # Ensure modifying one copy does not affect the others
        number_of_modifications = add_prefix_suffix_to_all_variables(
            expr=copies[0], prefix='', suffix='_mod'
        )

        self.assertNotEqual(str(copies[0]), str(copies[1]))
        self.assertNotEqual(str(copies[0]), str(panel_expr.child))


if __name__ == '__main__':
    unittest.main()
