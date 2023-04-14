"""
Test the vns module

:author: Michel Bierlaire
:date: Thu Mar 16 15:50:51 2023
"""

import unittest
import numpy as np
import biogeme.exceptions as excep
import tempfile
from biogeme import vns
from biogeme.pareto import SetElement
from knapsack import Knapsack, Sack


class TestVns(unittest.TestCase):
    def setUp(self):
        utility = np.array([80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67])
        weight = np.array([84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78])
        CAPACITY = 300
        Sack.utility_data = utility
        Sack.weight_data = weight
        self.the_knapsack = Knapsack(utility, weight, CAPACITY)

        self.empty_sack = Sack.empty_sack(size=len(utility))

    def test_vns_errors(self):
        # Empty set
        the_pareto = vns.ParetoClass(max_neighborhood=2, pareto_file=None)
        with self.assertRaises(excep.BiogemeError):
            the_pareto = vns.vns(
                self.the_knapsack,
                [],
                the_pareto,
                number_of_neighbors=10,
            )

    def test_vns(self):
        the_pareto = vns.ParetoClass(max_neighborhood=2, pareto_file=None)

        the_pareto = vns.vns(
            self.the_knapsack,
            [self.empty_sack.get_element()],
            the_pareto,
            number_of_neighbors=2,
        )

    def test_vns_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = tmpdirname + 'tmp.pareto'

            the_pareto = vns.ParetoClass(max_neighborhood=2, pareto_file=filename)

            the_pareto = vns.vns(
                self.the_knapsack,
                [self.empty_sack.get_element()],
                the_pareto,
                number_of_neighbors=2,
            )


if __name__ == '__main__':
    unittest.main()
