"""
Test the sampling module

:author: Michel Bierlaire
:date: Sun Jan  8 18:03:42 2023
"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
import numpy as np
import pandas as pd
import biogeme.sampling as sam
import biogeme.exceptions as excep


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.nbr_of_alt = 20
        self.nbr_of_obs = 10
        self.alternatives = pd.DataFrame(
            {
                'ID': list(range(self.nbr_of_alt)),
                'Attr1': np.random.rand(self.nbr_of_alt),
            }
        )
        self.individuals = pd.DataFrame(
            {
                'Char1': np.random.randint(2, size=self.nbr_of_obs),
                'choice': np.random.randint(self.nbr_of_alt, size=self.nbr_of_obs),
            }
        )

    def test_sampling(self):
        set1 = set(range(10))
        set2 = set(range(10, 20))
        partition = (
            sam.StratumTuple(subset=set1, sample_size=3),
            sam.StratumTuple(subset=set2, sample_size=3),
        )
        the_sample = sam.sampling_of_alternatives(
            partition=partition,
            individuals=self.individuals,
            choice_column='choice',
            alternatives=self.alternatives,
            id_column='ID',
            always_include_chosen=True,
        )
        the_sample = the_sample.astype({f'ID_{i}': int for i in range(6)})
        pd.testing.assert_series_equal(
            the_sample['choice'], the_sample['ID_0'], check_names=False
        )

    def test_partition_1(self):
        set1 = set([1, 2, 3])
        set2 = set([3, 4, 5])
        partition = (
            sam.StratumTuple(subset=set1, sample_size=3),
            sam.StratumTuple(subset=set2, sample_size=3),
        )
        with self.assertRaises(excep.BiogemeError):
            the_sample = sam.sampling_of_alternatives(
                partition=partition,
                individuals=self.individuals,
                choice_column='choice',
                alternatives=self.alternatives,
                id_column='ID',
                always_include_chosen=True,
            )

    def test_partition_2(self):
        set1 = set([1, 2])
        set2 = set([4, 5])
        partition = (
            sam.StratumTuple(subset=set1, sample_size=3),
            sam.StratumTuple(subset=set2, sample_size=3),
        )
        with self.assertRaises(excep.BiogemeError):
            the_sample = sam.sampling_of_alternatives(
                partition=partition,
                individuals=self.individuals,
                choice_column='choice',
                alternatives=self.alternatives,
                id_column='ID',
                always_include_chosen=True,
            )

    def test_partition_3(self):
        set1 = set([1, 2, 3])
        set2 = set([4, 5, 6])
        partition = (
            sam.StratumTuple(subset=set1, sample_size=3),
            sam.StratumTuple(subset=set2, sample_size=3),
        )
        with self.assertRaises(excep.BiogemeError):
            the_sample = sam.sampling_of_alternatives(
                partition=partition,
                individuals=self.individuals,
                choice_column='choice',
                alternatives=self.alternatives,
                id_column='ID',
                always_include_chosen=True,
            )


if __name__ == '__main__':
    unittest.main()
