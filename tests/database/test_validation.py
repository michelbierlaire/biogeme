import unittest
import pandas as pd
import numpy as np
from biogeme.validation.prepare_validation import split


class TestSplitFunction(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame for tests
        self.df = pd.DataFrame(
            {
                'id': range(100),
                'group': [i // 10 for i in range(100)],
                'feature': np.random.rand(100),
            }
        )

    def test_basic_split(self):
        result = split(self.df, slices=5)
        self.assertEqual(len(result), 5)
        total_rows = sum(len(ev.estimation) + len(ev.validation) for ev in result)
        self.assertEqual(total_rows, 5 * len(self.df))
        for ev in result:
            self.assertEqual(len(ev.estimation) + len(ev.validation), len(self.df))

    def test_split_with_groups(self):
        result = split(self.df, slices=5, groups='group')
        self.assertEqual(len(result), 5)
        all_ids = set(self.df['id'])
        for ev in result:
            group_ids_val = set(self.df.loc[ev.validation, 'group'].unique())
            group_ids_est = set(self.df.loc[ev.estimation, 'group'].unique())
            self.assertTrue(group_ids_val.isdisjoint(group_ids_est))
            combined_ids = set(ev.estimation).union(set(ev.validation))
            self.assertSetEqual(combined_ids, set(self.df.index))

    def test_invalid_slices(self):
        with self.assertRaises(ValueError):
            split(self.df, slices=1)

    def test_split_preserves_all_data(self):
        result = split(self.df, slices=4)
        combined_ids = pd.concat([self.df.loc[ev.validation] for ev in result])['id']
        self.assertSetEqual(set(combined_ids), set(self.df['id']))


if __name__ == '__main__':
    unittest.main()
