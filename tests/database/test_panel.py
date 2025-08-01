import unittest

import pandas as pd

from biogeme.database import Database, PanelDatabase
from biogeme.exceptions import BiogemeError


class TestFlattenDataFrame(unittest.TestCase):

    def test_contiguous_observations(self):
        df = pd.DataFrame(
            {
                'ID': [1, 1, 2, 2],
                'X': [10, 20, 30, 40],
            }
        )
        database = Database(name='test', dataframe=df)
        panel_database = PanelDatabase(database=database, panel_column='ID')
        result, largest_group = panel_database.flatten_database(missing_data=-999)
        self.assertEqual(result.shape[0], 2)
        self.assertIn('X__panel__01', result.columns)
        self.assertIn('X__panel__02', result.columns)
        self.assertTrue((result['relevant___panel__01'] == 1).all())
        self.assertTrue((result['relevant___panel__02'] == 1).all())
        self.assertEqual(largest_group, 2)

    def test_non_contiguous_observations(self):
        df = pd.DataFrame(
            {
                'ID': [2, 1, 2, 1],
                'X': [30, 10, 40, 20],
            }
        )
        database = Database(name='test', dataframe=df)
        panel_database = PanelDatabase(database=database, panel_column='ID')
        result, largest_group = panel_database.flatten_database(missing_data=-999)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.loc[result['ID'] == 1, 'X__panel__01'].values[0], 10)
        self.assertEqual(result.loc[result['ID'] == 1, 'X__panel__02'].values[0], 20)
        self.assertEqual(largest_group, 2)

    def test_single_observation_per_individual(self):
        df = pd.DataFrame(
            {
                'ID': [1, 2],
                'X': [100, 200],
            }
        )
        database = Database(name='test', dataframe=df)
        panel_database = PanelDatabase(database=database, panel_column='ID')
        result, largest_group = panel_database.flatten_database(missing_data=-999)
        self.assertTrue((result['relevant___panel__01'] == 1).all())
        self.assertEqual(largest_group, 1)

    def test_missing_grouping_column(self):
        df = pd.DataFrame(
            {
                'A': [1, 2],
                'X': [10, 20],
            }
        )
        with self.assertRaises(KeyError):
            database = Database(name='test', dataframe=df)
            panel_database = PanelDatabase(database=database, panel_column='ID')
            _ = panel_database.flatten_database(missing_data=-999)

    def test_missing_data_padding(self):
        df = pd.DataFrame(
            {
                'ID': [1, 1, 2],
                'X': [10, 20, 30],
            }
        )
        database = Database(name='test', dataframe=df)
        panel_database = PanelDatabase(database=database, panel_column='ID')
        result, largest_group = panel_database.flatten_database(missing_data=-1)
        self.assertEqual(result.loc[result['ID'] == 2, 'X__panel__02'].values[0], -1)
        self.assertEqual(
            result.loc[result['ID'] == 2, 'relevant___panel__02'].values[0], 0
        )
        self.assertEqual(largest_group, 2)

    def test_column_naming(self):
        df = pd.DataFrame(
            {
                'ID': [1, 1],
                'X': [5, 6],
                'Y': [7, 8],
            }
        )
        database = Database(name='test', dataframe=df)
        panel_database = PanelDatabase(database=database, panel_column='ID')
        result, largest_group = panel_database.flatten_database(missing_data=0)
        self.assertIn('X__panel__01', result.columns)
        self.assertIn('Y__panel__02', result.columns)
        self.assertEqual(result['X__panel__01'].values[0], 5)
        self.assertEqual(result['Y__panel__02'].values[0], 8)
        self.assertEqual(largest_group, 2)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['ID', 'X'])
        with self.assertRaises(BiogemeError):
            database = Database(name='test', dataframe=df)


if __name__ == '__main__':
    unittest.main()
