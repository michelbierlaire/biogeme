import io
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from biogeme.results_processing import RawEstimationResults
from biogeme.results_processing.recycle_pickle import (
    Beta,
    BiogemeUnpickler,
    RawResults,
    pickle_to_yaml,
    read_pickle_biogeme,
)


class TestRecyclePickle(unittest.TestCase):

    def setUp(self):
        # Dummy beta parameters
        beta = Beta()
        beta.name = 'ASC_CAR'
        beta.value = 1.23
        beta.lb = -10
        beta.ub = 10

        # Dummy results object
        self.raw_results = RawResults()
        self.raw_results.modelName = 'TestModel'
        self.raw_results.userNotes = 'Test note'
        self.raw_results.betas = [beta]
        self.raw_results.g = np.array([1.0])
        self.raw_results.H = np.array([[1.0]])
        self.raw_results.bhhh = np.array([[1.0]])
        self.raw_results.nullLogLike = -100.0
        self.raw_results.initLogLike = -120.0
        self.raw_results.logLike = -95.0
        self.raw_results.dataname = 'testdata'
        self.raw_results.sampleSize = 100
        self.raw_results.numberOfObservations = 100
        self.raw_results.monte_carlo = False
        self.raw_results.numberOfDraws = 0
        self.raw_results.typesOfDraws = {}
        self.raw_results.excludedData = 0
        self.raw_results.drawsProcessingTime = timedelta(seconds=2)
        self.raw_results.optimizationMessages = {'message': 'success'}
        self.raw_results.convergence = True
        self.raw_results.bootstrap = np.array([[1.1]])
        self.raw_results.bootstrap_time = timedelta(seconds=1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_read_pickle_biogeme_success(self, mock_pickle_load, mock_file):
        mock_pickle_load.return_value = self.raw_results
        result = read_pickle_biogeme('dummy.pkl')
        self.assertIsInstance(result, RawEstimationResults)
        self.assertEqual(result.model_name, 'TestModel')
        self.assertEqual(result.beta_names, ['ASC_CAR'])

    @patch('biogeme.results_processing.recycle_pickle.read_pickle_biogeme')
    @patch('biogeme.results_processing.recycle_pickle.serialize_to_yaml')
    def test_pickle_to_yaml(self, mock_serialize, mock_reader):
        mock_result = MagicMock(spec=RawEstimationResults)
        mock_reader.return_value = mock_result

        pickle_file = 'input.pkl'
        yaml_file = 'output.yaml'
        pickle_to_yaml(pickle_file, yaml_file)

        mock_reader.assert_called_once_with(filename=pickle_file)
        mock_serialize.assert_called_once_with(data=mock_result, filename=yaml_file)

    def test_biogeme_unpickler_custom_types(self):
        dummy_file = io.BytesIO(b"dummy")
        unpickler = BiogemeUnpickler(file=dummy_file)
        self.assertIs(unpickler.find_class('biogeme.results', 'beta'), Beta)
        self.assertIs(unpickler.find_class('biogeme.results', 'rawResults'), RawResults)

    def test_beta_class_fields(self):
        beta = Beta()
        self.assertIsNone(beta.name)
        self.assertIsNone(beta.stdErr)
        beta.name = 'B_COST'
        self.assertEqual(beta.name, 'B_COST')


if __name__ == '__main__':
    unittest.main()
