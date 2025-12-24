"""

Read of estimate
================

Function to estimate the parameters, or read them from a file, if available.

Michel Bierlaire, EPFL
Mon May 05 2025, 18:59:34
"""

from biogeme.bayesian_estimation import BayesianResults
from biogeme.biogeme import BIOGEME
from biogeme.latent_variables import EstimationMode
from biogeme.results_processing import EstimationResults


def read_or_estimate(
    the_biogeme: BIOGEME, estimation_mode: EstimationMode, directory: str = '.'
) -> EstimationResults | BayesianResults:
    """
    Function to estimate the parameters, or read them from a file, if available.

    :param the_biogeme: Biogeme object.
    :param estimation_mode: EstimationMode.BAYESIAN or EstimationMode.MAXIMUM_LIKELIHOOD
    :param directory: directory where the yaml file is supposed to be.

    :return: estimation results.
    """
    if estimation_mode == EstimationMode.BAYESIAN:
        try:
            filename = f'{directory}/{the_biogeme.model_name}.nc'
            results = BayesianResults.from_netcdf(filename=filename)
            print(f'Results are read from the file {filename}.')
        except FileNotFoundError:
            print('Parameters are being estimated.')
            results = the_biogeme.bayesian_estimation()
        return results

    if estimation_mode != EstimationMode.MAXIMUM_LIKELIHOOD:
        raise ValueError(f'Unknown estimation mode: {estimation_mode}')

    try:
        filename = f'{directory}/{the_biogeme.model_name}.yaml'
        results = EstimationResults.from_yaml_file(filename=filename)
        print(f'Results are read from the file {filename}.')
    except FileNotFoundError:
        print('Parameters are being estimated.')
        results = the_biogeme.estimate()
    return results
