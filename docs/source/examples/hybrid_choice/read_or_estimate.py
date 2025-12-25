"""
Read or estimate model parameters
=================================

Utility functions to either **read previously estimated parameters from disk**
or **run a new estimation** if no results are available.

This module provides a lightweight abstraction around Biogeme's estimation
routines, allowing scripts to be written in a reproducible way without
manually checking whether estimation results already exist.

Both maximum likelihood and Bayesian estimation paradigms are supported.

Michel Bierlaire (EPFL)
Thu Dec 25 2025, 08:28:26
"""

from biogeme.bayesian_estimation import BayesianResults
from biogeme.biogeme import BIOGEME
from biogeme.latent_variables import EstimationMode
from biogeme.results_processing import EstimationResults


def read_or_estimate(
    the_biogeme: BIOGEME, estimation_mode: EstimationMode, directory: str = '.'
) -> EstimationResults | BayesianResults:
    """Read estimation results from disk or estimate the model if needed.

    Depending on the selected estimation mode, this function attempts to read
    existing results from disk:

    - Bayesian estimation: results are read from a NetCDF file (``.nc``).
    - Maximum likelihood estimation: results are read from a YAML file
      (``.yaml``).

    If the corresponding file is not found, the model is estimated and the
    results are returned.

    This mechanism ensures that expensive estimations are not rerun
    unnecessarily while keeping the calling code simple and declarative.

    :param the_biogeme: Configured :class:`biogeme.biogeme.BIOGEME` object.
    :param estimation_mode: Estimation mode, either
        :class:`EstimationMode.BAYESIAN` or
        :class:`EstimationMode.MAXIMUM_LIKELIHOOD`.
    :param directory: Directory where result files are expected to be found.
    :return: Estimation results, either
        :class:`EstimationResults` or :class:`BayesianResults`.
    :raises ValueError: If an unsupported estimation mode is provided.
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
