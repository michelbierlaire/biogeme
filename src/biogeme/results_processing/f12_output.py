"""
Generates a F12 output for ALOGIT

Michel Bierlaire
Thu Oct 3 10:09:52 2024
"""

import datetime
import logging
import os

from biogeme.version import get_version
from .estimation_results import (
    EstimateVarianceCovariance,
    EstimationResults,
    calculates_correlation_matrix,
)
from .pandas_output import get_pandas_estimated_parameters

logger = logging.getLogger(__name__)


def get_f12(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
) -> str:
    """F12 is a format used by the software ALOGIT to
    report estimation results.

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :return: results formatted in F12 format
    """
    if variance_covariance_type is None:
        variance_covariance_type = estimation_results.get_default_variance_covariance_matrix()
    covar_header = str(variance_covariance_type)

    # checkline1 = (
    #    '0000000001111111111222222222233333333334444444444'
    #    '5555555555666666666677777777778'
    # )
    # checkline2 = (
    #    '1234567890123456789012345678901234567890123456789'
    #    '0123456789012345678901234567890'
    # )

    results = ''

    # results += f'{checkline1}\n'
    # results += f'{checkline2}\n'

    # Line 1, title, characters 1-79
    results += f'{estimation_results.model_name[:79]: >79}\n'

    # Line 2, subtitle, characters 1-27, and time-date, characters 57-77
    t = f'From biogeme {get_version()}'
    d = f'{datetime.datetime.now()}'[:19]
    results += f'{t[:27]: <56}{d: <21}\n'

    # Line 3, "END" (this is historical!)
    results += 'END\n'

    # results += f'{checkline1}\n'
    # results += f'{checkline2}\n'

    # Line 4-(K+3), coefficient values
    #  characters 1-4, "   0" (again historical)
    #  characters 6-15, coefficient label, suggest using first 10
    #      characters of label in R
    #  characters 16-17, " F" (this indicates whether or not the
    #      coefficient is constrained)
    #  characters 19-38, coefficient value   20 chars
    #  characters 39-58, standard error      20 chars

    # mystats = estimation_results.get_general_statistics()
    table = get_pandas_estimated_parameters(
        estimation_results=estimation_results,
        variance_covariance_type=variance_covariance_type,
    )
    parameters_indices = table.index.to_list()
    for parameter_index in parameters_indices:
        values = table.loc[parameter_index]
        name = values['Name']
        results += '   0 '
        results += f'{name[:10]: >10}'
        if 'Active bound' in values:
            if values['Active bound'] == 1:
                results += ' T'
            else:
                results += ' F'
        else:
            results += ' F'
        results += ' '
        results += f' {values["Value"]: >+19.12e}'
        column_name = f'{covar_header} std err.'
        results += f' {values[column_name]: >+19.12e}'
        results += '\n'

    # Line K+4, "  -1" indicates end of coefficients
    results += '  -1\n'

    # results += f'{checkline1}\n'
    # results += f'{checkline2}\n'

    # Line K+5, statistics about run
    #   characters 1-8, number of observations        8 chars
    #   characters 9-27, likelihood-with-constants   19 chars
    #   characters 28-47, null likelihood            20 chars
    #   characters 48-67, final likelihood           20 chars

    results += f'{estimation_results.sample_size: >8}'
    # The cte log likelihood is not available. We put 0 instead.
    results += f' {0: >18}'
    if estimation_results.null_log_likelihood is not None:
        results += f' {estimation_results.null_log_likelihood: >+19.12e}'
    else:
        results += f' {0: >19}'
    results += f' {estimation_results.final_log_likelihood: >+19.12e}'
    results += '\n'

    # results += f'{checkline1}\n'
    # results += f'{checkline2}\n'

    # Line K+6, more statistics
    #   characters 1-4, number of iterations (suggest use 0)        4 chars
    #   characters 5-8, error code (please use 0)                   4 chars
    #   characters 9-29, time and date (sugg. repeat from line 2)  21 chars

    if "Number of iterations" in estimation_results.optimization_messages:
        results += (
            f'{estimation_results.optimization_messages["Number of iterations"]: >4}'
        )
    else:
        results += f'{0: >4}'
    results += f'{0: >4}'
    results += f'{d: >21}'
    results += '\n'

    # results += f'{checkline1}\n'
    # results += f'{checkline2}\n'

    # Lines (K+7)-however many we need, correlations*100000
    #   10 per line, fields of width 7
    #   The order of these is that correlation i,j (i>j) is in position
    #   (i-1)*(i-2)/2+j, i.e.
    #   (2,1) (3,1) (3,2) (4,1) etc.

    count = 0
    variance_covariance_matrix = estimation_results.get_variance_covariance_matrix(
        variance_covariance_type=variance_covariance_type
    )
    correlation_matrix = calculates_correlation_matrix(
        covariance=variance_covariance_matrix
    )
    for i, coefi in enumerate(estimation_results.beta_names):
        for j in range(0, i):
            try:
                corr = int(100000 * correlation_matrix[i][j])
            except OverflowError:
                corr = 999999
            results += f'{corr:7d}'
            count += 1
            if count % 10 == 0:
                results += '\n'
    results += '\n'
    return results


def generate_f12_file(
    estimation_results: EstimationResults,
    filename: str,
    overwrite=False,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
) -> None:
    """Generate a F12 file with the estimation results

    :param estimation_results: estimation results
    :param filename: name of the file
    :param overwrite: if True and the file exists, it is overwritten
    :param variance_covariance_type: select which type of variance-covariance matrix is used to generate the
        statistics. If None, the bootstrap one is used if available. If not available, the robust one.
    """
    if variance_covariance_type is None:
        variance_covariance_type = (
            estimation_results.get_default_variance_covariance_matrix()
        )
    if (
        variance_covariance_type == EstimateVarianceCovariance.BOOTSTRAP
        and estimation_results.bootstrap_time is None
    ):
        logger.warning(
            f'No bootstrap data is available. The robust variance-covariance matrix is used instead.'
        )
        variance_covariance_type = EstimateVarianceCovariance.ROBUST

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")

    with open(filename, 'w') as file:
        content = get_f12(
            estimation_results=estimation_results,
            variance_covariance_type=variance_covariance_type,
        )
        print(content, file=file)
    logger.info(f'File {filename} has been generated.')
