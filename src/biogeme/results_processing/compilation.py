"""
Compilation of estimation results

Michel Bierlaire
Thu Oct 3 18:54:13 2024
"""

import glob
import logging
import os

import pandas as pd

from biogeme.results_processing.estimation_results import (
    EstimationResults,
    EstimateVarianceCovariance,
)
from biogeme.tools import ModelNames

logger = logging.getLogger(__name__)


def compile_estimation_results(
    dict_of_results: dict[str, EstimationResults | str],
    variance_covariance_type: EstimateVarianceCovariance = EstimateVarianceCovariance.ROBUST,
    statistics: tuple[str, ...] = (
        'Number of estimated parameters',
        'Sample size',
        'Final log likelihood',
        'Akaike Information Criterion',
        'Bayesian Information Criterion',
    ),
    include_parameter_estimates: bool = True,
    include_stderr: bool = False,
    include_t_test: bool = True,
    formatted: bool = True,
    use_short_names: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compile estimation results into a common table

    :param dict_of_results: dict of results, containing
        for each model the name, the ID and the results, or the name
        of the pickle file containing them.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param statistics: list of statistics to include in the summary
        table
    :param include_parameter_estimates: if True, the parameter
        estimates are included.
    :param include_stderr: if True, the robust standard errors
         of the parameters are included.
    :param include_t_test: if True, the t-test
         of the parameters are included.
    :param formatted: if True, a formatted string in included in the
         table results. If False, the numerical values are stored. Use
         "True" if you need to print the results. Use "False" if you
         need to use them for further calculation.
    :param use_short_names: if True, short names, such as Model_1,
        Model_2, are used to identify the model. It is nicer on for the
        reporting.
    :return: pandas dataframe with the requested results, and a dictionary reporting the
        specification of each model

    """
    model_names = ModelNames()

    def the_name(col: str) -> str:
        """Replace the name of a model by a shorter version for reporting

        :param col: name of the column, that is, name of the model.
        :return: name to be used in the reporting.
        """
        if use_short_names:
            return model_names(col)
        return col

    columns = [the_name(k) for k in dict_of_results.keys()]
    df = pd.DataFrame(columns=columns)

    configurations = {the_name(col): col for col in dict_of_results.keys()}

    for model, estimation_results in dict_of_results.items():
        if use_short_names:
            col = model_names(model)
        else:
            col = model
        if not isinstance(estimation_results, EstimationResults):
            try:
                estimation_results = EstimationResults.from_yaml_file(
                    filename=estimation_results
                )
            except FileNotFoundError:
                warning = f'Impossible to access result file {estimation_results}'
                logger.warning(warning)
                estimation_results = None

        if estimation_results is not None:
            stats_results = estimation_results.get_general_statistics()
            for s in statistics:
                df.loc[s, col] = stats_results[s]
            if include_parameter_estimates:
                for (
                    parameter_index,
                    parameter_name,
                ) in enumerate(estimation_results.beta_names):
                    parameter_value = estimation_results.get_parameter_value_from_index(
                        parameter_index=parameter_index
                    )
                    std_err_value = estimation_results.get_parameter_std_err_from_index(
                        parameter_index=parameter_index,
                        estimate_var_covar=variance_covariance_type,
                    )
                    t_test_value = estimation_results.get_parameter_t_test_from_index(
                        parameter_index=parameter_index,
                        estimate_var_covar=variance_covariance_type,
                    )
                    if formatted:

                        std_err_report = (
                            f'({std_err_value:.3g})' if include_stderr else ''
                        )
                        t_test_report = (
                            f'({t_test_value:.3g})' if include_t_test else ''
                        )
                        the_value = (
                            f'{parameter_value:.3g} {std_err_report} {t_test_report}'
                        )
                        row_std = ' (std)' if include_stderr else ''
                        row_t_test = ' (t-test)' if include_t_test else ''
                        row_title = f'{parameter_name}{row_std}{row_t_test}'
                        df.loc[row_title, col] = the_value
                    else:
                        df.loc[parameter_name, col] = parameter_value
                        if include_stderr:
                            df.loc[f'{parameter_name} (std)', col] = std_err_value
                        if include_t_test:
                            df.loc[f'{parameter_name} (t-test)', col] = t_test_value

    return df.fillna(''), configurations


def compile_results_in_directory(
    statistics: tuple[str, ...] = (
        'Number of estimated parameters',
        'Sample size',
        'Final log likelihood',
        'Akaike Information Criterion',
        'Bayesian Information Criterion',
    ),
    file_extension: str = 'yaml',
    variance_covariance_type: EstimateVarianceCovariance = EstimateVarianceCovariance.ROBUST,
    include_parameter_estimates: bool = True,
    include_stderr: bool = False,
    include_t_test: bool = True,
    formatted: bool = True,
    use_short_names: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """Compile estimation results found in the local directory into a
        common table. The results are supposed to be in a file with
        pickle extension.

    :param statistics: list of statistics to include in the summary
        table
    :param file_extension: extension of the files containing the estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param include_parameter_estimates: if True, the parameter
        estimates are included.
    :param include_stderr: if True, the robust standard errors
         of the parameters are included.
    :param include_t_test: if True, the t-test
         of the parameters are included.
    :param formatted: if True, a formatted string in included in the
         table results. If False, the numerical values are stored. Use
         "True" if you need to print the results. Use "False" if you
         need to use them for further calculation.
    :param use_short_names: if True, short names, such as Model_1,
        Model_2, are used to identify the model. It is nicer on for the
        reporting.
    :return: pandas dataframe with the requested results, and a dictionary reporting the
        specification of each model

    """
    files = glob.glob(f'*.{file_extension}')
    if not files:
        logger.warning(f'No .{file_extension} file found in {os.getcwd()}')
        return None

    the_dict = {k: k for k in files}
    return compile_estimation_results(
        dict_of_results=the_dict,
        variance_covariance_type=variance_covariance_type,
        statistics=statistics,
        include_parameter_estimates=include_parameter_estimates,
        include_stderr=include_stderr,
        include_t_test=include_t_test,
        formatted=formatted,
        use_short_names=use_short_names,
    )
