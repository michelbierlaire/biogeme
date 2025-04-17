"""
Recycle old pickle files and transform them into YAML files

Michel Bierlaire
Wed Oct 2 06:43:33 2024
"""

import pickle
from datetime import timedelta
from typing import Any

import numpy as np

from .raw_estimation_results import RawEstimationResults, serialize_to_yaml
from ..exceptions import BiogemeError


class RawResults:
    """Class containing the raw results from the estimation"""

    def __init__(
        self,
    ):
        """
        Constructor
        """
        self.modelName = None
        self.userNotes = None
        self.nparam = None
        self.betaValues = None
        self.betaNames = None
        self.initLogLike = None
        self.nullLogLike = None
        self.betas = None
        self.logLike = None
        self.g = None
        self.H = None
        self.bhhh = None
        self.dataname = None
        self.sampleSize = None
        self.numberOfObservations = None
        self.monte_carlo = None
        self.numberOfDraws = None
        self.typesOfDraws = None
        self.excludedData = None
        self.drawsProcessingTime = None
        self.gradientNorm = None
        self.optimizationMessages = None
        self.convergence = None
        self.htmlFileName = None
        self.F12FileName = None
        self.latexFileName = None
        self.pickleFileName = None
        self.bootstrap = None
        self.bootstrap_time = None
        self.secondOrderTable = None


class Beta:
    """Class gathering the information related to the parameters
    of the model
    """

    def __init__(self):
        """
        Constructor
        """
        self.name = None
        self.value = None
        self.lb = None
        self.ub = None
        self.stdErr = None
        self.tTest = None
        self.pValue = None
        self.robust_stdErr = None
        self.robust_tTest = None
        self.robust_pValue = None
        self.bootstrap_stdErr = None
        self.bootstrap_tTest = None
        self.bootstrap_pValue = None


class BiogemeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the module and class match the missing class, return the fake class
        if module == "biogeme.results" and name == "beta":
            return Beta
        if module == 'biogeme.results' and name == 'rawResults':
            return RawResults
        # Otherwise, proceed as normal
        return super().find_class(module, name)


def read_pickle_biogeme_3_2_14(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    with open(file=filename, mode='br') as file:
        pickled_results = pickle.load(file=file)

    model_name: str = pickled_results.modelName
    user_notes: str = pickled_results.userNotes
    beta_names: list[str] = [beta.name for beta in pickled_results.betas]
    beta_values: list[float] = [beta.value for beta in pickled_results.betas]
    lower_bounds: list[float] = [beta.lb for beta in pickled_results.betas]
    upper_bounds: list[float] = [beta.ub for beta in pickled_results.betas]
    gradient: list[float] = pickled_results.g.tolist()
    hessian: list[list[float]] = pickled_results.H.tolist()
    bhhh: list[list[float]] = pickled_results.bhhh.tolist()
    null_log_likelihood: float = pickled_results.nullLogLike
    initial_log_likelihood: float = pickled_results.initLogLike
    final_log_likelihood: float = pickled_results.logLike
    data_name: str = pickled_results.dataname
    sample_size: int = pickled_results.sampleSize
    number_of_observations: int = pickled_results.numberOfObservations
    monte_carlo: bool = pickled_results.monte_carlo
    number_of_draws: int = pickled_results.numberOfDraws
    types_of_draws: dict[str, str] = pickled_results.typesOfDraws
    number_of_excluded_data: int = pickled_results.excludedData
    draws_processing_time: timedelta = pickled_results.drawsProcessingTime
    optimization_messages: dict[str, Any] = pickled_results.optimizationMessages
    for key, value in optimization_messages.items():
        if isinstance(value, np.ndarray):  # Check if the value is a numpy array
            optimization_messages[key] = value.tolist()
    convergence: bool = pickled_results.convergence
    bootstrap: list[list[float]] = pickled_results.bootstrap.tolist()
    try:
        bootstrap_time: timedelta | None = pickled_results.bootstrap_time
    except AttributeError:
        bootstrap_time = None

    raw_estimation_results = RawEstimationResults(
        model_name=model_name,
        user_notes=user_notes,
        beta_names=beta_names,
        beta_values=beta_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gradient=gradient,
        hessian=hessian,
        bhhh=bhhh,
        null_log_likelihood=null_log_likelihood,
        initial_log_likelihood=initial_log_likelihood,
        final_log_likelihood=final_log_likelihood,
        data_name=data_name,
        sample_size=sample_size,
        number_of_observations=number_of_observations,
        monte_carlo=monte_carlo,
        number_of_draws=number_of_draws,
        types_of_draws=types_of_draws,
        number_of_excluded_data=number_of_excluded_data,
        draws_processing_time=draws_processing_time,
        optimization_messages=optimization_messages,
        convergence=convergence,
        bootstrap=bootstrap,
        bootstrap_time=bootstrap_time,
    )
    return raw_estimation_results


def read_pickle_biogeme_3_2_13(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    with open(file=filename, mode='br') as file:
        pickled_results = BiogemeUnpickler(file=file).load()

    model_name: str = pickled_results.modelName
    user_notes: str = pickled_results.userNotes
    beta_names: list[str] = [beta.name for beta in pickled_results.betas]
    beta_values: list[float] = [beta.value for beta in pickled_results.betas]
    lower_bounds: list[float] = [beta.lb for beta in pickled_results.betas]
    upper_bounds: list[float] = [beta.ub for beta in pickled_results.betas]
    gradient: list[float] = pickled_results.g.tolist()
    hessian: list[list[float]] = pickled_results.H.tolist()
    bhhh: list[list[float]] = pickled_results.bhhh.tolist()
    null_log_likelihood: float = pickled_results.nullLogLike
    initial_log_likelihood: float = pickled_results.initLogLike
    final_log_likelihood: float = pickled_results.logLike
    data_name: str = pickled_results.dataname
    sample_size: int = pickled_results.sampleSize
    number_of_observations: int = pickled_results.numberOfObservations
    monte_carlo: bool = pickled_results.monteCarlo
    number_of_draws: int = pickled_results.numberOfDraws
    types_of_draws: dict[str, str] = pickled_results.typesOfDraws
    number_of_excluded_data: int = pickled_results.excludedData
    draws_processing_time: timedelta = pickled_results.drawsProcessingTime
    optimization_messages: dict[str, Any] = pickled_results.optimizationMessages
    for key, value in optimization_messages.items():
        if isinstance(value, np.ndarray):  # Check if the value is a numpy array
            optimization_messages[key] = value.tolist()
    convergence: bool = pickled_results.convergence
    bootstrap: list[list[float]] = pickled_results.bootstrap.tolist()
    try:
        bootstrap_time: timedelta | None = pickled_results.bootstrap_time
    except AttributeError:
        bootstrap_time = None

    raw_estimation_results = RawEstimationResults(
        model_name=model_name,
        user_notes=user_notes,
        beta_names=beta_names,
        beta_values=beta_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gradient=gradient,
        hessian=hessian,
        bhhh=bhhh,
        null_log_likelihood=null_log_likelihood,
        initial_log_likelihood=initial_log_likelihood,
        final_log_likelihood=final_log_likelihood,
        data_name=data_name,
        sample_size=sample_size,
        number_of_observations=number_of_observations,
        monte_carlo=monte_carlo,
        number_of_draws=number_of_draws,
        types_of_draws=types_of_draws,
        number_of_excluded_data=number_of_excluded_data,
        draws_processing_time=draws_processing_time,
        optimization_messages=optimization_messages,
        convergence=convergence,
        bootstrap=bootstrap,
        bootstrap_time=bootstrap_time,
    )
    return raw_estimation_results


def read_pickle_biogeme_3_2_12(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    with open(file=filename, mode='br') as file:
        pickled_results = BiogemeUnpickler(file=file).load()

    model_name: str = pickled_results.modelName
    user_notes: str = pickled_results.userNotes
    beta_names: list[str] = [beta.name for beta in pickled_results.betas]
    beta_values: list[float] = [beta.value for beta in pickled_results.betas]
    lower_bounds: list[float] = [beta.lb for beta in pickled_results.betas]
    upper_bounds: list[float] = [beta.ub for beta in pickled_results.betas]
    gradient: list[float] = pickled_results.g.tolist()
    hessian: list[list[float]] = pickled_results.H.tolist()
    bhhh: list[list[float]] = pickled_results.bhhh.tolist()
    null_log_likelihood: float = pickled_results.nullLogLike
    initial_log_likelihood: float = pickled_results.initLogLike
    final_log_likelihood: float = pickled_results.logLike
    data_name: str = pickled_results.dataname
    sample_size: int = pickled_results.sampleSize
    number_of_observations: int = pickled_results.numberOfObservations
    monte_carlo: bool = pickled_results.monteCarlo
    number_of_draws: int = pickled_results.numberOfDraws
    types_of_draws: dict[str, str] = pickled_results.typesOfDraws
    number_of_excluded_data: int = pickled_results.excludedData
    draws_processing_time: timedelta = pickled_results.drawsProcessingTime
    optimization_messages: dict[str, Any] = pickled_results.optimizationMessages
    for key, value in optimization_messages.items():
        print(f'{key} {type(value)}')
        if isinstance(
            value, (np.ndarray, np.float64, np.float32)
        ):  # Check if the value is a numpy array
            optimization_messages[key] = value.tolist()
    convergence: bool = True
    bootstrap: list[list[float]] = (
        pickled_results.bootstrap.tolist()
        if pickled_results.bootstrap is not None
        else None
    )
    try:
        bootstrap_time: timedelta | None = pickled_results.bootstrap_time
    except AttributeError:
        bootstrap_time = None

    raw_estimation_results = RawEstimationResults(
        model_name=model_name,
        user_notes=user_notes,
        beta_names=beta_names,
        beta_values=beta_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gradient=gradient,
        hessian=hessian,
        bhhh=bhhh,
        null_log_likelihood=null_log_likelihood,
        initial_log_likelihood=initial_log_likelihood,
        final_log_likelihood=final_log_likelihood,
        data_name=data_name,
        sample_size=sample_size,
        number_of_observations=number_of_observations,
        monte_carlo=monte_carlo,
        number_of_draws=number_of_draws,
        types_of_draws=types_of_draws,
        number_of_excluded_data=number_of_excluded_data,
        draws_processing_time=draws_processing_time,
        optimization_messages=optimization_messages,
        convergence=convergence,
        bootstrap=bootstrap,
        bootstrap_time=bootstrap_time,
    )
    return raw_estimation_results


def read_pickle_biogeme_3_2_11(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    return read_pickle_biogeme_3_2_12(filename=filename)


def read_pickle_biogeme_3_2_10(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    return read_pickle_biogeme_3_2_12(filename=filename)


def read_pickle_biogeme_3_2_8(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    return read_pickle_biogeme_3_2_12(filename=filename)


def read_pickle_biogeme_3_2_7(filename: str) -> RawEstimationResults:
    """

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    return read_pickle_biogeme_3_2_12(filename=filename)


def read_pickle_biogeme(filename: str) -> RawEstimationResults:
    """Read an old pickle file, when the version of Biogeme used to create it is unknown

    :param filename: name of the pickle file
    :return: raw estimation results
    """
    try:
        results = read_pickle_biogeme_3_2_14(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_13(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_12(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_11(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_10(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_8(filename=filename)
        return results
    except AttributeError:
        ...

    try:
        results = read_pickle_biogeme_3_2_7(filename=filename)
        return results
    except AttributeError:
        ...

    error_msg = f'It was not possible to identify the format of the file {filename}'
    raise BiogemeError(error_msg)


def pickle_to_yaml(pickle_filename: str, yaml_filename: str) -> None:
    """
    Transforms a pickle file into a YAML format with the estimation results.

    :param pickle_filename: name of the input file
    :param yaml_filename: name of the output file
    """
    results = read_pickle_biogeme(filename=pickle_filename)
    serialize_to_yaml(data=results, filename=yaml_filename)
