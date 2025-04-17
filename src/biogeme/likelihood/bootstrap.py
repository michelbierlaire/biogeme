from tqdm import tqdm

from biogeme.model_elements import ModelElements
from biogeme.optimization import OptimizationAlgorithm
from .model_estimation import model_estimation, AlgorithmResults
from ..default_parameters import ParameterValue


def bootstrap(
    number_of_bootstrap_samples,
    the_algorithm: OptimizationAlgorithm,
    modeling_elements: ModelElements,
    parameters: dict[str, ParameterValue],
    starting_values: dict[str, float],
) -> list[AlgorithmResults]:
    """
    Perform bootstrap estimation to assess the variability of model parameters.

    This function generates a specified number of bootstrap samples from the
    original dataset, estimates the model on each sample using the provided
    algorithm and parameters, and returns the collection of estimation results.

    :param number_of_bootstrap_samples: Number of bootstrap replications to perform.
    :param the_algorithm: The optimization algorithm used to estimate the model.
    :param modeling_elements: The components defining the model, including the database and log-likelihood expression.
    :param parameters: Configuration parameters used during estimation.
    :param starting_values: Dictionary of initial values for the model's free parameters.

    :return: A list of tuples containing:
        - estimated parameter values (NumPy array),
        - diagnostic information from the optimizer (dictionary),
        - convergence status (boolean).
    """
    the_database = modeling_elements.database
    results = []
    for _ in tqdm(range(number_of_bootstrap_samples)):
        bootstrap_modeling_elements = modeling_elements
        bootstrap_modeling_elements.database = the_database.bootstrap_sample()
        one_result = model_estimation(
            the_algorithm=the_algorithm,
            modeling_elements=bootstrap_modeling_elements,
            parameters=parameters,
            some_starting_values=starting_values,
        )
        results.append(one_result)
    return results
