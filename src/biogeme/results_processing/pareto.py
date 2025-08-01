"""
Identification of Pareto optimal models

Michel Bierlaire
Fri Oct 4 09:22:22 2024
"""

from biogeme_optimization import pareto
from biogeme_optimization.pareto import Pareto
from .estimation_results import EstimationResults


def pareto_optimal(
    dict_of_results: dict[str, EstimationResults], a_pareto: Pareto | None = None
) -> dict[str, EstimationResults]:
    """Identifies the non dominated models, with respect to maximum
    log likelihood and minimum number of parameters

    :param dict_of_results: dict of results associated with their config ID
    :param a_pareto: if not None, Pareto set where the results will be inserted.
    :return: a dict of named results with pareto optimal results
    """
    if a_pareto is None:
        the_pareto = pareto.Pareto()
    else:
        the_pareto = a_pareto
    for config_id, estimation_results in dict_of_results.items():
        the_element = pareto.SetElement(
            element_id=config_id,
            objectives=[
                -estimation_results.final_log_likelihood,
                estimation_results.number_of_parameters,
            ],
        )
        the_pareto.add(the_element)

    selected_results = {
        element.element_id: dict_of_results[element.element_id]
        for element in the_pareto.pareto
    }
    the_pareto.dump()
    return selected_results
