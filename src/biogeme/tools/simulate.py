"""Shortcut to perform simulation of a choice model

Michel Bierlaire
Sat Jan 18 11:24:47 2025
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from biogeme.expressions import Expression
    from biogeme.database import Database
    from biogeme.results_processing import EstimationResults

logger = logging.getLogger(__name__)


def simulate(
    database: Database,
    dict_of_expressions: dict[str, Expression],
    estimation_results: EstimationResults,
    csv_filename: str | None = None,
) -> DataFrame:
    """Simulate a discrete choice model on a database.

    :param database: database
    :param dict_of_expressions: expressions that will be simulated
    :param estimation_results: estimation results.
    :param csv_filename: Name of the output CSV file. If None, no file is generated.
    :return: a pandas data frame with the simulated value. Each row corresponds to a row in the database, and each
              column to a formula.
    """
    from biogeme.biogeme import BIOGEME

    biosim = BIOGEME(database, dict_of_expressions)
    biosim.model_name = database.name + '_' + '_'.join(dict_of_expressions.keys())
    simulation_results = biosim.simulate(
        the_beta_values=estimation_results.get_beta_values()
    )
    if csv_filename is not None:
        simulation_results.to_csv(csv_filename, index=False)
        logger.info(f'File {csv_filename} has been generated.')
    return simulation_results
