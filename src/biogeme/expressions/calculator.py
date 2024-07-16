""" Interface with the C++ implementation

:author: Michel Bierlaire
:date: Sat Sep  9 15:25:07 2023

"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import cythonbiogeme.cythonbiogeme as ee

from biogeme.exceptions import BiogemeError

from biogeme.function_output import (
    BiogemeFunctionOutput,
    BiogemeDisaggregateFunctionOutput,
    BiogemeFunctionOutputSmartOutputProxy,
    BiogemeDisaggregateFunctionOutputSmartOutputProxy,
)

if TYPE_CHECKING:
    from biogeme.database import Database
    from biogeme.expressions import Expression

logger = logging.getLogger(__name__)


def calculate_function_and_derivatives(
    the_expression: Expression,
    database: Database,
    calculate_gradient: bool,
    calculate_hessian: bool,
    calculate_bhhh: bool,
    aggregation: bool,
) -> (
    BiogemeFunctionOutputSmartOutputProxy
    | BiogemeDisaggregateFunctionOutputSmartOutputProxy
):
    """Interface with the C++ implementation calculating the expression

    :param the_expression: expression to calculate for each entry in the database.
    :param database: database. If no database is provided, the
        expression must not contain any variable.
    :param calculate_gradient: If True, the gradient is calculated.
    :param calculate_hessian: if True, the hessian is  calculated.
    :param calculate_bhhh: if True, the BHHH matrix is calculated.
    :param aggregation: if a database is provided, and this
        parameter is True, the expression is applied on each entry
        of the database, and all values are aggregated, so that
        the sum is returned. If False, the list of all values is returned.

    :raise BiogemeError: if a database is needed and not available.
    """
    the_cpp = ee.pyEvaluateOneExpression()
    if database is not None:
        the_cpp.setData(database.data)
    if the_expression.embed_expression('PanelLikelihoodTrajectory'):
        if database is None:
            raise BiogemeError('No database has been provided')
        if database.is_panel():
            database.build_panel_map()
            the_cpp.setDataMap(database.individualMap)
        else:
            error_msg = (
                'The expression involves '
                '"PanelLikelihoodTrajectory" '
                'that requires panel data'
            )
            raise BiogemeError(error_msg)

    if the_expression.requires_draws():
        if database is None:
            raise BiogemeError('No database has been provided')
        the_cpp.setDraws(database.theDraws)

    the_cpp.setExpression(the_expression.get_signature())
    the_cpp.setFreeBetas(the_expression.id_manager.free_betas_values)
    the_cpp.setFixedBetas(the_expression.id_manager.fixed_betas_values)
    the_cpp.setMissingData(the_expression.missingData)

    the_cpp.calculate(
        gradient=calculate_gradient,
        hessian=calculate_hessian,
        bhhh=calculate_bhhh,
        aggregation=aggregation,
    )
    f, g, h, b = the_cpp.getResults()
    gres = g if calculate_gradient else None
    hres = h if calculate_hessian else None
    bhhhres = b if calculate_bhhh else None
    if aggregation:
        result = BiogemeFunctionOutput(
            function=f[0],
            gradient=None if gres is None else g[0],
            hessian=None if hres is None else h[0],
            bhhh=None if bhhhres is None else b[0],
        )
        return BiogemeFunctionOutputSmartOutputProxy(result)
    disaggregate_result = BiogemeDisaggregateFunctionOutput(
        functions=f, gradients=gres, hessians=hres, bhhhs=bhhhres
    )
    if database is not None:
        return BiogemeDisaggregateFunctionOutputSmartOutputProxy(disaggregate_result)

    # If database is None, only one value is calculated
    result = disaggregate_result.unique_entry()
    if result is None:
        error_msg = (
            f'Incorrect number of entries: {len(disaggregate_result)}. Expected 1.'
        )
        raise BiogemeError(error_msg)
    return BiogemeFunctionOutputSmartOutputProxy(result)
