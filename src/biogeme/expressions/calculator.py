""" Interface with the C++ implementation

:author: Michel Bierlaire
:date: Sat Sep  9 15:25:07 2023

"""
import logging
from typing import TYPE_CHECKING
import cythonbiogeme.cythonbiogeme as ee
from biogeme.exceptions import BiogemeError

if TYPE_CHECKING:
    from .base_expressions import Expression
    from biogeme.database import Database

logger = logging.getLogger(__name__)


def calculate_function_and_derivatives(
    the_expression: 'Expression',
    database: 'Database',
    calculate_gradient: bool,
    calculate_hessian: bool,
    calculate_bhhh: bool,
    aggregation: bool,
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
    if the_expression.embedExpression('PanelLikelihoodTrajectory'):
        if database is None:
            raise BiogemeError('No database has been provided')
        if database.isPanel():
            database.buildPanelMap()
            the_cpp.setDataMap(database.individualMap)
        else:
            error_msg = (
                'The expression involves '
                '"PanelLikelihoodTrajectory" '
                'that requires panel data'
            )
            raise BiogemeError(error_msg)

    if the_expression.requiresDraws():
        if database is None:
            raise BiogemeError('No database has been provided')
        the_cpp.setDraws(database.theDraws)

    the_cpp.setExpression(the_expression.getSignature())
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
        results = (
            f[0],
            None if gres is None else g[0],
            None if hres is None else h[0],
            None if bhhhres is None else b[0],
        )
    else:
        results = (f, gres, hres, bhhhres)

    return results
