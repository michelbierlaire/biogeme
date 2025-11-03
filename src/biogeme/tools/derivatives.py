from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from tabulate import tabulate

from biogeme.floating_point import SQRT_EPS
from biogeme.function_output import FunctionOutput, NamedFunctionOutput

if TYPE_CHECKING:
    from biogeme.jax_calculator import CallableExpression

logger = logging.getLogger(__name__)


class CheckDerivativesResults(NamedTuple):
    function: float
    analytical_gradient: np.ndarray
    analytical_hessian: np.ndarray
    finite_differences_gradient: np.ndarray
    finite_differences_hessian: np.ndarray
    errors_gradient: np.ndarray
    errors_hessian: np.ndarray


def findiff_g(the_function: CallableExpression, x: np.ndarray) -> np.ndarray:
    """Calculates the gradient of a function :math:`f` using finite differences

    :param the_function: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`. The other elements are not
                        used.

    :param x: argument of the function

    :return: numpy vector, same dimension as x, containing the gradient
       calculated by finite differences.
    """
    x = x.astype(float)
    tau = SQRT_EPS
    n = len(x)
    g = np.zeros(n)
    f = the_function(x, gradient=False, hessian=False, bhhh=False).function
    for i in range(n):
        xi = x.item(i)
        xp = x.copy()
        if abs(xi) >= 1:
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        xp[i] = xi + s
        fp = the_function(xp, gradient=False, hessian=False, bhhh=False).function
        g[i] = (fp - f) / s
    return g


def findiff_h(
    the_function: CallableExpression,
    x: np.ndarray,
) -> np.ndarray:
    """Calculates the hessian of a function :math:`f` using finite differences


    :param the_function: A function object that takes a vector as an
                        argument, and returns a tuple. The first
                        element of the tuple is the value of the
                        function :math:`f`, and the second is the
                        gradient of the function.  The other elements
                        are not used.

    :param x: argument of the function
    :return: numpy matrix containing the hessian calculated by
             finite differences.
    """
    tau = SQRT_EPS
    n = len(x)
    h = np.zeros((n, n))
    the_function_output: FunctionOutput | NamedFunctionOutput = the_function(
        x, gradient=True, hessian=False, bhhh=False
    )
    if isinstance(the_function_output, NamedFunctionOutput):
        the_function_output = the_function_output.function_output
    g = the_function_output.gradient
    eye = np.eye(n, n)
    for i in range(n):
        xi = x.item(i)
        if abs(xi) >= 1:
            s = tau * xi
        elif xi >= 0:
            s = tau
        else:
            s = -tau
        ei = eye[i]
        the_function_output: FunctionOutput | NamedFunctionOutput = the_function(
            x + s * ei, gradient=True, hessian=False, bhhh=False
        )
        if isinstance(the_function_output, NamedFunctionOutput):
            the_function_output = the_function_output.function_output
        gp = the_function_output.gradient
        h[:, i] = (gp - g).flatten() / s
    return h


def check_derivatives(
    the_function: CallableExpression,
    x: np.ndarray,
    names: list[str] | None = None,
    logg: bool | None = False,
) -> CheckDerivativesResults:
    """Verifies the analytical derivatives of a function by comparing
    them with finite difference approximations.

    :param the_function: A function object that takes a vector as an argument,
        and returns a tuple:

        - The first element of the tuple is the value of the
          function :math:`f`,
        - the second is the gradient of the function,
        - the third is the hessian.

    :param x: arguments of the function

    :param names: the names of the entries of x (for reporting).

    :param logg: if True, messages will be displayed.

    :return: tuple f, g, h, gdiff, hdiff where

          - f is the value of the function at x,
          - g is the analytical gradient,
          - h is the analytical hessian,
          - gdiff is the difference between the analytical gradient
            and the finite difference approximation
          - hdiff is the difference between the analytical hessian
            and the finite difference approximation

    """
    x = np.array(x, dtype=float)
    the_function_output: FunctionOutput = the_function(
        x, gradient=True, hessian=True, bhhh=False
    )
    # if isinstance(the_function_output, NamedFunctionOutput):
    #    the_function_output = the_function_output.function_output
    g_num = findiff_g(the_function, x)
    gdiff = the_function_output.gradient - g_num
    if logg:
        headers = ['x', 'Gradient', 'FinDiff', 'Difference']

        if names is None:
            names = [f'x[{i}]' for i in range(len(x))]
        rows = [
            [
                f'{names[k]}',
                f'{the_function_output.gradient[k]:+E}',
                f'{g_num[k]:+E}',
                f'{v:+E}',
            ]
            for k, v in enumerate(gdiff)
        ]
        logger.info('Comparing first derivatives')
        logger.info(tabulate(rows, headers=headers, tablefmt='plain'))

    h_num = findiff_h(the_function, x)
    hdiff = the_function_output.hessian - h_num
    if logg:
        headers = ['Row', 'Col', 'Hessian', 'FinDiff', 'Difference']
        rows = [
            [
                names[row],
                names[col],
                f'{the_function_output.hessian[row, col]:+E}',
                f'{h_num[row, col]:+E}',
                f'{hdiff[row, col]:+E}',
            ]
            for col in range(len(hdiff))
            for row in range(len(hdiff))
        ]
        logger.info('Comparing second derivatives')
        logger.info(tabulate(rows, headers=headers, tablefmt='plain'))

    return CheckDerivativesResults(
        function=the_function_output.function,
        analytical_gradient=the_function_output.gradient,
        analytical_hessian=the_function_output.hessian,
        finite_differences_gradient=g_num,
        finite_differences_hessian=h_num,
        errors_gradient=gdiff,
        errors_hessian=hdiff,
    )
