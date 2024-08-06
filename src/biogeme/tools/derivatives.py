import logging
from typing import Callable

import numpy as np

from biogeme.deprecated import deprecated
from biogeme.function_output import FunctionOutput, NamedFunctionOutput

logger = logging.getLogger(__name__)


def findiff_g(
    the_function: Callable[[np.ndarray], FunctionOutput], x: np.ndarray
) -> np.ndarray:
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
    tau = 0.0000001
    n = len(x)
    g = np.zeros(n)
    f = the_function(x).function
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
        fp = the_function(xp).function
        g[i] = (fp - f) / s
    return g


def findiff_h(
    the_function: (
        Callable[[np.ndarray], FunctionOutput]
        | Callable[[np.ndarray], NamedFunctionOutput]
    ),
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
    tau = 1.0e-7
    n = len(x)
    h = np.zeros((n, n))
    the_function_output: FunctionOutput | NamedFunctionOutput = the_function(x)
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
            x + s * ei
        )
        if isinstance(the_function_output, NamedFunctionOutput):
            the_function_output = the_function_output.function_output
        gp = the_function_output.gradient
        h[:, i] = (gp - g).flatten() / s
    return h


@deprecated(findiff_h)
def findiff_H(
    the_function: Callable[[np.ndarray], tuple[float, np.ndarray, ...]], x: np.ndarray
) -> np.ndarray:
    pass


def check_derivatives(
    the_function: (
        Callable[[np.ndarray], FunctionOutput]
        | Callable[[np.ndarray], NamedFunctionOutput]
    ),
    x: np.ndarray,
    names: list[str] | None = None,
    logg: bool | None = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    the_function_output: FunctionOutput | NamedFunctionOutput = the_function(x)
    if isinstance(the_function_output, NamedFunctionOutput):
        the_function_output = the_function_output.function_output
    g_num = findiff_g(the_function, x)
    gdiff = the_function_output.gradient - g_num
    if logg:
        if names is None:
            names = [f'x[{i}]' for i in range(len(x))]
        logger.info('x\t\tGradient\tFinDiff\t\tDifference')
        for k, v in enumerate(gdiff):
            logger.info(
                f'{names[k]:15}\t{the_function_output.gradient[k]:+E}\t{g_num[k]:+E}\t{v:+E}'
            )

    h_num = findiff_h(the_function, x)
    hdiff = the_function_output.hessian - h_num
    if logg:
        logger.info('Row\t\tCol\t\tHessian\tFinDiff\t\tDifference')
        for row in range(len(hdiff)):
            for col in range(len(hdiff)):
                logger.info(
                    f'{names[row]:15}\t{names[col]:15}\t{the_function_output.hessian[row, col]:+E}\t'
                    f'{h_num[row, col]:+E}\t{hdiff[row, col]:+E}'
                )
    return (
        the_function_output.function,
        the_function_output.gradient,
        the_function_output.hessian,
        gdiff,
        hdiff,
    )


@deprecated(check_derivatives)
def checkDerivatives(
    the_function: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
    x: np.ndarray,
    names: list[str] | None = None,
    logg: bool | None = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pass
