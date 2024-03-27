"""Class that provides the function to the optimization algorithm

:author: Michel Bierlaire
:date: Wed Nov 30 10:17:26 2022

"""

from typing import Callable

import numpy as np
from biogeme_optimization.function import FunctionToMinimize, FunctionData
import biogeme.exceptions as excep
from biogeme.function_output import FunctionOutput


class NegativeLikelihood(FunctionToMinimize):
    """Provides the value of the function to be minimized, as well as its
    derivatives. To be used by the optimization package.

    """

    def __init__(
        self,
        dimension: int,
        like: Callable[[np.ndarray, bool, float | None], float],
        like_derivatives: Callable[
            [np.ndarray, bool, bool, bool, float | None], FunctionOutput
        ],
        parameters=None,
    ):
        """Constructor"""

        tolerance = None
        steptol = None
        if parameters is not None:
            if 'tolerance' in parameters:
                tolerance = parameters['tolerance']
            if 'steptol' in parameters:
                steptol = parameters['steptol']

        super().__init__(epsilon=tolerance, steptol=steptol)

        self.the_dimension: int = dimension  #: number of parameters to estimate

        self.like: Callable[[np.ndarray, bool, float | None], float] = (
            like  #: function calculating the log likelihood
        )

        self.like_derivatives: Callable[
            [np.ndarray, bool, bool, bool, float | None], FunctionOutput
        ] = like_derivatives
        """function calculating the log likelihood and its derivatives.
        """

    def dimension(self) -> int:
        """Provides the number of variables of the problem"""
        return self.the_dimension

    def _f(self) -> float:
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')
        return -self.like(self.x, scaled=False, batch=None)

    def _f_g(self) -> FunctionData:
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')

        the_function_output: FunctionOutput = self.like_derivatives(
            self.x, scaled=False, hessian=False, bhhh=False, batch=None
        )

        return FunctionData(
            function=-the_function_output.function,
            gradient=-the_function_output.gradient,
            hessian=None,
        )

    def _f_g_h(self) -> FunctionData:
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')

        the_function_output: FunctionOutput = self.like_derivatives(
            self.x, scaled=False, hessian=True, bhhh=False, batch=None
        )
        return FunctionData(
            function=-the_function_output.function,
            gradient=-the_function_output.gradient,
            hessian=-the_function_output.hessian,
        )
