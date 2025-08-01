"""Class that provides the function to the optimization algorithm

Michel Bierlaire
Sat Mar 29 16:27:47 2025
"""

import logging
from typing import Callable

import numpy as np
from biogeme_optimization.function import FunctionData, FunctionToMinimize

from biogeme.exceptions import BiogemeError
from biogeme.function_output import FunctionOutput

logger = logging.getLogger(__name__)


class NegativeLikelihood(FunctionToMinimize):
    """Provides the value of the function to be minimized, as well as its
    derivatives. To be used by the optimization package.

    """

    def __init__(
        self,
        dimension: int,
        loglikelihood: Callable[[np.ndarray, bool, bool, bool], FunctionOutput],
        parameters: dict[str, float] | None = None,
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

        self.loglikelihood: Callable[[np.ndarray, bool, bool, bool], FunctionOutput] = (
            loglikelihood
        )
        self.filename_for_best_iteration: str | None = None
        self.free_beta_names: list[str] | None = None
        self.best_value: float | None = None

    def save_iterations(
        self, filename_for_best_iteration: str, free_betas_names: list[str]
    ) -> None:
        self.filename_for_best_iteration = filename_for_best_iteration
        self.free_beta_names = free_betas_names

    def dimension(self) -> int:
        """Provides the number of variables of the problem"""
        return self.the_dimension

    def _f(self) -> float:
        if self.x is None:
            raise BiogemeError('The variables must be set first.')
        results = self.loglikelihood(self.x, gradient=False, hessian=False, bhhh=False)
        self._save_best_iteration(self.x, results.function)
        return -results.function

    def _f_g(self) -> FunctionData:
        if self.x is None:
            raise BiogemeError('The variables must be set first.')
        the_function_output: FunctionOutput = self.loglikelihood(
            self.x, gradient=True, hessian=False, bhhh=False
        )
        self._save_best_iteration(self.x, the_function_output.function)
        return FunctionData(
            function=-the_function_output.function,
            gradient=-the_function_output.gradient,
            hessian=None,
        )

    def _f_g_h(self) -> FunctionData:
        if self.x is None:
            raise BiogemeError('The variables must be set first.')

        the_function_output: FunctionOutput = self.loglikelihood(
            self.x, gradient=True, hessian=True, bhhh=False
        )
        self._save_best_iteration(self.x, the_function_output.function)
        return FunctionData(
            function=-the_function_output.function,
            gradient=-the_function_output.gradient,
            hessian=-the_function_output.hessian,
        )

    def _save_best_iteration(self, x: np.ndarray, f: float) -> None:
        if self.filename_for_best_iteration is None:
            return
        if self.best_value is None:
            self.best_value = f
        if f >= self.best_value:
            self.best_value = f
            with open(
                self.filename_for_best_iteration,
                "w",
                encoding="utf-8",
            ) as pf:
                for i, v in enumerate(x):
                    print(
                        f"{self.free_beta_names[i]} = {v}",
                        file=pf,
                    )
