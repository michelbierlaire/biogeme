"""Class that provides the function to the optimization algorithm

:author: Michel Bierlaire
:date: Wed Nov 30 10:17:26 2022

"""
from biogeme_optimization.function import FunctionToMinimize, FunctionData
import biogeme.exceptions as excep


class NegativeLikelihood(FunctionToMinimize):
    """Provides the value of the function to be minimized, as well as its
    derivatives. To be used by the opimization package.

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, dimension, like, like_deriv, parameters=None):
        """Constructor"""

        tolerance = None
        steptol = None
        if parameters is not None:
            if 'tolerance' in parameters:
                tolerance = parameters['tolerance']
            if 'steptol' in parameters:
                steptol = parameters['steptol']

        super().__init__(epsilon=tolerance, steptol=steptol)

        self.the_dimension = dimension  #: number of parameters to estimate

        self.like = like  #: function calculating the log likelihood

        self.like_deriv = like_deriv
        """function calculating the log likelihood and its derivatives.
        """

    def dimension(self):
        """Provides the number of variables of the problem"""
        return self.the_dimension

    def _f(self):
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')

        return -self.like(self.x, scaled=False, batch=None)

    def _f_g(self):
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')

        f, g, *_ = self.like_deriv(
            self.x, scaled=False, hessian=False, bhhh=False, batch=None
        )

        return FunctionData(
            function=-f,
            gradient=-g,
            hessian=None,
        )

    def _f_g_h(self):
        if self.x is None:
            raise excep.BiogemeError('The variables must be set first.')

        f, g, h, _ = self.like_deriv(
            self.x, scaled=False, hessian=True, bhhh=False, batch=None
        )
        return FunctionData(
            function=-f,
            gradient=-g,
            hessian=-h,
        )
