"""Class that provides the function to the optimization allgorithm

:author: Michel Bierlaire
:date: Wed Nov 30 10:17:26 2022

"""
import biogeme.exceptions as excep
from biogeme.algorithms import functionToMinimize


class NegativeLikelihood(functionToMinimize):
    """Provides the value of the function to be minimized, as well as its
    derivatives. To be used by the opimization package.

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, like, like_deriv, scaled):
        """Constructor"""
        self.recalculate = True
        """True if the log likelihood must be recalculated
        """

        self.x = None  #: Vector of unknown parameters values

        self.batch = None
        """Value betwen 0 and 1 defining the size of the batch, that is the
        percentage of the data that should be used to approximate the
        log likelihood.
        """

        self.fv = None  #: value of the function

        self.gv = None  #: vector with the gradient

        self.hv = None  #: second derivatives matrix

        self.bhhhv = None  #: BHHH matrix

        self.like = like  #: function calculating the log likelihood

        self.like_deriv = like_deriv
        """function calculating the log likelihood and its derivatives.
        """

        self.scaled = scaled
        """if True, the value of the log likelihood is divided by the number
        of observations used to calculate it. In this case, the values
        with different sample sizes are comparable.
        """

    def setVariables(self, x):
        self.recalculate = True
        self.x = x
        self.fv = None
        self.gv = None
        self.hv = None
        self.bhhhv = None

    def f(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv = self.like(self.x, self.scaled, self.batch)
            self.gv = None
            self.hv = None
            self.bhhhv = None

        return -self.fv

    def f_g(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None or self.gv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv, self.gv, *_ = self.like_deriv(
                self.x, self.scaled, hessian=False, bhhh=False, batch=batch
            )
            self.hv = None
            self.bhhhv = None

        return -self.fv, -self.gv

    def f_g_h(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None or self.gv is None or self.hv is None:
            self.recalculate = True

        if self.recalculate:

            self.fv, self.gv, self.hv, _ = self.like_deriv(
                self.x, self.scaled, hessian=True, bhhh=False, batch=batch
            )
            self.bhhhv = None

        return -self.fv, -self.gv, -self.hv

    def f_g_bhhh(self, batch=None):
        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if self.fv is None or self.gv is None or self.bhhhv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv, self.gv, _, self.bhhhv = self.like_deriv(
                self.x, self.scaled, hessian=False, bhhh=True, batch=batch
            )
            self.hv = None

        return (-self.fv, -self.gv, -self.bhhhv)
