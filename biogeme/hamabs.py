"""
Experimental algorithm by Gael Lederrey. Not yet documented.

:author: Michel Bierlaire and Gael Lederrey
:date: Thu Dec 26 17:41:18 2019
"""

# Too constraining
# pylint: disable=invalid-name, too-many-statements, too-many-branches
# pylint: disable=too-many-arguments, too-many-locals

import numpy as np
import biogeme.optimization as opt
import biogeme.messaging as msg
import biogeme.exceptions as excep

logger = msg.bioMessage()


class smoothing:
    def __init__(self, windowSize=10):
        self.windowSize = windowSize
        self.f = []
        self.g = []
        self.h = []
        self.batch = []
        self.n = None

    def numberOfValues(self):
        return len(self.f)

    def removeLast(self):
        self.f.pop()
        self.g.pop()
        self.h.pop()
        self.batch.pop()

    def add(self, f, g, h, batch, discount=0.95):
        if g is not None:
            if self.n is None:
                self.n = len(g)
            elif len(g) != self.n:
                raise excep.biogemeError(
                    f'Incompatible dimensions {len(g)} and {self.n}'
                )
        if h is not None:
            if h.shape != (self.n, self.n):
                raise excep.biogemeError(
                    f'Incompatible dimensions {h.shape} '
                    f'and ({self.n},{self.n})'
                )
        if batch <= 0.0 or batch > 1.0:
            raise excep.biogemeError(
                f'Batch size must be between 0 and 1: {batch}'
            )
        self.f += [f]
        self.g += [g]
        self.h += [h]
        self.batch += [batch]
        return self.f_g_h(discount)

    def f_g_h(self, discount=0.95):
        if len(self.f) == 0:
            return None, None, None
        if self.batch[-1] == 1.0:
            return self.f[-1], self.g[-1], self.h[-1]

        # The last value has a weight proportional to the batch that
        # was used.  Each other value has a weight proportional to one
        # minus the batch of the last value, times a discounted factor
        scale = [self.batch[-1]] + [
            discount**k * (1.0 - self.batch[-1]) / (self.windowSize - 1)
            for k in range(1, min(len(self.f), self.windowSize))
        ]
        normscale = [x / sum(scale) for x in scale]
        f = 0.0
        for k in range(0, min(len(self.f), self.windowSize)):
            f += normscale[k] * self.f[len(self.f) - k - 1]

        if self.n is None:
            # No derivative has yet been stored
            return f, None, None

        g = np.zeros(self.n)
        h = np.zeros((self.n, self.n))
        gscale = 0.0
        hscale = 0.0
        for k in range(0, min(len(self.f), self.windowSize)):
            i = len(self.f) - k - 1
            if self.g[i] is not None:
                g += scale[k] * self.g[i]
                gscale += scale[k]
            if self.h[i] is not None:
                h += scale[k] * self.h[i]
                hscale += scale[k]

        if hscale == 0.0:
            return f, g / gscale, None

        return f, g / gscale, h / hscale


def generateCandidateFirstOrder(
    fct, x, f, g, h, batch, delta, dogleg, maxiter, maxDelta, eta1, eta2
):
    for k in range(maxiter):
        if dogleg:
            step, _ = opt.dogleg(g, h, delta)
        else:
            step, _ = opt.truncatedConjugateGradient(g, h, delta)
        xc = x + step
        fct.setVariables(xc)
        fc, gc = fct.f_g(batch=batch)
        num = f - fc
        denom = -np.inner(step, g) - 0.5 * np.inner(step, h @ step)
        rho = num / denom
        if rho < eta1:
            # Failure: reduce the trust region
            delta = np.linalg.norm(step) / 2.0
            success = False
        else:
            success = True
            y = gc - g
            hc = opt.bfgs(h, step, y)
            if rho >= eta2:
                # Enlarge the trust region
                delta = min(2 * delta, maxDelta)
            return success, xc, fc, gc, hc, delta
    return success, None, None, None, None, delta


def generateCandidateSecondOrder(
    fct, x, f, g, h, batch, delta, dogleg, maxiter, maxDelta, eta1, eta2
):
    """To be documented..."""
    for k in range(maxiter):
        if dogleg:
            step, _ = opt.dogleg(g, h, delta)
        else:
            step, _ = opt.truncatedConjugateGradient(g, h, delta)
        xc = x + step
        fct.setVariables(xc)
        fc, gc, hc = fct.f_g_h(batch=batch)
        num = f - fc
        denom = -np.inner(step, g) - 0.5 * np.inner(step, h @ step)
        rho = num / denom
        if rho < eta1:
            # Failure: reduce the trust region
            delta = np.linalg.norm(step) / 2.0
            success = False
        else:
            success = True
            if rho >= eta2:
                # Enlarge the trust region
                delta = min(2 * delta, maxDelta)
            return success, xc, fc, gc, hc, delta
    return success, None, None, None, None, delta


def hamabs(fct, initBetas, fixedBetas, betaIds, bounds, parameters=None):
    """
    Algorithm inspired by `Lederrey et al. (2019)`

    .. _`Lederrey et al. (2019)`: https://transp-or.epfl.ch/documents/technicalReports/LedLurHilBie19.pdf

    :param fct: object to calculate the objective function and its derivatives.
    :type obj: optimization.functionToMinimize

    :param initBetas: initial value of the parameters.
    :type initBetas: numpy.array

    :param fixedBetas: betas that stay fixed suring the optimization.
    :type fixedBetas: numpy.array

    :param betaIds: internal identifiers of the non fixed betas.
    :type betaIds: numpy.array

    :param bounds: list of tuples (ell,u) containing the lower and upper
        bounds for each free parameter. Note that this algorithm does not
        support bound constraints. Therefore, all the bounds must be None.
    :type bounds: list(tuples)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

         - tolerance: when the relative gradient is below that threshold, the
           algorithm has reached convergence
           (default: :math:`\\varepsilon^{\\frac{1}{3}}`);
         - maxiter: the maximum number of iterations (default: 100).

    :type parameters: dict(string:float or int)

    :return: tuple x, messages, where

            - x is the solution found,
            - messages is a dictionary reporting various aspects related to
              the run of the algorithm.
    :rtype: numpy.array, dict(str:object)


    """

    for ell, u in bounds:
        if ell is not None or u is not None:
            raise excep.biogemeError(
                'This algorithm does not handle bound constraints. Remove the '
                'bounds, or select another algorithm.'
            )

    tol = np.finfo(np.float64).eps ** 0.3333
    maxiter = 1000
    # The size of the first batch is such that it can be increased 5 times
    firstBatch = 1.0 / 2.0**4
    # The critical of the batch when BFGS is applied allows for 2 increases
    hybrid = 1.0 / 2.0**2
    firstRadius = 1.0
    # Premature convergence for small batch sizes
    # scaleEps = 10.0
    # Maximum number of iterations before updating the batch size
    maxFailure = 2

    dogleg = False
    eta1 = 0.01
    eta2 = 0.9

    if parameters is not None:
        if 'tolerance' in parameters:
            tol = parameters['tolerance']
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']
        if 'firstBatch' in parameters:
            firstBatch = parameters['firstBatch']
        if 'firstRadius' in parameters:
            firstRadius = parameters['firstRadius']
        if 'hybrid' in parameters:
            hybrid = parameters['hybrid']
        if 'maxFailure' in parameters:
            maxFailure = parameters['maxFailure']
        if 'scaleEps' in parameters:
            scaleEps = parameters['scaleEps']
        if 'dogleg' in parameters:
            dogleg = parameters['dogleg']
        if 'eta1' in parameters:
            eta1 = parameters['eta1']
        if 'eta2' in parameters:
            eta2 = parameters['eta2']

    logger.detailed("** Optimization: HAMABS")

    avging = smoothing()

    k = 0
    xk = initBetas
    batch = firstBatch

    fct.setVariables(xk)
    f, g, h = fct.f_g_h(batch=batch)
    avgf, avgg, avgh = avging.add(f, g, h, batch)

    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)

    if batch == 1.0:
        relgrad = opt.relativeGradient(xk, f, g, typx, typf)
        if relgrad <= tol:
            message = f"Relative gradient = {relgrad} <= {tol}"
            return xk, 0, 1, message

    delta = firstRadius
    cont = True

    maxDelta = np.finfo(float).max
    minDelta = np.finfo(float).eps

    # Collect statistics per iteration
    #    columns = ['Batch','f','relgrad','Time','AbsDiff',
    #               'RelDiff', 'AbsEff', 'RelEff']
    #    stats = pd.DataFrame(columns=columns)

    while cont:
        logger.debug(f'***************** Iteration {k} **************')
        logger.debug(
            f'N={avging.numberOfValues()} xk={xk} avgf={avgf} delta={delta}'
        )
        k += 1
        if batch <= hybrid:
            success, xc, fc, gc, hc, delta = generateCandidateSecondOrder(
                fct,
                xk,
                avgf,
                avgg,
                avgh,
                batch,
                delta,
                dogleg,
                maxFailure,
                maxDelta,
                eta1,
                eta2,
            )
        else:
            success, xc, fc, gc, hc, delta = generateCandidateFirstOrder(
                fct,
                xk,
                avgf,
                avgg,
                avgh,
                batch,
                delta,
                dogleg,
                maxFailure,
                maxDelta,
                eta1,
                eta2,
            )

        if success:
            xk = xc
            avgf, avgg, avgh = avging.add(fc, gc, hc, batch)
            if batch == 1.0:
                relgrad = opt.relativeGradient(xk, avgf, avgg, typx, typf)
                if relgrad <= tol:
                    message = f"Relative gradient = {relgrad} <= {tol}"
                    cont = False
        else:
            if batch < 1.0:
                batch = min(2.0 * batch, 1.0)
                delta = firstRadius
                if batch <= hybrid:
                    fct.setVariables(xk)
                    f, g, h = fct.f_g_h(batch=batch)
                    avgf, avgg, avgh = avging.add(f, g, h, batch)
                else:
                    fct.setVariables(xk)
                    f, g = fct.f_g(batch=batch)
                    avgf, avgg, _ = avging.add(f, g, None, batch)

        if delta <= minDelta:
            if batch == 1.0:
                message = f"Trust region is too small: {delta}"
                cont = False

        if k == maxiter:
            message = f"Maximum number of iterations reached: {maxiter}"
            cont = False
        logger.detailed(
            f"{k} f={avgf:10.7g} delta={delta:6.2g} batch={100*batch:6.2g}%"
        )

    logger.detailed(message)
    messages = {
        'Algorithm': 'HAMABS prototype',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
    }

    return xk, messages
