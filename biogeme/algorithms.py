"""
Optimization algorithms.

:author: Michel Bierlaire
:date: Sun Apr  5 16:48:54 2020

"""

# There seems to be a bug in PyLint.
# pylint: disable=invalid-unary-operand-type, no-member

# Too constraining
# pylint: disable=invalid-name
# pylint: disable=too-many-lines, too-many-locals
# pylint: disable=too-many-arguments, too-many-branches
# pylint: disable=too-many-statements, too-many-return-statements
# pylint: disable=bare-except


from abc import abstractmethod
import numpy as np
import scipy.linalg as la
import biogeme.exceptions as excep
import biogeme.messaging as msg


logger = msg.bioMessage()


class functionToMinimize:
    """This is an abstract class. The actual function to minimize
    must be implemented in a concrete class deriving from this one.

    """

    @abstractmethod
    def setVariables(self, x):

        """Set the values of the variables for which the function
        has to be calculated.

        :param x: values
        :type x: numpy.array
        """

    @abstractmethod
    def f(self, batch=None):
        """Calculate the value of the function

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      thre random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function
        :rtype: float
        """

    @abstractmethod
    def f_g(self, batch=None):
        """Calculate the value of the function and the gradient

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function and the gradient
        :rtype: tuple float, numpy.array
        """

    @abstractmethod
    def f_g_h(self, batch=None):
        """Calculate the value of the function, the gradient and the Hessian

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function, the gradient and the Hessian
        :rtype: tuple float, numpy.array, numpy.array
        """

    @abstractmethod
    def f_g_bhhh(self, batch=None):
        """Calculate the value of the function, the gradient and
        the BHHH matrix

        :param batch: for data driven functions (such as a log
                      likelikood function), it is possible to
                      approximate the value of the function using a
                      sample of the data called a batch. This argument
                      is a value between 0 and 1 representing the
                      percentage of the data that should be used for
                      the random batch. If None, the full data set is
                      used. Default: None pass
        :type batch: float

        :return: value of the function, the gradient and the BHHH
        :rtype: tuple float, numpy.array, numpy.array
        """


class bioBounds:
    """This class is designed for the management of simple bound constraints"""

    def __init__(self, b):
        """

        :param b: list of tuples (ell,u) containing the lower and
                  upper bounds for each free parameter.

        :type b: list(tuple)

        :raises biogeme.exceptions.biogemeError: if the bounds are incompatible

        """

        def noneToMinusInfinity(x):
            if x is None:
                return -np.inf

            return x

        def noneToPlusInfinity(x):
            if x is None:
                return np.inf

            return x

        self.bounds = b
        """list of tuples (ell,u) containing the lower and upper bounds for
        each free parameter.
        """

        self.n = len(b)  #: number of optimization variables

        self.lowerBounds = [noneToMinusInfinity(bb[0]) for bb in b]
        """ List of lower bounds """

        self.upperBounds = [noneToPlusInfinity(bb[1]) for bb in b]
        """ List of upper bounds """

        wrongBounds = np.array(
            [
                bb[0] > bb[1]
                for bb in b
                if bb[0] is not None and bb[1] is not None
            ]
        )
        if wrongBounds.any():
            errorMsg = (
                f'Incompatible bounds for indice(s) '
                f'{np.nonzero(wrongBounds)[0]}: '
                f'{[b[i] for i in np.nonzero(wrongBounds)[0]]}'
            )
            raise excep.biogemeError(errorMsg)

    def __str__(self):
        return self.bounds.__str__()

    def __repr__(self):
        return self.bounds.__repr__()

    def project(self, x):

        """Project a point onto the feasible domain defined by the bounds.

        :param x: point to project
        :type x: numpy.array


        :return: projected point
        :rtype: numpy.array

        :raises biogeme.exceptions.biogemeError: if the dimensions
                are inconsistent
        """

        if len(x) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {len(x)}' f' and {self.n}'
            )

        y = x
        for i in range(self.n):
            if self.lowerBounds[i] is not None and x[i] < self.lowerBounds[i]:
                y[i] = self.lowerBounds[i]
            if self.upperBounds[i] is not None and x[i] > self.upperBounds[i]:
                y[i] = self.upperBounds[i]
        return y

    def intersect(self, otherBounds):
        """
        Create a bounds object representing the intersection of two regions.

        :param otherBounds: other bound object that must be intersected.
        :type otherBounds: class bioBounds

        :return:  bound object, intersection of the two.
        :rtype: class bioBounds

        :raises biogeme.exceptions.biogemeError: if the dimensions
              are inconsistent
        """

        if otherBounds.n != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {otherBounds.n} ' f'and {self.n}'
            )

        newBounds = [
            (
                np.maximum(self.lowerBounds[i], otherBounds.lowerBounds[i]),
                np.minimum(self.upperBounds[i], otherBounds.upperBounds[i]),
            )
            for i in range(self.n)
        ]

        # Note that an exception with be raised at the creation of the
        # new 'bioBounds' object is the bounds are incompatible.

        result = bioBounds(newBounds)
        return result

    def intersectionWithTrustRegion(self, x, delta):
        """Create a bioBounds object representing the intersection
        between the feasible domain and the trust region.

        :param x: center of the trust region
        :type x: numpy.array

        :param delta: radius of the tust region (infinity norm)
        :type delta: float

        :return: intersection between the feasible region and the trus region
        :rtype: class bioBounds

        :raises biogeme.exceptions.biogemeError: if the dimensions
                are inconsistent
        """
        if len(x) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {len(x)}' f' and {self.n}'
            )

        trustRegion = bioBounds([(xk - delta, xk + delta) for xk in x])
        return self.intersect(trustRegion)

    def subspace(self, selectedVariables):
        """Generate a bioBounds object for selected variables

        :param selectedVariables: boolean vector. If an entry is True,
                   the corresponding variables is considered.
        :type selectedVariables: numpy.array(bool)

        :return: bound object
        :rtype: class bioBounds

        :raises biogeme.exceptions.biogemeError: if the dimensions
                are inconsistent
        """
        if len(selectedVariables) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: ' f'{len(selectedVariables)} and {self.n}'
            )

        return bioBounds(
            [i for i, j in zip(self.bounds, selectedVariables) if j]
        )

    def feasible(self, x):
        """Check if point verifies the bound constraints

        :param x: point to project
        :type x: numpy.array

        :return: True if x is feasible, False otherwise.
        :rtype: bool

        :raises biogeme.exceptions.biogemeError: if the dimensions
                 are inconsistent
        """
        if len(x) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: ' f'{len(x)} and {self.n}'
            )
        for i in range(self.n):
            if (
                self.lowerBounds[i] is not None
                and self.lowerBounds[i] - x[i] > np.finfo(float).eps
            ):
                return False
            if (
                self.upperBounds[i] is not None
                and x[i] - self.upperBounds[i] > np.finfo(float).eps
            ):
                return False
        return True

    def maximumStep(self, x, d):
        """Calculates the maximum step that can be performed
        along a direction while staying feasible.

        :param x: reference point
        :type x: numpy.array

        :param d: direction
        :type d: numpy.array

        :return: the largest alpha such that x + alpha * d is feasible
                 and the list of indices achieving this value.
        :rtype: float, int

        :raises biogeme.exceptions.biogemeError: if the point is infeasible

        """

        if not self.feasible(x):
            raise excep.biogemeError(f'Infeasible point: {x}')

        alpha = np.array(
            [
                (self.upperBounds[i] - x[i]) / d[i]
                if d[i] > np.finfo(float).eps
                else (self.lowerBounds[i] - x[i]) / d[i]
                if d[i] < -np.finfo(float).eps
                else np.inf
                for i in range(self.n)
            ]
        )
        m = np.amin(alpha)
        return m, np.where(alpha == m)[0]

    def activity(self, x, epsilon=np.finfo(float).eps):

        """Determines the activity status of each variable.

        :param x: point for which the activity must be determined.
        :type x: numpy.array

        :param epsilon: a bound is considered active if the distance
                        to it is less rhan epsilon.
        :type epsilon: float

        :return: a vector, same length as x, where each entry reports
            the activity of the corresponding variable:

            - 0 if no bound is active
            - -1 if the lower bound is active
            - 1 if the upper bound is active

        :rtype: numpy.array

        :raises biogeme.exceptions.biogemeError: if the vector x is
                    not feasible

        :raises biogeme.exceptions.biogemeError: if the dimensions
                    of x and bounds do not match.

        """
        if len(x) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {len(x)}' f' and {self.n}'
            )
        if not self.feasible(x):
            raise excep.biogemeError(
                f'{x} is not feasible for the ' f'bounds {self.bounds}'
            )

        activity = np.zeros_like(x, dtype=int)
        for i in range(self.n):
            if (
                self.lowerBounds[i] != -np.inf
                and x[i] - self.lowerBounds[i] <= epsilon
            ):
                activity[i] = -1
            elif (
                self.upperBounds[i] != np.inf
                and self.upperBounds[i] - x[i] <= epsilon
            ):
                activity[i] = 1
        return activity

    def breakpoints(self, x, d):
        """Projects the direction d, starting from x,
        on the intersection of the bound constraints

        :param x: current point
        :type x: numpy.array

        :param d: search direction
        :type d: numpy.array

        :return: list of tuple (index, value), where index is the index
                 of the variable, and value the value of the corresponding
                 breakpoint.
        :rtype: list(tuple(int,float))

        :raises biogeme.exceptions.biogemeError: if the dimensions
                are inconsistent

        :raises biogeme.exceptions.biogemeError: if x is infeasible

        """
        if len(d) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {self.n}' f' and {len(d)}'
            )
        bp = [
            (self.upperBounds[i] - x[i]) / d[i]
            if d[i] > np.finfo(float).eps
            else (self.lowerBounds[i] - x[i]) / d[i]
            if d[i] < -np.finfo(float).eps
            else 0
            for i in range(self.n)
        ]

        if any(b < 0 for b in bp):
            raise excep.biogemeError('Infeasible point')

        return sorted(enumerate(bp), key=lambda x: x[1])

    def generalizedCauchyPoint(self, xk, gk, H, direction):
        """Implementation of Step 2 of the Specific Algorithm by
        `Conn et al. (1988)`_.

        .. _`Conn et al. (1988)`: https://www.ams.org/journals/mcom/1988-50-182/S0025-5718-1988-0929544-3/S0025-5718-1988-0929544-3.pdf

        The quadratic model is defined as

        .. math:: m(x) = f(x_k) + (x - x_k)^T g_k +
                  \\frac{1}{2} (x-x_k)^T H (x-x_k).

        :param xk: current point
        :type xk: numpy.array. Dimension n.

        :param gk: vector g involved in the quadratic model definition.
        :type gk: numpy.array. Dimension n.

        :param H: matrix H involved in the quadratic model definition.
        :type H: numpy.array. Dimension n x n.

        :param direction: direction along which the GCP is searched.
        :type direction: numpy.array. Dimension n.

        :return: generalized Cauchy point based on inexact line search.
        :rtype: numpy.array. Dimension n.

        :raises biogeme.exceptions.biogemeError: if the dimensions are
               inconsistent
        :raises biogeme.exceptions.biogemeError: if xk is infeasible
        """

        if len(xk) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {len(xk)}' f' and {self.n}'
            )
        if len(gk) != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {len(gk)}' f' and {self.n}'
            )
        if H.shape[0] != self.n or H.shape[1] != self.n:
            raise excep.biogemeError(
                f'Incompatible size: {H.shape}' f' and {self.n}'
            )

        if not self.feasible(xk):
            raise excep.biogemeError('Infeasible iterate')

        x = xk
        g = gk - H @ xk
        d = direction

        n = len(xk)

        J = set()

        fprime = np.inner(gk, d)

        if fprime >= 0:
            if n <= 10:
                logger.warning(
                    f'GCP: direction {d} is not a descent '
                    f'direction at {x}.'
                )
            else:
                logger.warning('GCP: not a descent direction.')

        fsecond = np.inner(d, H @ d)

        while len(J) < self.n:
            delta_t, ind = self.maximumStep(x, d)
            # J is the set of active variables
            J.update(ind)

            # Test whether the GCP has been found
            ratio = -fprime / fsecond
            if fsecond > 0 and 0 < ratio < delta_t:
                x = x + ratio * d
                return x

            # Update line derivatives
            bd = np.zeros_like(d)
            bd[ind] = d[ind]
            b = H @ bd

            # In theory, x + delta_t * d must be feasible. However,
            # there may be some numerical problem. Therefore, we
            # project it on the feasible domain to make sure to obtain
            # a feasible point.
            x = self.project(x + delta_t * d)

            dg = np.sum([d[i] * g[i] for i in ind])
            fprime += delta_t * fsecond - np.inner(b, x) - dg
            fsecond += np.inner(b, bd - 2 * d)
            d[ind] = 0.0

            if fprime >= 0:
                return x

        return x


def schnabelEskow(
    A,
    tau=np.finfo(np.float64).eps ** 0.3333,
    taubar=np.finfo(np.float64).eps ** 0.6666,
    mu=0.1,
):

    """Modified Cholesky factorization by `Schnabel and Eskow (1999)`_.

    .. _`Schnabel and Eskow (1999)`: https://doi.org/10.1137/s105262349833266x

    If the matrix is 'safely' positive definite, the output is the
    classical Cholesky factor. If not, the diagonal elements are
    inflated in order to make it positive definite. The factor :math:`L`
    is such that :math:`A + E = PLL^TP^T`, where :math:`E` is a diagonal
    matrix contaninig the terms added to the diagonal, :math:`P` is a
    permutation matrix, and :math:`L` is w lower triangular matrix.

    :param A: matrix to factorize. Must be square and symmetric.
    :type A: numpy.array
    :param tau: tolerance factor.
                Default: :math:`\\varepsilon^{\\frac{1}{3}}`.
                See `Schnabel and Eskow (1999)`_
    :type tau: float
    :param taubar: tolerance factor.
                   Default: :math:`\\varepsilon^{\\frac{2}{3}}`.
                   See `Schnabel and Eskow (1999)`_
    :type taubar: float
    :param mu: tolerance factor.
               Default: 0.1.  See `Schnabel and Eskow (1999)`_
    :type mu: float

    :return: tuple :math:`L`, :math:`E`, :math:`P`,
                    where :math:`A + E = PLL^TP^T`.
    :rtype: numpy.array, numpy.array, numpy.array

    :raises biogeme.exceptions.biogemeError: if the matrix A is not square.
    :raises biogeme.exceptions.biogemeError: if the matrix A is not symmetric.
    """

    def pivot(j):
        A[j, j] = np.sqrt(A[j, j])
        for i in range(j + 1, dim):
            A[j, i] = A[i, j] = A[i, j] / A[j, j]
            A[i, j + 1 : i + 1] -= A[i, j] * A[j + 1 : i + 1, j]
            A[j + 1 : i + 1, i] = A[i, j + 1 : i + 1]

    def permute(i, j):
        A[[i, j]] = A[[j, i]]
        E[[i, j]] = E[[j, i]]
        A[:, [i, j]] = A[:, [j, i]]
        P[:, [i, j]] = P[:, [j, i]]

    A = A.astype(np.float64)
    dim = A.shape[0]
    if A.shape[1] != dim:
        raise excep.biogemeError('The matrix must be square')

    if not np.all(np.abs(A - A.T) < np.sqrt(np.finfo(np.float64).eps)):
        raise excep.biogemeError('The matrix must be symmetric')

    E = np.zeros(dim, dtype=np.float64)
    P = np.identity(dim)
    phaseOne = True
    gamma = abs(A.diagonal()).max()
    j = 0
    while j < dim and phaseOne is True:
        a_max = A.diagonal()[j:].max()
        a_min = A.diagonal()[j:].min()
        if a_max < taubar * gamma or a_min < -mu * a_max:
            phaseOne = False
            break

        # Pivot on maximum diagonal of remaining submatrix
        i = j + np.argmax(A.diagonal()[j:])
        if i != j:
            # Switch rows and columns of i and j of A
            permute(i, j)
        if j < dim - 1 and (
            (
                A.diagonal()[j + 1 :] - A[j + 1 :, j] ** 2 / A.diagonal()[j]
            ).min()
            < -mu * gamma
        ):
            phaseOne = False  # go to phase two
        else:
            # perform jth iteration of factorization
            pivot(j)
            j += 1

    # Phase two, A not positive-definite
    if not phaseOne:
        if j == dim - 1:
            E[-1] = delta = -A[-1, -1] + max(
                tau * (-A[-1, -1]) / (1 - tau), taubar * gamma
            )
            A[-1, -1] += delta
            A[-1, -1] = np.sqrt(A[-1, -1])
        else:
            deltaPrev = 0.0
            g = np.zeros(dim)
            k = j - 1  # k = number of iterations performed in phase one
            # Calculate lower Gerschgorin bounds of A[k+1]
            for i in range(k + 1, dim):
                g[i] = (
                    A[i, i]
                    - abs(A[i, k + 1 : i]).sum()
                    - abs(A[i + 1 : dim, i]).sum()
                )
            # Modified Cholesky Decomposition
            for j in range(k + 1, dim - 2):
                # Pivot on maximum lower Gerschgorin bound estimate
                i = j + np.argmax(g[j:])
                if i != j:
                    # Switch rows and columns of i and j of A
                    permute(i, j)
                # Calculate E[j, j] and add to diagonal
                norm_j = abs(A[j + 1 : dim, j]).sum()
                E[j] = delta = max(
                    0, -A[j, j] + max(norm_j, taubar * gamma), deltaPrev
                )
                if delta > 0:
                    A[j, j] += delta
                    deltaPrev = delta  # deltaPrev will contain E_inf
                # Update Gerschgorin bound estimates
                if A[j, j] != norm_j:
                    temp = 1.0 - norm_j / A[j, j]
                    g[j + 1 :] += abs(A[j + 1 :, j]) * temp
                # perform jth iteration of factorization
                pivot(j)

            # Final 2 by 2 submatrix
            e = np.linalg.eigvalsh(A[-2:, -2:])
            e.sort()
            E[-2] = E[-1] = delta = max(
                0,
                -e[0] + max(tau * (e[1] - e[0]) / (1 - tau), taubar * gamma),
                deltaPrev,
            )
            if delta > 0:
                A[-2, -2] += delta
                A[-1, -1] += delta
                deltaPrev = delta
            A[-2, -2] = np.sqrt(A[-2, -2])  # overwrites A[-2, -2]
            A[-1, -2] = A[-1, -2] / A[-2, -2]  # overwrites A[-1, -2]
            A[-2, -1] = A[-1, -2]
            # overwrites A[-1, -1]
            A[-1, -1] = np.sqrt(A[-1, -1] - A[-1, -2] * A[-1, -2])

    return np.tril(A), np.diag(P @ E), P


def lineSearch(fct, x, f, g, d, alpha0=1.0, beta1=1.0e-4, beta2=0.99, lbd=2.0):
    """
    Calculate a step along a direction that satisfies both Wolfe conditions


    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param x: current iterate.
    :type x: numpy.array

    :param f: value of the objective function at the current iterate.
    :type f: float

    :param g: value of the gradient at the current iterate.
    :type g: numpy.array

    :param d: descent direction.
    :type d: numpy.array

    :param alpha0: first step to test.
    :type alpha0: float

    :param beta1: parameter of the first Wolfe condition.
    :type beta1: float

    :param beta2: parameter of the second Wolfe condition.
    :type beta2: float

    :param lbd: expansion factor for a short step.
    :type lbd: float

    :return: a step verifing both Wolfe conditions
    :rtype: float

    :raises biogeme.exceptions.biogemeError: if ``lbd`` :math:`\\leq` 1
    :raises biogeme.exceptions.biogemeError: if ``alpha0`` :math:`\\leq` 0
    :raises biogeme.exceptions.biogemeError: if ``beta1`` :math:`\\geq` beta2
    :raises biogeme.exceptions.biogemeError: if ``d`` is not a descent
                                             direction

    """
    if lbd <= 1:
        raise excep.biogemeError(f'lambda is {lbd} and must be > 1')
    if alpha0 <= 0:
        raise excep.biogemeError(f'alpha0 is {alpha0} and must be > 0')
    if beta1 >= beta2:
        errorMsg = (
            f'Incompatible Wolfe cond. parameters: '
            f'beta1= {beta1} is greater than '
            f'beta2={beta2}'
        )
        raise excep.biogemeError(errorMsg)

    nfev = 1
    deriv = np.inner(g, d)

    if deriv >= 0:
        raise excep.biogemeError(f'd is not a descent direction: {deriv} >= 0')

    alpha = alpha0
    alphal = 0
    alphar = np.finfo(float).max
    finished = False
    while not finished:
        xnew = x + alpha * d
        fct.setVariables(xnew)
        fnew, gnew = fct.f_g()
        nfev += 1
        finished = True
        # First Wolfe condition violated?
        if fnew > f + alpha * beta1 * deriv:
            alphar = alpha
            alpha = (alphal + alphar) / 2.0
            finished = False
        elif np.inner(gnew, d) < beta2 * deriv:
            alphal = alpha
            if alphar == np.finfo(float).max:
                alpha = lbd * alpha
            else:
                alpha = (alphal + alphar) / 2.0
            finished = False
    return alpha, nfev


def relativeGradient(x, f, g, typx, typf):
    """Calculates the relative gradients.

    It is typically used as stopping criterion.

    :param x: current iterate.
    :type x: numpy.array
    :param f: value of f(x)
    :type f: float
    :param g: :math:`\\nabla f(x)`, gradient of f at x
    :type g: numpy.array
    :param typx: typical value for x.
    :type typx: numpy.array
    :param typf: typical value for f.
    :type typf: float

    :return: relative gradient

    .. math:: \\max_{i=1,\\ldots,n}\\frac{(\\nabla f(x))_i
              \\max(x_i,\\text{typx}_i)}
              {\\max(|f(x)|, \\text{typf})}

    :rtype: float
    """
    relgrad = np.array(
        [
            g[i] * max(abs(x[i]), typx[i]) / max(abs(f), typf)
            for i in range(len(x))
        ]
    )
    result = abs(relgrad).max()
    if np.isfinite(result):
        return result

    return np.finfo(float).max


def newtonLineSearch(
    fct, x0, eps=np.finfo(np.float64).eps ** 0.3333, maxiter=100
):
    """
    Newton method with inexact line search (Wolfe conditions)

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param x0: starting point
    :type x0: numpy.array

    :param eps: the algorithm stops when this precision is reached.
                 Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Defaut: 100
    :type maxiter: int

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

    :rtype: numpay.array, dict(str:object)

    """

    xk = x0
    fct.setVariables(xk)
    f, g, H = fct.f_g_h()
    nfev = 1
    ngev = 1
    nhev = 1
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relativeGradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.3g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Unconstrained Newton with line search',
            'Relative gradient': relgrad,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
            'Number of hessian evaluations': nhev,
            'Cause of termination': message,
        }
        return xk, messages

    k = 0
    cont = True
    while cont:
        L, _, P = schnabelEskow(H)
        y3 = -P.T @ g
        y2 = la.solve_triangular(L, y3, lower=True)
        y1 = la.solve_triangular(L.T, y2, lower=False)
        d = P @ y1
        alpha, nfls = lineSearch(fct, xk, f, g, d)
        nfev += nfls
        ngev += nfls
        xk = xk + alpha * d
        fct.setVariables(xk)
        f, g, H = fct.f_g_h()
        nfev += 1
        ngev += 1
        nhev += 1
        k += 1
        typf = max(np.abs(f), 1.0)
        relgrad = relativeGradient(xk, f, g, typx, typf)
        if relgrad <= eps:
            message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        logger.detailed(
            f'{k} f={f:10.7g} relgrad={relgrad:6.2g}' f' alpha={alpha:6.2g}'
        )

    messages = {
        'Algorithm': 'Unconstrained Newton with line search',
        'Relative gradient': relgrad,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
        'Number of hessian evaluations': nhev,
        'Cause of termination': message,
    }

    return xk, messages


def trustRegionIntersection(dc, d, delta):
    """Calculates the intersection with the boundary of the trust region.

    Consider a trust region of radius :math:`\\delta`, centered at
    :math:`\\hat{x}`. Let :math:`x_c` be in the trust region, and
    :math:`d_c = x_c - \\hat{x}`, so that :math:`\\|d_c\\| \\leq
    \\delta`. Let :math:`x_d` be out of the trust region, and
    :math:`d_d = x_d - \\hat{x}`, so that :math:`\\|d_d\\| \\geq
    \\delta`.  We calculate :math:`\\lambda` such that

    .. math:: \\| d_c + \\lambda (d_d - d_c)\\| = \\delta

    :param dc: xc - xhat.
    :type dc: numpy.array
    :param d: dd - dc.
    :type d: numpy.array
    :param delta: radius of the trust region.
    :type delta: float

    :return: :math:`\\lambda` such that :math:`\\| d_c +
              \\lambda (d_d - d_c)\\| = \\delta`

    :rtype: float

    """
    a = np.inner(d, d)
    b = 2 * np.inner(dc, d)
    c = np.inner(dc, dc) - delta**2
    discriminant = b * b - 4.0 * a * c
    return (-b + np.sqrt(discriminant)) / (2 * a)


def cauchyNewtonDogleg(g, H):
    """Calculate the Cauchy, the Newton and the dogleg points.

    The Cauchy point is defined as

    .. math:: d_c = - \\frac{\\nabla f(x)^T \\nabla f(x)}
                    {\\nabla f(x)^T \\nabla^2 f(x)
                    \\nabla f(x)} \\nabla f(x)

    The Newton point :math:`d_n` verifies Newton equation:

    .. math:: H_s d_n = - \\nabla f(x)

    where :math:`H_s` is a positive definite matrix generated with the
    method by `Schnabel and Eskow (1999)`_.

    The Dogleg point is

    .. math:: d_d = \\eta d_n

    where

    .. math:: \\eta = 0.2 + 0.8 \\frac{\\alpha^2}{\\beta |\\nabla f(x)^T d_n|}

    and :math:`\\alpha= \\nabla f(x)^T \\nabla f(x)`,
    :math:`\\beta=\\nabla f(x)^T \\nabla^2 f(x)\\nabla f(x).`

    :param g: gradient :math:`\\nabla f(x)`

    :type g: numpy.array

    :param H: hessian :math:`\\nabla^2 f(x)`

    :type H: numpy.array

    :return: tuple with Cauchy point, Newton point, Dogleg point
    :rtype: numpy.array, numpy.array, numpy.array

    :raises biogeme.exceptions.biogemeError: if the quadratic model is
        not convex.

    """
    alpha = np.inner(g, g)
    beta = np.inner(g, H @ g)
    dc = -(alpha / beta) * g
    L, E, P = schnabelEskow(H)
    negativeEigenvalue = E < 0
    if np.any(negativeEigenvalue):
        print(E)
        raise excep.biogemeError(
            'The dogleg method requires a ' 'convex optimization problem.'
        )

    y3 = -P.T @ g
    y2 = la.solve_triangular(L, y3, lower=True)
    y1 = la.solve_triangular(L.T, y2, lower=False)
    dn = P @ y1
    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(g, dn))))
    return dc, dn, eta * dn


def dogleg(g, H, delta):
    """
    Find an approximation of the trust region subproblem using
    the dogleg method

    :param g: gradient of the quadratic model.
    :type g: numpy.array
    :param H: hessian of the quadratic model.
    :type H: numpy.array
    :param delta: radius of the trust region.
    :type delta: float

    :return: d, diagnostic where

          - d is an approximate solution of the trust region subproblem
          - diagnostic is the nature of the solution:

             * -2 if negative curvature along Newton direction
             * -1 if negative curvature along Cauchy direction
               (i.e. along the gradient)
             * 1 if partial Cauchy step
             * 2 if Newton step
             * 3 if partial Newton step
             * 4 if Dogleg

    :rtype: numpy.array, int
    """

    dc, dn, dl = cauchyNewtonDogleg(g, H)

    # Check if the model is convex along the gradient direction

    alpha = np.inner(g, g)
    beta = np.inner(g, H @ g)
    if beta <= 0:
        dstar = -delta * g / np.sqrt(alpha)
        return dstar, -1

    # Compute the Cauchy point

    normdc = alpha * np.sqrt(alpha) / beta
    if normdc >= delta:
        # The Cauchy point is outside the trust
        # region. We move along the Cauchy
        # direction until the border of the trust
        # region.

        dstar = (delta / normdc) * dc
        return dstar, 1

    # Compute Newton point

    normdn = la.norm(dn)

    # Check the convexity of the model along Newton direction

    if np.inner(dn, H @ dn) <= 0.0:
        # Return the Cauchy point
        return dc, -2

    if normdn <= delta:
        # Newton point is inside the trust region
        return dn, 2

    # Compute the dogleg point

    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(g, dn))))

    partieldn = eta * la.norm(dn)

    if partieldn <= delta:
        # Dogleg point is inside the trust region
        dstar = (delta / normdn) * dn
        return dstar, 3

    # Between Cauchy and dogleg
    nu = dl - dc
    lbd = trustRegionIntersection(dc, nu, delta)
    dstar = dc + lbd * nu
    return dstar, 4


def truncatedConjugateGradient(g, H, delta):
    """Find an approximation of the trust region subproblem using the
    truncated conjugate gradient method

    :param g: gradient of the quadratic model.
    :type g: numpy.array
    :param H: hessian of the quadrartic model.
    :type H: numpy.array
    :param delta: radius of the trust region.
    :type delta: float

    :return: d, diagnostic, where

          - d is the approximate solution of the trust region subproblem,
          - diagnostic is the nature of the solution:

            * 1 for convergence,
            * 2 if out of the trust region,
            * 3 if negative curvature detected.
            * 4 if a numerical problem has been encountered

    :rtype: numpy.array, int

    """
    tol = 1.0e-6
    n = len(g)
    xk = np.zeros(n)
    gk = g
    dk = -gk
    for _ in range(n):
        try:
            curv = np.inner(dk, H @ dk)
            if curv <= 0:
                # Negative curvature has been detected
                diagnostic = 3
                a = np.inner(dk, dk)
                b = 2 * np.inner(xk, dk)
                c = np.inner(xk, xk) - delta * delta
                rho = b * b - 4 * a * c
                step = xk + ((-b + np.sqrt(rho)) / (2 * a)) * dk
                return step, diagnostic
            alphak = -np.inner(dk, gk) / curv
            xkp1 = xk + alphak * dk
            if np.isnan(xkp1).any() or la.norm(xkp1) > delta:
                # Out of the trust region
                diagnostic = 2
                a = np.inner(dk, dk)
                b = 2 * np.inner(xk, dk)
                c = np.inner(xk, xk) - delta * delta
                rho = b * b - 4 * a * c
                step = xk + ((-b + np.sqrt(rho)) / (2 * a)) * dk
                return step, diagnostic
            xk = xkp1
            gkp1 = H @ xk + g
            betak = np.inner(gkp1, gkp1) / np.inner(gk, gk)
            dk = -gkp1 + betak * dk
            gk = gkp1
            if la.norm(gkp1) <= tol:
                diagnostic = 1
                step = xk
                return step, diagnostic
        except ValueError:
            # Numerical problem
            diagnostic = 4
            a = np.inner(dk, dk)
            b = 2 * np.inner(xk, dk)
            c = np.inner(xk, xk) - delta * delta
            rho = b * b - 4 * a * c
            step = xk + ((-b + np.sqrt(rho)) / (2 * a)) * dk
            return step, diagnostic
    diagnostic = 1
    step = xk
    return step, diagnostic


def newtonTrustRegion(
    fct,
    x0,
    delta0=1.0,
    eps=np.finfo(np.float64).eps ** 0.3333,
    dl=False,
    maxiter=1000,
    eta1=0.01,
    eta2=0.9,
):
    """Newton method with trust region

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param x0: starting point
    :type x0: numpy.array

    :param delta0: initial radius of the trust region. Default: 100.
    :type delta0: float

    :param eps: the algorithm stops when this precision is reached.
              Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param dl: If True, the Dogleg method is used to solve the
       trut region subproblem. If False, the truncated conjugate
       gradient is used. Default: False.
    :type dl: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.

    :type maxiter: int

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: tuple x, messages, where

            - x is the solution found,
            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    """

    k = 0
    xk = x0
    fct.setVariables(xk)
    f, g, H = fct.f_g_h()
    nfev = 1
    ngev = 1
    nhev = 1
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relativeGradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Unconstrained Newton with trust region',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
            'Number of hessian evaluations': nhev,
        }
        return xk, messages
    delta = delta0
    nfev = 0
    cont = True
    maxDelta = np.finfo(float).max
    minDelta = np.finfo(float).eps
    rho = 0.0
    while cont:
        k += 1
        if dl:
            step, _ = dogleg(g, H, delta)
        else:
            step, _ = truncatedConjugateGradient(g, H, delta)
        xc = xk + step
        fct.setVariables(xc)
        # Calculate the value of the function
        fc = fct.f()
        nfev += 1
        num = f - fc
        denom = -np.inner(step, g) - 0.5 * np.inner(step, H @ step)
        rho = num / denom
        if rho < eta1:
            # Failure: reduce the trust region
            delta = la.norm(step) / 2.0
            status = '-'
        else:
            # Candidate accepted
            fc, gc, Hc = fct.f_g_h()
            nfev += 1
            ngev += 1
            nhev += 1
            xk = xc
            f = fc
            g = gc
            H = Hc
            if rho >= eta2:
                # Enlarge the trust region
                delta = min(2 * delta, maxDelta)
                status = '++'
            else:
                status = '+'
            relgrad = relativeGradient(xk, f, g, typx, typf)
            if relgrad <= eps:
                message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
                cont = False
        if delta <= minDelta:
            message = f'Trust region is too small: {delta}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        logger.detailed(
            f'{k} f={f:10.7g} relgrad={relgrad:6.2g} '
            f'delta={delta:6.2g} '
            f'rho={rho:6.2g} {status}'
        )

    messages = {
        'Algorithm': 'Unconstrained Newton with trust region',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
        'Number of hessian evaluations': nhev,
    }

    return xk, messages


def bfgs(H, d, y):
    """Update the BFGS matrix. Formula (13.12) of `Bierlaire (2015)`_
    where the method proposed by `Powell (1977)`_ is applied

    .. _`Bierlaire (2015)`: http://optimizationprinciplesalgorithms.com/
    .. _`Powell (1977)`: https://link.springer.com/content/pdf/10.1007/BFb0067703.pdf

    :param H: current approximation of the inverse of the Hessian
    :type H: numpy.array (2D)

    :param d: difference between two consecutive iterates.
    :type d: numpy.array (1D)

    :param y: difference between two consecutive gradients.
    :type y: numpy.array (1D)

    :return: updated approximation of the inverse of the Hessian.
    :rtype: numpy.array (2D)

    """
    Hd = H @ d
    dHd = np.inner(d, Hd)
    denom = np.inner(d, y)
    if denom >= 0.2 * dHd:
        eta = y
    else:
        theta = 0.8 * dHd / (dHd - denom)
        eta = theta * y + (1 - theta) * Hd

    return H - np.outer(Hd, Hd) / dHd + np.outer(eta, eta) / np.inner(d, eta)


def inverseBfgs(Hinv, d, y):
    """Update the inverse BFGS matrix. Formula (13.13) of `Bierlaire (2015)`_

    .. _`Bierlaire (2015)`: http://optimizationprinciplesalgorithms.com/

    :param Hinv: current approximation of the inverse of the Hessian
    :type Hinv: numpy.array (2D)

    :param d: difference between two consecutive iterates.
    :type d: numpy.array (1D)

    :param y: difference between two consecutive gradients.
    :type y: numpy.array (1D)

    :return: updated approximation of the inverse of the Hessian.
    :rtype: numpy.array (2D)
    """
    n = len(d)

    denom = np.inner(d, y)
    if denom <= 0.0:
        logger.warning(f"Unable to perform BFGS update as d'y = {denom} <= 0")
        return Hinv
    dy = np.outer(d, y)
    yd = np.outer(y, d)
    dd = np.outer(d, d)
    eye = np.identity(n)
    return ((eye - (dy / denom)) @ Hinv @ (eye - (yd / denom))) + dd / denom


def bfgsLineSearch(
    fct,
    x0,
    initBfgs=None,
    eps=np.finfo(np.float64).eps ** 0.3333,
    maxiter=1000,
):
    """BFGS method with inexact line search (Wolfe conditions)

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param x0: starting point
    :type x0: numpy.array

    :param initBfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type initBfgs: numpy.array

    :param eps: the algorithm stops when this precision is reached.
                 Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: tuple x, messages, where

            - x is the solution found,

            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    :raises biogeme.exceptions.biogemeError: if the dimensions of the
             matrix initBfgs do not match the length of x0.

    """

    n = len(x0)
    xk = x0
    fct.setVariables(xk)
    f, g = fct.f_g()
    nfev = 1
    ngev = 1
    if initBfgs is None:
        Hinv = np.identity(n)
    else:
        if initBfgs.shape != (n, n):
            errorMsg = (
                f'BFGS must be initialized with a {n}x{n} '
                f'matrix and not a {initBfgs.shape[0]}'
                'x{initBfgs.shape[1]} matrix.'
            )
            raise excep.biogemeError(errorMsg)
        Hinv = initBfgs
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relativeGradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'Inverse BFGS with line search',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
        }
        return xk, messages
    k = 0
    nfev = 0
    cont = True
    while cont:
        d = -Hinv @ g
        alpha, nfls = lineSearch(fct, xk, f, g, d)
        nfev += nfls
        delta = alpha * d
        xk = xk + delta
        gprev = g
        fct.setVariables(xk)
        f, g = fct.f_g()
        nfev += 1
        ngev += 1
        Hinv = inverseBfgs(Hinv, delta, g - gprev)
        nfev += 1
        k += 1
        relgrad = relativeGradient(xk, f, g, typx, typf)
        if relgrad <= eps:
            message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        logger.detailed(
            f'{k} f={f:10.7g} relgrad={relgrad:6.2g}' f' alpha={alpha:6.2g}'
        )
    messages = {
        'Algorithm': 'Inverse BFGS with line search',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
    }

    return xk, messages


def bfgsTrustRegion(
    fct,
    x0,
    initBfgs=None,
    delta0=1.0,
    eps=np.finfo(np.float64).eps ** 0.3333,
    dl=False,
    maxiter=1000,
    eta1=0.01,
    eta2=0.9,
):
    """BFGS method with trust region

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param x0: starting point
    :type x0: numpy.array

    :param initBfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type initBfgs: numpy.array

    :param delta0: initial radius of the trust region. Default: 100.
    :type delta0: float

    :param eps: the algorithm stops when this precision is reached.
                Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type eps: float

    :param dl: If True, the Dogleg method is used to solve the
       trut region subproblem. If False, the truncated conjugate
       gradient is used. Default: False.
    :type dl: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: tuple x, messages, where

            - x is the solution found,

            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    :raises biogeme.exceptions.biogemeError: if the dimensions of the
              matrix initBfgs do not match the length of x0.

    """
    k = 0
    xk = x0
    n = len(x0)
    fct.setVariables(xk)
    f, g = fct.f_g()
    nfev = 1
    ngev = 1
    if initBfgs is None:
        H = np.identity(n)
    else:
        if initBfgs.shape != (n, n):
            errorMsg = (
                f'BFGS must be initialized with a {n}x{n} '
                f'matrix and not a {initBfgs.shape[0]}'
                f'x{initBfgs.shape[1]} matrix.'
            )
            raise excep.biogemeError(errorMsg)
        H = initBfgs
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relgrad = relativeGradient(xk, f, g, typx, typf)
    if relgrad <= eps:
        message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
        messages = {
            'Algorithm': 'BFGS with trust region',
            'Relative gradient': relgrad,
            'Cause of termination': message,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
        }
        return xk, messages
    delta = delta0
    cont = True
    maxDelta = np.finfo(float).max
    minDelta = np.finfo(float).eps
    rho = 0.0
    while cont:
        k += 1
        if dl:
            step, _ = dogleg(g, H, delta)
        else:
            step, _ = truncatedConjugateGradient(g, H, delta)
        xc = xk + step
        fct.setVariables(xc)
        # Calculate the value of the function
        fc = fct.f()
        nfev += 1
        if fc >= f:
            delta = la.norm(step) / 2.0
            status = '-'
        else:
            num = f - fc
            denom = -np.inner(step, g) - 0.5 * np.inner(step, H @ step)
            rho = num / denom
            if rho < eta1:
                # Failure: reduce the trust region
                delta = la.norm(step) / 2.0
                status = '-'
            else:
                # Candidate accepted
                fc, gc = fct.f_g()
                nfev += 1
                ngev += 1
                d = xc - xk
                y = gc - g
                xk = xc
                f = fc
                g = gc
                H = bfgs(H, d, y)
                if rho >= eta2:
                    # Enlarge the trust region
                    delta = min(2 * delta, maxDelta)
                    status = '++'
                else:
                    status = '+'
                relgrad = relativeGradient(xk, f, g, typx, typf)
                if relgrad <= eps:
                    message = f'Relative gradient = {relgrad:.2g} <= {eps:.2g}'
                    cont = False
        if delta <= minDelta:
            message = f'Trust region is too small: {delta}'
            cont = False
        if k == maxiter:
            message = f'Maximum number of iterations reached: {maxiter}'
            cont = False
        logger.detailed(
            f'{k} f={f:10.7g} relgrad={relgrad:6.2g}'
            f' delta={delta:6.2g} '
            f'rho={rho:6.2g} {status}'
        )

    messages = {
        'Algorithm': 'BFGS with trust region',
        'Relative gradient': relgrad,
        'Cause of termination': message,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
    }

    return xk, messages


def truncatedConjugateGradientSubspace(
    xk,
    gk,
    Hk,
    delta,
    bounds,
    infeasibleIterate=False,
    tol=np.finfo(np.float64).eps ** 0.3333,
):
    """Find an approximation of the solution of the trust region
    subproblem using the truncated conjugate gradient method within
    the subspace of free variables. Free variables are those
    corresponding to inactive constraints at the generalized Cauchy
    point.

    :param xk: current iterate.
    :type xk: numpy.array

    :param gk: gradient of the quadratic model.
    :type gk: numpy.array

    :param Hk: hessian of the quadrartic model.
    :type Hk: numpy.array

    :param delta: radius of the trust region.
    :type delta: float

    :param bounds: bounds on the variables.
    :type bounds: class bioBounds

    :param infeasibleIterate: if True, the algorithm may continue
                              until termination.  The result will then
                              be projected on the feasible domain.  If
                              False, the algorithm stops as soon as an
                              infeasible iterate is generated.
                              Default: False.

    :type infeasibleIterate: bool

    :param tol: tolerance used for stopping criterion.
    :type tol: float

    :return: d, diagnostic, where

          - d is the approximate solution of the trust region subproblem,
          - diagnostic is the nature of the solution:

            * 1 for convergence,
            * 2 if out of the trust region,
            * 3 if negative curvature detected.
            * 4 if a numerical problem has been encountered

    :rtype: numpy.array, int

    :raises biogeme.exceptions.biogemeError: if the dimensions are inconsistent

    """

    if np.isnan(xk).any():
        raise excep.biogemeError(f'Invalid xk: {xk}')

    if np.isnan(gk).any():
        raise excep.biogemeError(f'Invalid gk: {gk}')

    if np.isnan(Hk).any():
        raise excep.biogemeError(f'Invalid Hk: {Hk}')

    # First, we calculate the intersection between the trust region on
    # the bounds. The trust region is also a bound constraint (based
    # on infinity norm) of radius 'delta', centered at xk.

    intersection = bounds.intersectionWithTrustRegion(xk, delta)

    # Then, we calculate the generalized Cauchy point
    gcp = intersection.generalizedCauchyPoint(xk, gk, Hk, -gk)

    x = gcp
    r = -gk - Hk @ (x - xk)

    etak = tol

    activityStatus = intersection.activity(gcp)
    freeVariables = activityStatus == 0

    if not freeVariables.any():
        return gcp, 1

    xbar = x[freeVariables]
    rbar = r[freeVariables]

    # Extract the  bounds for the free variables
    boundsBar = intersection.subspace(freeVariables)
    Bbar = Hk[freeVariables][:, freeVariables]
    pbar = np.zeros_like(rbar)
    rho1 = 1
    rho2 = np.inner(rbar, rbar)

    while rho2 >= etak * etak:
        try:
            beta = rho2 / rho1
            pbar = rbar + beta * pbar
            ybar = Bbar @ pbar

            if boundsBar.feasible(xbar):
                feasible = True
                alpha1, _ = boundsBar.maximumStep(xbar, pbar)
            else:
                feasible = False

            if np.inner(pbar, ybar) <= 0:
                # Negative curvature has been detected.
                if feasible:
                    x[freeVariables] = xbar + alpha1 * pbar
                else:
                    x[freeVariables] = xbar
                return bounds.project(x), 3

            alpha2 = rho2 / np.inner(pbar, ybar)
            if not infeasibleIterate and alpha2 > alpha1:
                # Infeasible iterate
                x[freeVariables] = xbar + alpha1 * pbar
                return x, 2

            xbar = xbar + alpha2 * pbar
            rbar = rbar - alpha2 * ybar
            rho1 = rho2
            rho2 = np.inner(rbar, rbar)

        except ValueError:
            # Numerical problem detected. Return the current value of x
            logger.warning('Numerical problem in conjugate gradient algorithm')
            x[freeVariables] = xbar
            if infeasibleIterate:
                return bounds.project(x), 4
            return x, 4

        x[freeVariables] = xbar
    if infeasibleIterate:
        return bounds.project(x), 4
    return x, 1


def simpleBoundsNewtonAlgorithm(
    fct,
    bounds,
    x0,
    proportionTrueHessian=1.0,
    infeasibleConjugateGradient=False,
    delta0=1.0,
    tol=np.finfo(np.float64).eps ** 0.3333,
    steptol=1.0e-5,
    cgtol=np.finfo(np.float64).eps ** 0.3333,
    maxiter=1000,
    eta1=0.01,
    eta2=0.9,
    enlargingFactor=10,
):
    """Trust region algorithm for problems with simple bounds

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: optimization.functionToMinimize

    :param bounds: bounds on the variables
    :type bounds: class bounds

    :param x0: starting point
    :type x0: numpy.array

    :param proportionTrueHessian: proportion of the iterations where
                                  the true hessian is calculated. When
                                  not, the BFGS update is used. If
                                  1.0, it is used for all
                                  iterations. If 0.0, it is not used
                                  at all.
    :type proportionTrueHessian: float

    :param infeasibleConjugateGradient: if True, the conjugate
                              gradient algorithm may generate until
                              termination.  The result will then be
                              projected on the feasible domain.  If
                              False, the algorithm stops as soon as an
                              infeasible iterate is generated.
                              Default: False.
    :type infeasibleConjugateGradient: bool

    :param delta0: initial radius of the trust region. Default: 100.
    :type delta0: float

    :param tol: the algorithm stops when this precision is reached.
                Default: :math:`\\varepsilon^{\\frac{1}{3}}`
    :type tol: float

    :param steptol: the algorithm stops when the relative change in x
                is below this threshold. Basically, if p significant
                digits of x are needed, steptol should be set to
                1.0e-p.  Default: :math:`10^{-5}`

    :type steptol: float


    :param cgtol: the conjugate gradient algorithm stops when this
                  precision is reached.  Default:
                  :math:`\\varepsilon^{\\frac{1}{3}}`

    :type cgtol: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.

    :type maxiter: int

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :param enlargingFactor: if an iteration is very successful, the
                            radius of the trust region is multiplied
                            by this factor. Default 10.
    :type enlargingFactor: float

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

    :rtype: numpay.array, dict(str:object)

    :raises biogeme.exceptions.biogemeError: if the dimensions of the
             matrix initBfgs do not match the length of x0.

    """

    if len(x0) != bounds.n:
        raise excep.biogemeError(
            f'Incompatible size:' f' {len(x0)} and {len(bounds)}'
        )

    if not bounds.feasible(x0):
        logger.warning(
            'Initial point not feasible. '
            'It will be projected onto the feasible domain.'
        )

    numberOfTrueHessian = 0
    numberOfMatrices = 0

    if proportionTrueHessian == 1.0:
        algo = 'Newton with trust region for simple bound constraints'
    elif proportionTrueHessian == 0.0:
        algo = 'BFGS with trust region for simple bound constraints'
    else:
        algo = (
            f'Hybrid Newton [{100*proportionTrueHessian}%] with trust '
            f'region for simple bound constraints'
        )

    k = 0
    xk = bounds.project(x0)
    fct.setVariables(xk)
    nfev = 0
    ngev = 0
    nhev = 0
    if proportionTrueHessian > 0:
        f, g, H = fct.f_g_h()
        nfev += 1
        ngev += 1
        nhev += 1

        numberOfTrueHessian += 1
        numberOfMatrices += 1
        # If there is a numerical problem with the Hessian, we use BFGS instead
        if np.isnan(H).any() or np.linalg.norm(H) > 1.0e100:
            logger.warning(
                f'Numerical problem with the second derivative'
                f' matrix at the starting point. Norm = '
                f'{np.linalg.norm(H)}. Replaced by the '
                f'identity matrix.'
            )
            H = np.eye(len(xk))
    else:
        f, g = fct.f_g()
        nfev += 1
        ngev += 1
        H = np.eye(len(xk))
        numberOfMatrices += 1

    projectedGradient = bounds.project(xk - g) - xk
    typx = np.ones(np.asarray(xk).shape)
    typf = max(np.abs(f), 1.0)
    relchange = None
    relgrad = relativeGradient(xk, f, projectedGradient, typx, typf)
    if relgrad <= tol:
        message = f'Relative gradient = {relgrad:.2g} <= {tol:.2g}'
        messages = {
            'Algorithm': algo,
            'Relative projected gradient': relgrad,
            'Number of iterations': 0,
            'Number of function evaluations': nfev,
            'Number of gradient evaluations': ngev,
            'Number of hessian evaluations': nhev,
            'Cause of termination': message,
        }
        return xk, messages

    delta = delta0
    cont = True
    maxDelta = np.finfo(float).max
    minDelta = np.finfo(float).eps
    rho = 0.0
    while cont:
        k += 1

        # Solve the quandratic problem in the subspace defined by the GCP

        xc, _ = truncatedConjugateGradientSubspace(
            xk, g, H, delta, bounds, infeasibleConjugateGradient, cgtol
        )

        if np.isnan(xc).any():
            delta = delta / 2.0
            status = '-'
        else:
            failed = False
            try:
                fct.setVariables(xc)
                fc = fct.f()
                nfev += 1

                num = f - fc
                step = xc - xk
                denom = -np.inner(step, g) - 0.5 * np.inner(step, H @ step)
                rho = num / denom
                failed = rho < eta1
            except RuntimeError as e:
                raise e
                logger.debug(xc)
                failed = True

            if failed:
                # Failure: reduce the trust region
                delta = min(delta / 2.0, la.norm(step, np.inf) / 2.0)
                status = '-'
            else:
                # Candidate accepted
                if (
                    proportionTrueHessian > 0
                    and float(numberOfTrueHessian) / float(numberOfMatrices)
                    <= proportionTrueHessian
                ):

                    try:
                        fc, gc, Hc = fct.f_g_h()
                        nfev += 1
                        ngev += 1
                        nhev += 1
                        numberOfTrueHessian += 1
                        numberOfMatrices += 1
                        # If there is a numerical problem with the
                        # Hessian, we apply BFGS
                        if np.linalg.norm(Hc) > 1.0e100:
                            logger.warning(
                                f'Numerical problem with the second '
                                f'derivative matrix. '
                                f'Norm = {np.linalg.norm(Hc)}. '
                                f'Replaced by the identity matrix.'
                            )
                            y = gc - g
                            Hc = bfgs(H, step, y)
                        fghCalculated = True
                    except excep.biogemeError:
                        # Failure: reduce the trust region
                        delta = min(delta / 2.0, la.norm(step, np.inf) / 2.0)
                        status = '-'
                        fghCalculated = False
                else:
                    try:
                        fc, gc = fct.f_g()
                        nfev += 1
                        ngev += 1
                        y = gc - g
                        Hc = bfgs(H, step, y)
                        numberOfMatrices += 1
                        fghCalculated = True
                    except excep.biogemeError:
                        # Failure: reduce the trust region
                        delta = min(delta / 2.0, la.norm(step, np.inf) / 2.0)
                        status = '-'
                        fghCalculated = False
                if fghCalculated:
                    nfev += 1
                    xpred = xk
                    xk = xc
                    f = fc
                    g = gc
                    H = Hc
                    if rho >= eta2:
                        # Enlarge the trust region
                        delta = min(enlargingFactor * delta, maxDelta)
                        status = '++'
                    else:
                        status = '+'

                    projectedGradient = bounds.project(xk - g) - xk
                    relgrad = relativeGradient(
                        xk, f, projectedGradient, typx, typf
                    )
                    if relgrad <= tol:
                        message = (
                            f'Relative gradient = {relgrad:.2g} <= {tol:.2g}'
                        )
                        cont = False
                    relchange = relativeChange(xk, xpred, typx)
                    if relchange <= steptol:
                        message = (
                            f'Relative change = '
                            f'{relchange:.3g} <= {steptol:.2g}'
                        )
                        cont = False
            if delta <= minDelta:
                message = f'Trust region is too small: {delta}'
                cont = False
            if k == maxiter:
                message = f'Maximum number of iterations reached: {maxiter}'
                cont = False
            if relchange is None:
                logger.detailed(
                    f'{k} f={f:10.7g} projected '
                    f'rel. grad.={relgrad:6.2g} '
                    f'delta={delta:6.2g} rho={rho:6.2g} {status}'
                )
            else:
                logger.detailed(
                    f'{k} f={f:10.7g} projected '
                    f'rel. grad.={relgrad:6.2g} '
                    f'rel. change={relchange:6.2g} '
                    f'delta={delta:6.2g} rho={rho:6.2g} {status}'
                )
    if numberOfMatrices != 0:
        actualProp = 100 * float(numberOfTrueHessian) / float(numberOfMatrices)
    else:
        actualProp = 0
    m = f'Proportion of Hessian calculation: ' f'{actualProp}%'
    logger.detailed(m)

    messages = {
        'Algorithm': algo,
        'Proportion analytical hessian': f'{actualProp:.1f}%',
        'Relative projected gradient': relgrad,
        'Relative change': relchange,
        'Number of iterations': k,
        'Number of function evaluations': nfev,
        'Number of gradient evaluations': ngev,
        'Number of hessian evaluations': nhev,
        'Cause of termination': message,
    }

    return xk, messages


def relativeChange(x, xpred, typx):
    """Calculates the relative change.

    It is typically used as stopping criterion.

    :param x: current iterate.
    :type x: numpy.array

    :param xpred: previous iterate.
    :type xpred: numpy.array

    :param typx: typical value for x.
    :type typx: numpy.array

    :return: relative change:

    .. math:: \\max_i \\frac{|(x_k)_i - (x_{k-1})_i|}{\\max(|(x_k)_i|,
              \\text{typx}_i)}

    :rtype: float

    """

    relx = np.array(
        [abs(x[i] - xpred[i]) / max(abs(x[i]), typx[i]) for i in range(len(x))]
    )
    result = relx.max()
    if np.isfinite(result):
        return result

    return np.finfo(float).max
