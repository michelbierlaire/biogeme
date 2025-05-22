"""
Optimization algorithms for Biogeme

:author: Michel Bierlaire
:date: Mon Dec 21 10:24:24 2020

"""

# There seems to be a bug in PyLint.
# pylint: disable=invalid-unary-operand-type, no-member

# Too constraining
# pylint: disable=invalid-name
# pylint: disable=too-many-lines, too-many-locals
# pylint: disable=too-many-arguments, too-many-branches
# pylint: disable=too-many-statements, too-many-return-statements
# pylint: disable=bare-except


import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.optimize as sc
from biogeme_optimization.bounds import Bounds
from biogeme_optimization.diagnostics import OptimizationResults
from biogeme_optimization.function import FunctionToMinimize
from biogeme_optimization.linesearch import bfgs_line_search, newton_line_search
from biogeme_optimization.simple_bounds import simple_bounds_newton_algorithm
from biogeme_optimization.trust_region import bfgs_trust_region, newton_trust_region

logger = logging.getLogger(__name__)

OptimizationAlgorithm = Callable[
    [
        FunctionToMinimize,
        np.ndarray,
        list[tuple[float, float]],
        list[str],
        dict[str, Any] | None,
    ],
    OptimizationResults,
]


def scipy(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: Any | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on the scipy
    minimize function.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the Beta parameters
    :type init_betas: numpy.array

    :param bounds: list of tuples (ell,u) containing the lower and upper bounds
          for each free parameter
    :type bounds: list(tuple)

    :param variable_names: names of the variables. Ignored
        here. Included to comply with the syntax.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
         optimization routine. See the `scipy`_ documentation.

    .. _`scipy`: https://docs.scipy.org/doc/scipy/reference/optimize.html

    :type parameters: dict(string:float or string)

    :return: named tuple:

            - betas is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResult

    """

    def f_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        fct.set_variables(x)
        function_data = fct.f_g()
        return function_data.function, function_data.gradient

    logger.info('Optimization algorithm: scipy')
    # Absolute tolerance
    absgtol = 1.0e-7
    opts = {'ftol': np.finfo(np.float64).eps, 'gtol': absgtol}
    allowed_scipy_opts = {'gtol', 'ftol', 'maxiter', 'disp', 'eps'}
    if parameters is not None:
        opts.update({k: v for k, v in parameters.items() if k in allowed_scipy_opts})

    if 'gtol' in opts.keys():
        logger.info(f'Minimize with tol {opts["gtol"]}')

    results = sc.minimize(f_and_grad, init_betas, bounds=bounds, jac=True, options=opts)

    messages = {
        'Algorithm': 'scipy.optimize',
        'Cause of termination': results.message,
        'Number of iterations': results.nit,
        'Number of function evaluations': results.nfev,
    }

    return OptimizationResults(
        solution=results.x, messages=messages, convergence=results.success
    )


def newton_linesearch_for_biogeme(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on Newton method.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.
    :type init_betas: numpy.array

    :param bounds: list of tuples (ell,u) containing the lower and
                   upper bounds for each free parameter. Note that
                   this algorithm does not support bound constraints.
                   Therefore, all the bounds will be ignored.

    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that
          threshold, the algorithm has reached convergence
          (default: :math:`\\varepsilon^{\\frac{1}{3}}`);
        - maxiter: the maximum number of iterations (default: 100).

    :type parameters: dict(string:float or int)

    :return: named tuple:

            - betas is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    logger.info('Optimization algorithm: Newton with line search [LS-newton]')

    for ell, u in bounds:
        if ell is not None or u is not None:
            warning_msg = (
                'This algorithm does not handle bound constraints. '
                'The bounds will be ignored.'
            )
            logger.warning(warning_msg)

    maxiter = 100
    if parameters is not None:
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']

    logger.info('** Optimization: Newton with linesearch')
    return newton_line_search(
        the_function=fct, starting_point=init_betas, maxiter=maxiter
    )


def newton_trust_region_for_biogeme(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on Newton method with TR.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell, u) containing the lower and
                   upper bounds for each free parameter. Note that
                   this algorithm does not support bound constraints.
                   Therefore, all the bounds will be ignored.
    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that threshold,
          the algorithm has reached convergence
          (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - maxiter: the maximum number of iterations (default: 100).
        - dogleg: if True, the trust region subproblem is solved using
          the Dogleg method. If False, it is solved using the
          truncated conjugate gradient method (default: False).
        - radius: the initial radius of the truat region (default: 1.0).

    :type parameters: dict(string:float or int)

    :return: named tuple:

            - betas is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    logger.info('Optimization algorithm: Newton with trust region [TR-newton]')

    for ell, u in bounds:
        if ell is not None or u is not None:
            warning_msg = (
                'This algorithm does not handle bound constraints. '
                'The bounds will be ignored.'
            )
            logger.warning(warning_msg)

    maxiter = 100
    apply_dogleg = False
    radius = 1.0
    if parameters is not None:
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']
        if 'dogleg' in parameters:
            apply_dogleg = parameters['dogleg']
        if 'radius' in parameters:
            radius = parameters['radius']

    logger.info('** Optimization: Newton with trust region')
    return newton_trust_region(
        the_function=fct,
        starting_point=init_betas,
        use_dogleg=apply_dogleg,
        maxiter=maxiter,
        initial_radius=radius,
    )


def bfgs_linesearch_for_biogeme(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on BFGS
    quasi-Newton method with LS.

    :param fct: object to calculate the objective function and its derivatives.

    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell,u) containing the lower and
                   upper bounds for each free parameter. Note that
                   this algorithm does not support bound constraints.
                   Therefore, all the bounds will be ignored.

    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that
          threshold, the algorithm has reached convergence
          (default: :math:`\\varepsilon^{\\frac{1}{3}}`);

        - maxiter: the maximum number of iterations (default: 100).

        - | initBfgs: the positive definite matrix that initalizes the
                      BFGS updates. If None, the identity matrix is
                      used. Default: None.


    :type parameters: dict(string:float or int)

    :return: tuple x, messages, where

            - x is the solution found,

            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.

    :rtype: numpy.array, dict(str:object)

    """
    logger.info('Optimization algorithm: BFGS with line search [LS-BFGS]')

    for ell, u in bounds:
        if ell is not None or u is not None:
            warning_msg = (
                'This algorithm does not handle bound constraints. '
                'The bounds will be ignored.'
            )
            logger.warning(warning_msg)

    maxiter = 100
    init_bfgs = None
    if parameters is not None:
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']
        if 'initBfgs' in parameters:
            init_bfgs = parameters['initBfgs']

    logger.info('** Optimization: BFGS with line search')
    return bfgs_line_search(
        the_function=fct,
        starting_point=init_betas,
        init_bfgs=init_bfgs,
        maxiter=maxiter,
    )


def bfgs_trust_region_for_biogeme(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on Newton method with TR.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell,u) containing the lower and
                   upper bounds for each free parameter. Note that
                   this algorithm does not support bound constraints.
                   Therefore, all the bounds will be ignored.
    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
         optimization routine:

         - tolerance: when the relative gradient is below that
           threshold, the algorithm has reached convergence
           (default: :math:`\\varepsilon^{\\frac{1}{3}}`);

         - maxiter: the maximum number of iterations (default: 100).

         - dogleg: if True, the trust region subproblem is solved using
           the Dogleg method. If False, it is solved using the
           truncated conjugate gradient method (default: False).

         - radius: the initial radius of the truat region (default: 1.0).

         - initBfgs: the positive definite matrix that initalizes the
           BFGS updates. If None, the identity matrix is
           used. Default: None.


    :type parameters: dict(string:float or int)

    :return: tuple x, messages, where

            - x is the solution found,
            - messages is a dictionary reporting various aspects
              related to the run of the algorithm.
    :rtype: numpy.array, dict(str:object)

    """
    logger.info('Optimization algorithm: BFGS with trust region [TR-BFGS]')
    for ell, u in bounds:
        if ell is not None or u is not None:
            warning_msg = (
                'This algorithm does not handle bound constraints. '
                'The bounds will be ignored.'
            )
            logger.warning(warning_msg)

    maxiter = 100
    apply_dogleg = False
    radius = 1.0
    init_bfgs = None
    if parameters is not None:
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']
        if 'dogleg' in parameters:
            apply_dogleg = parameters['dogleg']
        if 'radius' in parameters:
            radius = parameters['radius']
        if 'initBfgs' in parameters:
            init_bfgs = parameters['initBfgs']

    logger.info('** Optimization: BFGS with trust region')
    return bfgs_trust_region(
        the_function=fct,
        starting_point=init_betas,
        init_bfgs=init_bfgs,
        use_dogleg=apply_dogleg,
        maxiter=maxiter,
        initial_radius=radius,
    )


def simple_bounds_newton_algorithm_for_biogeme(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on variants of Newton
    method with simple bounds.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell,u) containing the lower and upper
                   bounds for each free
                   parameter. Note that this algorithm does not support bound
                   constraints.
                   Therefore, all the bounds will be ignored.
    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that threshold,
          the algorithm has reached convergence
          (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - steptol: the algorithm stops when the relative change in x
          is below this threshold. Basically, if p significant digits
          of x are needed, steptol should be set to 1.0e-p. Default:
          :math:`10^{-5}`
        - cgtolerance: when the norm of the residual is below that
          threshold, the conjugate gradient algorithm has reached
          convergence (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - proportionAnalyticalHessian: proportion (between 0 and 1) of
          iterations when the analytical Hessian is calculated (default: 1).
        - infeasibleConjugateGradient: if True, the conjugate gradient
          algorithm may generate infeasible solutiona until
          termination.  The result will then be projected on the
          feasible domain.  If False, the algorithm stops as soon as
          an infeasible iterate is generated (default: False).
        - maxiter: the maximum number of iterations (default: 1000).
        - radius: the initial radius of the trust region (default: 1.0).
        - eta1: threshold for failed iterations (default: 0.01).
        - eta2: threshold for very successful iteration (default 0.9).
        - enlargingFactor: factor used to enlarge the trust region
          during very successful iterations (default 10).

    :type parameters: dict(string:float or int)

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

    :rtype: numpay.array, dict(str:object)

    """
    logger.info(
        'Optimization algorithm: hybrid Newton/BFGS with simple bounds [simple_bounds]'
    )

    cgtol = np.finfo(np.float64).eps ** 0.3333
    maxiter = 1000
    radius = 1.0
    eta1 = 0.1
    eta2 = 0.9
    proportion_true_hessian = 1.0
    enlarging_factor = 2

    # We replace the default value by user defined value, if any.
    if parameters is not None:
        if 'cgtolerance' in parameters:
            cgtol = parameters['cgtolerance']
        if 'maxiter' in parameters:
            maxiter = parameters['maxiter']
        if 'radius' in parameters:
            radius = parameters['radius']
        if 'eta1' in parameters:
            eta1 = parameters['eta1']
        if 'eta2' in parameters:
            eta2 = parameters['eta2']
        if 'enlargingFactor' in parameters:
            enlarging_factor = parameters['enlargingFactor']
        if 'proportionAnalyticalHessian' in parameters:
            proportion_true_hessian = parameters['proportionAnalyticalHessian']

    if proportion_true_hessian == 1.0:
        logger.info('** Optimization: Newton with trust region for simple bounds')
    elif proportion_true_hessian == 0.0:
        logger.info('** Optimization: BFGS with trust region for simple bounds')
    else:
        logger.info(
            f'** Optimization: Hybrid Newton '
            f'{100*proportion_true_hessian}%/BFGS '
            f'with trust region for simple bounds'
        )
    return simple_bounds_newton_algorithm(
        the_function=fct,
        bounds=Bounds(bounds),
        starting_point=init_betas,
        variable_names=variable_names,
        proportion_analytical_hessian=proportion_true_hessian,
        first_radius=radius,
        conjugate_gradient_tol=cgtol,
        maxiter=maxiter,
        eta1=eta1,
        eta2=eta2,
        enlarging_factor=enlarging_factor,
    )


def bio_newton(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on Newton's method with simple
    bounds.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell,u) containing the lower and upper
                   bounds for each free
                   parameter. Note that this algorithm does not support bound
                   constraints.
                   Therefore, all the bounds must be None.
    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that threshold,
          the algorithm has reached convergence
          (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - steptol: the algorithm stops when the relative change in x
          is below this threshold. Basically, if p significant digits
          of x are needed, steptol should be set to 1.0e-p. Default:
          :math:`10^{-5}`
        - cgtolerance: when the norm of the residual is below that
          threshold, the conjugate gradient algorithm has reached
          convergence (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - infeasibleConjugateGradient: if True, the conjugate
          gradient algorithm may generate until termination.  The
          result will then be projected on the feasible domain.  If
          False, the algorithm stops as soon as an infeasible iterate
          is generated (default: False).
        - maxiter: the maximum number of iterations (default: 1000).
        - radius: the initial radius of the truat region (default: 1.0).
        - eta1: threshold for failed iterations (default: 0.01).
        - eta2: threshold for very successful iteration (default 0.9).
        - enlargingFactor: factor used to enlarge the trust region
          during very successful iterations (default 10).

    :type parameters: dict(string:float or int)

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

    :rtype: numpay.array, dict(str:object)

    """
    logger.info(
        'Optimization algorithm: Newton with simple bounds [simple_bounds_newton].'
    )

    if parameters is None:
        parameters = {'proportionAnalyticalHessian': 1}
    else:
        parameters['proportionAnalyticalHessian'] = 1
    return simple_bounds_newton_algorithm_for_biogeme(
        fct, init_betas, bounds, variable_names, parameters
    )


def bio_bfgs(
    fct: FunctionToMinimize,
    init_betas: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any] | None = None,
) -> OptimizationResults:
    """Optimization interface for Biogeme, based on BFGS quasi-Newton
    method with simple bounds.

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: algorithms.functionToMinimize

    :param init_betas: initial value of the parameters.

    :param bounds: list of tuples (ell,u) containing the lower and upper
                   bounds for each free
                   parameter. Note that this algorithm does not support bound
                   constraints.
                   Therefore, all the bounds must be None.
    :type bounds: list(tuples)

    :param variable_names: names of the variables.
    :type variable_names: list(str)

    :param parameters: dict of parameters to be transmitted to the
        optimization routine:

        - tolerance: when the relative gradient is below that threshold,
          the algorithm has reached convergence
          (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - steptol: the algorithm stops when the relative change in x
          is below this threshold. Basically, if p significant digits
          of x are needed, steptol should be set to 1.0e-p. Default:
          :math:`10^{-5}`
        - cgtolerance: when the norm of the residual is below that
          threshold, the conjugate gradient algorithm has reached
          convergence (default:  :math:`\\varepsilon^{\\frac{1}{3}}`);
        - infeasibleConjugateGradient: if True, the conjugate
          gradient algorithm may generate until termination.  The
          result will then be projected on the feasible domain.  If
          False, the algorithm stops as soon as an infeasible iterate
          is generated (default: False).
        - maxiter: the maximum number of iterations (default: 1000).
        - radius: the initial radius of the truat region (default: 1.0).
        - eta1: threshold for failed iterations (default: 0.01).
        - eta2: threshold for very successful iteration (default 0.9).
        - enlargingFactor: factor used to enlarge the trust region
          during very successful iterations (default 10).

    :type parameters: dict(string:float or int)

    :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the algorithm

    :rtype: numpay.array, dict(str:object)

    """
    logger.info('Optimization algorithm: BFGS with simple bounds [simple_bounds_BFGS].')

    if parameters is None:
        parameters = {'proportionAnalyticalHessian': 0}
    else:
        parameters['proportionAnalyticalHessian'] = 0
    return simple_bounds_newton_algorithm_for_biogeme(
        fct, init_betas, bounds, variable_names, parameters
    )


algorithms: dict[str, OptimizationAlgorithm] = {
    'scipy': scipy,
    'LS-newton': newton_linesearch_for_biogeme,
    'TR-newton': newton_trust_region_for_biogeme,
    'LS-BFGS': bfgs_linesearch_for_biogeme,
    'TR-BFGS': bfgs_trust_region_for_biogeme,
    'simple_bounds': simple_bounds_newton_algorithm_for_biogeme,
    'simple_bounds_newton': bio_newton,
    'simple_bounds_BFGS': bio_bfgs,
}
