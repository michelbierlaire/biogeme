"""Generation of the default parameter file and values

:author: Michel Bierlaire
:date: Wed Nov 30 10:14:35 2022

IMPORTANT: when only one "check" function is provided, do not forget
to insert a comma at the end, before the closing parenthesis for the
tuple.
See https://www.w3schools.com/python/gloss_python_tuple_one_item.asp
"""

from typing import NamedTuple, Union, Type, Callable
import numpy as np
import biogeme.optimization as opt
import biogeme.check_parameters as cp


class ParameterTuple(NamedTuple):
    name: str
    value: Union[bool, int, float, str]
    type: Type
    section: str
    description: str
    check: Callable


all_parameters_tuple = (
    ParameterTuple(
        name='identification_threshold',
        value=1.0e-5,
        type=float,
        section='Output',
        description=(
            'float: if the smallest eigenvalue of the second derivative '
            'matrix is lesser or equal to this parameter, the model is '
            'considered not identified. The corresponding eigenvector '
            'is then reported to identify the parameters involved in the issue.'
        ),
        check=(cp.is_number,),
    ),
    ParameterTuple(
        name='only_robust_stats',
        value=True,
        type=bool,
        section='Output',
        description=(
            'bool: "True" if only the robust statistics need to be reported.'
            ' If "False", the statistics from the Rao-Cramer bound are '
            'also reported.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='generate_html',
        value=True,
        type=bool,
        section='Output',
        description=(
            'bool: "True" if the HTML file with the results must be generated.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='generate_pickle',
        value=True,
        type=bool,
        section='Output',
        description=(
            'bool: "True" if the pickle file with the ' 'results must be generated.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='number_of_threads',
        value=0,
        type=int,
        section='MultiThreading',
        description=(
            'int: Number of threads/processors to be used. If'
            ' the parameter is 0, the number of'
            ' available threads is calculated using'
            ' cpu_count().'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='number_of_draws',
        value=20000,
        type=int,
        section='MonteCarlo',
        description=('int: Number of draws for Monte-Carlo integration.'),
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='missing_data',
        value=99999,
        type=int,
        section='Specification',
        description=(
            'number: If one variable has this value, it is assumed'
            ' that a data is missing and an exception will'
            ' be triggered.'
        ),
        check=(cp.is_number,),
    ),
    ParameterTuple(
        name='seed',
        value=0,
        type=int,
        section='MonteCarlo',
        description=(
            'int: Seed used for the pseudo-random number generation. It is'
            ' useful only when each run should generate the exact same'
            ' result. If 0, a new seed is used at each run.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='bootstrap_samples',
        value=100,
        type=int,
        section='Estimation',
        description=('int: number of re-estimations for bootstrap sampling.'),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='max_number_parameters_to_report',
        value=15,
        type=int,
        section='Estimation',
        description=(
            'int: maximum number of parameters to report during the estimation.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='save_iterations',
        value=True,
        type=bool,
        section='Estimation',
        description=(
            'bool: If True, the current iterate is saved after'
            ' each iteration, in a file named'
            ' ``__[modelName].iter``, where'
            ' ``[modelName]`` is the name given to the'
            ' model. If such a file exists, the starting'
            ' values for the estimation are replaced by'
            ' the values saved in the file.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='maximum_number_catalog_expressions',
        value=100,
        type=int,
        section='Estimation',
        description=(
            'If the expression contrains catalogs, the parameter sets an '
            'upper bound of the total number of possible combinations '
            'that can be estimated in the same loop.'
        ),
        check=(
            cp.is_integer,
            cp.is_positive,
        ),
    ),
    ParameterTuple(
        name='optimization_algorithm',
        value='simple_bounds',
        type=str,
        section='Estimation',
        description=(
            f'str: optimization algorithm to be used for estimation. '
            f'Valid values: {list(opt.algorithms.keys())}'
        ),
        check=(cp.check_algo_name,),
    ),
    ParameterTuple(
        name='second_derivatives',
        value=1.0,
        type=float,
        section='SimpleBounds',
        description=(
            'float: proportion (between 0 and 1) of iterations when the '
            'analytical Hessian is calculated'
        ),
        check=(cp.zero_one, cp.is_number),
    ),
    ParameterTuple(
        name='tolerance',
        value=np.finfo(np.float64).eps ** 0.3333,
        type=float,
        section='SimpleBounds',
        description='float: the algorithm stops when this precision is reached',
        check=(cp.is_number,),
    ),
    ParameterTuple(
        name='max_iterations',
        value=100,
        type=int,
        section='SimpleBounds',
        description='int: maximum number of iterations',
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='infeasible_cg',
        value=False,
        type=bool,
        section='SimpleBounds',
        description=(
            'If True, the conjugate gradient algorithm may generate '
            'infeasible solutions until termination.  The result will '
            'then be projected on the feasible domain.  If False, the '
            'algorithm stops as soon as an infeasible iterate is '
            'generated'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='initial_radius',
        value=1,
        type=float,
        section='SimpleBounds',
        description='Initial radius of the trust region',
        check=(cp.is_number, cp.is_positive),
    ),
    ParameterTuple(
        name='steptol',
        value=1.0e-5,
        type=float,
        section='SimpleBounds',
        description=(
            'The algorithm stops when the relative change in x is below '
            'this threshold. Basically, if p significant digits of x '
            'are needed, steptol should be set to 1.0e-p.'
        ),
        check=(cp.is_number, cp.is_positive),
    ),
    ParameterTuple(
        name='enlarging_factor',
        value=10,
        type=float,
        section='SimpleBounds',
        description=(
            'If an iteration is very successful, the radius of '
            'the trust region is multiplied by this factor'
        ),
        check=(cp.is_number, cp.is_positive),
    ),
    ParameterTuple(
        name='dogleg',
        value=True,
        type=bool,
        section='TrustRegion',
        description=(
            'bool: choice of the method to solve the trust region subproblem. '
            'True: dogleg. False: truncated conjugate gradient.'
        ),
        check=(cp.is_boolean,),
    ),
    ParameterTuple(
        name='maximum_number_parameters',
        value=50,
        type=int,
        section='AssistedSpecification',
        description=(
            'int: maximum number of parameters allowed in a model. Each specification '
            'with a higher number is deemed invalid and not estimated.'
        ),
        check=(cp.is_integer, cp.is_positive),
    ),
    ParameterTuple(
        name='number_of_neighbors',
        value=20,
        type=int,
        section='AssistedSpecification',
        description=(
            'int: maximum number of neighbors that are visited by the VNS algorithm.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='largest_neighborhood',
        value=20,
        type=int,
        section='AssistedSpecification',
        description=(
            'int: size of the largest neighborhood copnsidered by the Variable '
            'Neighborhood Search (VNS) algorithm.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
    ParameterTuple(
        name='maximum_attempts',
        value=100,
        type=int,
        section='AssistedSpecification',
        description=(
            'int: an attempts consists in selecting a solution in the Pareto '
            'set, and trying to improve it. The parameter imposes an upper bound '
            'on the total number of attempts, irrespectively if they are '
            'successful or not.'
        ),
        check=(cp.is_integer, cp.is_non_negative),
    ),
)
