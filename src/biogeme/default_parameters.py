"""Generation of the default parameter file and values

:author: Michel Bierlaire
:date: Wed Nov 30 10:14:35 2022

IMPORTANT: when only one "check" function is provided, do not forget
to insert a comma at the end, before the closing parenthesis for the
tuple.
See https://www.w3schools.com/python/gloss_python_tuple_one_item.asp
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from tabulate import tabulate

import biogeme.check_parameters as cp
import biogeme.optimization as opt
from biogeme.floating_point import NUMPY_FLOAT
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_version

ParameterValue = bool | int | float | str

MISSING_VALUE = 99999


class ParameterTuple(NamedTuple):
    name: str
    value: ParameterValue
    type: type
    section: str
    description: str
    check: tuple[Callable[[ParameterValue], tuple[bool, str | None]], ...]


def all_parameters_tuple() -> tuple[ParameterTuple, ...]:
    from biogeme.bayesian_estimation import describe_strategies

    return (
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
            name='generate_yaml',
            value=True,
            type=bool,
            section='Output',
            description=(
                'bool: "True" if the yaml file with the results must be generated.'
            ),
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='generate_netcdf',
            value=True,
            type=bool,
            section='Output',
            description=(
                'bool: "True" if the netcdf file with the Bayesian estimation results must be generated.'
            ),
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='save_validation_results',
            value=True,
            type=bool,
            section='Output',
            description=(
                'bool: "True" if the validation results are saved in CSV files.'
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
            value=10_000,
            type=int,
            section='MonteCarlo',
            description='int: Number of draws for Monte-Carlo integration.',
            check=(cp.is_integer, cp.is_positive),
        ),
        ParameterTuple(
            name='missing_data',
            value=MISSING_VALUE,
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
            name='numerically_safe',
            value=False,
            type=bool,
            section='Specification',
            description=(
                'If true, Biogeme is doing its best to deal with numerical issues, '
                'such as division by a number close to zero, at the possible expense of speed.'
            ),
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='use_jit',
            value=True,
            type=bool,
            section='Specification',
            description=(
                'If True, the model is compiled using jit (just-in-time) to speed up the calculation. For complex '
                'models, compilation time may exceed the gain due to compilation, so that it is worth turning it off.'
            ),
            check=(cp.is_boolean,),
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
            description='int: number of re-estimations for bootstrap sampling.',
            check=(cp.is_integer, cp.is_non_negative),
        ),
        ParameterTuple(
            name='calculating_second_derivatives',
            value='analytical',
            type=str,
            section='Estimation',
            description=(
                f'Defines how to calculate the second derivatives: {",".join([value.value for value in SecondDerivativesMode])}. '
            ),
            check=(cp.check_calculating_second_derivatives,),
        ),
        ParameterTuple(
            name='large_data_set',
            value=100_000,
            type=int,
            section='Estimation',
            description=(
                'If the number of observations is larger than this value, the data set is deemed large, and the '
                'default estimation algorithm will not use second derivatives.'
            ),
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
                'If the expression contains catalogs, the parameter sets an '
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
            value='automatic',
            type=str,
            section='Estimation',
            description=(
                f'str: optimization algorithm to be used for estimation. '
                f'Valid values: {["automatic"]+list(opt.algorithms.keys())}'
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
            value=float(np.finfo(NUMPY_FLOAT).eps ** (1.0 / 3.0)),
            type=float,
            section='SimpleBounds',
            description='float: the algorithm stops when this precision is reached',
            check=(cp.is_number,),
        ),
        ParameterTuple(
            name='max_iterations',
            value=1000,
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
            value=float(np.finfo(NUMPY_FLOAT).eps ** (2.0 / 3.0)),
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
                'int: size of the largest neighborhood considered by the Variable '
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
        ParameterTuple(
            name='number_of_jobs',
            value=2,
            type=int,
            section='Bootstrap',
            description=(
                'int: The maximum number of concurrently running jobs. If -1 is given, joblib tries to use all CPUs.'
            ),
            check=(cp.is_integer,),
        ),
        ParameterTuple(
            name='version',
            value=get_version(),
            type=str,
            section='Biogeme',
            description='Version of Biogeme that created the TOML file. Do not modify this value.',
            check=(),
        ),
        ParameterTuple(
            name='mcmc_sampling_strategy',
            value='automatic',
            type=str,
            section='Bayesian',
            description=(
                f"Defines how MCMC sampling is performed: 'automatic' (selected based on hardware), {describe_strategies()}"
            ),
            check=(cp.check_sampling_strategy,),
        ),
        ParameterTuple(
            name='sample_from_prior',
            value=True,
            type=bool,
            section='Bayesian',
            description=(
                'bool: if "True", samples from the prior distributions are generated. This may help in the diagnostic of indentification issues.'
            ),
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='bayesian_draws',
            value=2000,
            type=int,
            section='Bayesian',
            description='Number of draws per chain from the posterior distribution',
            check=(cp.is_integer, cp.is_non_negative),
        ),
        ParameterTuple(
            name='warmup',
            value=2000,
            type=int,
            section='Bayesian',
            description='Number of warm-up / burn-in iterations per chain that are used only to adapt the sampler, not to estimate the posterior.',
            check=(cp.is_integer, cp.is_non_negative),
        ),
        ParameterTuple(
            name='chains',
            value=4,
            type=int,
            section='Bayesian',
            description='Number of independent Markov chains to run in parallel.',
            check=(cp.is_integer, cp.is_non_negative),
        ),
        ParameterTuple(
            name='target_accept',
            value=0.9,
            type=float,
            section='Bayesian',
            description='Target acceptance probability for the No-U-Turn Sampler (NUTS) algorithm. Higher values like 0.9 or 0.95 often work better for problematic posteriors.',
            check=(cp.is_number, cp.zero_one),
        ),
        ParameterTuple(
            name='calculate_waic',
            value=True,
            type=bool,
            section='Bayesian',
            description='Calculates the Widely Applicable Information Criterion (WAIC)',
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='calculate_loo',
            value=True,
            type=bool,
            section='Bayesian',
            description='Calculates the Leave-One-Out Cross-Validation (LOO)',
            check=(cp.is_boolean,),
        ),
        ParameterTuple(
            name='calculate_likelihood',
            value=True,
            type=bool,
            section='Bayesian',
            description='Calculates likelihood-based statistics from the posterior draws',
            check=(cp.is_boolean,),
        ),
    )


def print_list_of_parameters(table_format='plain') -> str:
    """Generate a table describing all the parameters"""

    headers = ['Parameter', 'Default value', 'Type', 'Section', 'Description']
    rows = [
        [
            parameter.name,
            parameter.value,
            parameter.type,
            parameter.section,
            parameter.description,
        ]
        for parameter in all_parameters_tuple()
    ]
    return tabulate(rows, headers=headers, tablefmt='plain')
