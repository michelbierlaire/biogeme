"""Interface with the C++ implementation

:author: Michel Bierlaire
:date: Sat Sep  9 15:25:07 2023

"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import cythonbiogeme.cythonbiogeme as ee

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .elementary_expressions import TypeOfElementaryExpression

from biogeme.function_output import (
    BiogemeFunctionOutput,
    BiogemeDisaggregateFunctionOutput,
)


logger = logging.getLogger(__name__)


def calculate_aggregate_function_and_derivatives(
    the_expression: Expression,
    betas: dict[str, float],
    database: Database,
    calculate_gradient: bool,
    calculate_hessian: bool,
    calculate_bhhh: bool,
) -> BiogemeFunctionOutput:
    """Uses Jax to calculate the expression.

    :param the_expression: expression to calculate for each entry in the database.
    :param betas: values for the Beta parameters. If irrelevant
       parameters are included, an exception is raised.
    :param database: database. If no database is provided, the
        expression must not contain any variable.
    :param calculate_gradient: If True, the gradient is calculated.
    :param calculate_hessian: if True, the hessian is  calculated.
    :param calculate_bhhh: if True, the BHHH matrix is calculated.

    :raise BiogemeError: if a database is needed and not available.
    """
    if not calculate_gradient and (calculate_hessian or calculate_bhhh):
        raise ValueError(
            'If you need second order derivatives, you also need first order.'
        )

    if not betas and calculate_gradient:
        raise BiogemeError('The gradient is requested but no parameter is provided.')

    if calculate_gradient:
        betas_in_expression = the_expression.dict_of_elementary_expression(
            TypeOfElementaryExpression.FREE_BETA
        )
        missing_betas = betas.keys() - betas_in_expression.keys()
        if missing_betas:
            raise BiogemeError(
                f'Parameters not present in the arithmetic expression: {missing_betas}'
            )

    number_of_draws = database.number_of_draws

    # Prepare betas
    self.id_manager.free_betas_values = [
        (
            float(betas[x])
            if x in betas
            else self.id_manager.free_betas.expressions[x].initValue
        )
        for x in self.id_manager.free_betas.names
    ]
    # List of values of the fixed Beta parameters (those not estimated)
    self.fixedBetaValues = [
        (
            float(betas[x])
            if x in betas
            else self.id_manager.fixed_betas.expressions[x].initValue
        )
        for x in self.id_manager.fixed_betas.names
    ]

    the_function = self.recursive_construct_jax_function()

    vectorized_function = build_vectorized_function(the_function)

    @jit
    def sum_function(params, data, draws):
        val = vectorized_function(params, data, draws)
        return jnp.asarray(jnp.sum(val), dtype=JAX_FLOAT)

    data_jax = jnp.asarray(database.data.to_numpy(), dtype=JAX_FLOAT)
    draws_jax = jnp.asarray(
        database.theDraws
        if database.theDraws is not None
        else jnp.zeros((database.get_number_of_observations(), 0, 0))
    )

    if not gradient or not self.id_manager.free_betas_values:
        value_jax = sum_function(self.id_manager.free_betas_values, data_jax, draws_jax)
        value = float(value_jax)
        results = BiogemeFunctionOutput(
            function=float(value),
            gradient=None,
            hessian=None,
            bhhh=None,
        )
        self.set_id_manager(self.keep_id_manager)
        return results
    try:
        value_and_grad_fn = jax.jit(jax.value_and_grad(sum_function, argnums=0))
        value, the_gradient = value_and_grad_fn(
            self.id_manager.free_betas_values, data_jax, draws_jax
        )
    except (TracerIntegerConversionError, TypeError) as e:
        raise BiogemeError(
            "This expression is not differentiable with JAX. "
            "It likely involves discrete or logical operations such as AND, OR, or IF."
        ) from e
    if hessian:
        if jnp.all(the_gradient == 0.0):
            the_hessian = jnp.zeros(
                (
                    len(self.id_manager.free_betas_values),
                    len(self.id_manager.free_betas_values),
                ),
                dtype=JAX_FLOAT,
            )
        else:
            hessian_fn = jax.jit(
                jax.jacfwd(jax.grad(sum_function, argnums=0), argnums=0)
            )
            the_hessian = hessian_fn(
                self.id_manager.free_betas_values, data_jax, draws_jax
            )
    else:
        the_hessian = None

    results = BiogemeFunctionOutput(
        function=float(value),
        gradient=np.asarray(the_gradient),
        hessian=np.asarray(the_hessian),
        bhhh=None,
    )

    self.set_id_manager(self.keep_id_manager)

    return results
