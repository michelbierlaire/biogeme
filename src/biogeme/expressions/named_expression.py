""" Decorator for the derivative calculation, so that the names of the variables are associated with their values.

:author: Michel Bierlaire
:date: Sat Apr 20 08:06:54 2024
"""

from functools import wraps

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression
from biogeme.expressions.idmanager import ElementsTuple
from biogeme.function_output import FunctionOutput, NamedFunctionOutput, convert_to_dict


def named_expression(func):
    @wraps(func)
    def wrapper(self: Expression, *args, **kwargs):
        # Call the original function/method
        result: FunctionOutput = func(self, *args, **kwargs)
        if not isinstance(result, FunctionOutput):
            raise TypeError(
                f'Expected function {func.__name__} to return FunctionOutput, got {type(result).__name__} instead.'
            )

        if self.id_manager is None:
            error_msg = f'Internal error. No id manager'
            raise BiogemeError(error_msg)
        the_betas: ElementsTuple = self.id_manager.free_betas()
        if the_betas is None:
            error_msg = f'Expression does not contain any free parameter.'
            raise BiogemeError(error_msg)

        the_result = NamedFunctionOutput(
            function=result.function,
            gradient=convert_to_dict(result.gradient, the_betas.indices),
            hessian=convert_to_dict(
                [convert_to_dict(row, the_betas.indices) for row in result.hessian],
                the_betas.indices,
            ),
        )

        return the_result

    return wrapper
