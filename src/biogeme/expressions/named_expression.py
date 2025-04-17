"""Decorator for the derivative calculation, so that the names of the variables are associated with their values.

:author: Michel Bierlaire
:date: Sat Apr 20 08:06:54 2024
"""

from functools import wraps

from biogeme.expressions import Expression
from biogeme.function_output import FunctionOutput, NamedFunctionOutput, convert_to_dict


def named_function_output(func):
    """
    Decorator to convert a FunctionOutput into a NamedFunctionOutput,
    using explicitly passed beta indices.

    The decorated function must now accept an `indices` keyword argument.
    """

    @wraps(func)
    def wrapper(self: Expression, *args, indices=None, **kwargs):
        if indices is None:
            raise ValueError(
                "Parameter 'indices' must be provided to use @named_function_output."
            )
        # Call the original function/method
        result: FunctionOutput = func(self, *args, **kwargs)
        if not isinstance(result, FunctionOutput):
            raise TypeError(
                f'Expected function {func.__name__} to return FunctionOutput, got {type(result).__name__} instead.'
            )

        the_result = NamedFunctionOutput(
            function=result.function,
            gradient=convert_to_dict(result.gradient, indices),
            hessian=convert_to_dict(
                [convert_to_dict(row, indices) for row in result.hessian],
                indices,
            ),
        )

        return the_result

    return wrapper
