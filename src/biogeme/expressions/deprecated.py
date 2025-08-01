"""Deprecated expression, for backward compatibility

Michel Bierlaire
Thu Apr 3 09:55:57 2025
"""

import warnings

from .binary_max import BinaryMax
from .binary_min import BinaryMin
from .draws import Draws
from .integrate import IntegrateNormal
from .linear_utility import LinearUtility
from .multiple_sum import MultipleSum
from .normalcdf import NormalCdf


def deprecated_wrapper(old_name, new_class, comment=None):
    def wrapper(*args, **kwargs):
        message = (
            f"'{old_name}' is deprecated and will be removed in a future version. "
            f"Use '{new_class.__name__}' instead."
        )
        if comment:
            message += f"\n{comment}"
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return new_class(*args, **kwargs)

    wrapper.__name__ = old_name
    doc = (
        f"Deprecated wrapper for :class:`{new_class.__name__}`.\n\n"
        f".. warning::\n   This function is deprecated and will be removed in a future version.\n"
        f"   Use :class:`{new_class.__name__}` instead.\n"
    )
    if comment:
        doc += f"\n   {comment}\n"
    wrapper.__doc__ = doc
    return wrapper


bioLinearUtility = deprecated_wrapper("bioLinearUtility", LinearUtility)
bioMultSum = deprecated_wrapper("bioMultSum", MultipleSum)
bioDraws = deprecated_wrapper("bioDraws", Draws)
bioNormalCdf = deprecated_wrapper("bioNormalCdf", NormalCdf)
bioMin = deprecated_wrapper("bioMin", BinaryMin)
bioMax = deprecated_wrapper("bioMax", BinaryMax)
comment_integrate = (
    'In Biogeme 3.2, Integrate calculated the integral from -infinity to +infinity of f(x) dx.\n'
    'Since Biogeme 3.3, it is not available anymore.\n'
    'Instead, IntegrateNormal calculates the integral from -infinity to +infinity of f(x) * phi(x) dx,\n'
    'where phi(x) is the probability density function of the normal distribution.'
)
Integrate = deprecated_wrapper("Integrate", IntegrateNormal, comment=comment_integrate)
