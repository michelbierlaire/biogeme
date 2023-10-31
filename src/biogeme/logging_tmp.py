import warnings
from .biogeme_logging import *  # Import everything from the new module

# Issue the deprecation warning
warnings.warn(
    "The module 'biogeme.logging' is deprecated and will be removed in a future version. "
    "Please use 'biogeme.biogeme_logging' instead. This is due to a name conflict "
    "with the main logging module.",
    DeprecationWarning,
    stacklevel=2,
)
