"""Deprecated expression, for backward compatibility

Michel Bierlaire
Thu Apr 3 09:55:57 2025
"""

import warnings
from .nary_expressions import LinearUtility, MultipleSum
from .elementary_expressions import Draws
from .unary_expressions import NormalCdf


def bioLinearUtility(*args, **kwargs):
    """
    Deprecated wrapper for :class:`LinearUtility`.

    .. warning::
       This function is deprecated and will be removed in a future version.
       Use :class:`LinearUtility` instead.

    This function issues a deprecation warning and returns an instance of
    :class:`LinearUtility` with the provided arguments.

    :param \\*args: Positional arguments passed to :class:`LinearUtility`.
    :param \\*\\*kwargs: Keyword arguments passed to :class:`LinearUtility`.
    :return: An instance of :class:`LinearUtility`.
    :rtype: LinearUtility
    """
    warnings.warn(
        "'bioLinearUtility' is deprecated and will be removed in a future version. "
        "Use 'LinearUtility' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return LinearUtility(*args, **kwargs)


def bioMultSum(*args, **kwargs):
    """
    Deprecated wrapper for :class:`MultipleSum`.

    .. warning::
       This function is deprecated and will be removed in a future version.
       Use :class:`MultipleSum` instead.

    This function issues a deprecation warning and returns an instance of
    :class:`MultipleSum` with the provided arguments.

    :param \\*args: Positional arguments passed to :class:`MultipleSum`.
    :param \\*\\*kwargs: Keyword arguments passed to :class:`MultipleSum`.
    :return: An instance of :class:`MultipleSum`.
    :rtype: LinearUtility
    """
    warnings.warn(
        "'bioMultSum' is deprecated and will be removed in a future version. "
        "Use 'MultipleSum' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return MultipleSum(*args, **kwargs)


def bioDraws(*args, **kwargs):
    """
    Deprecated wrapper for :class:`Draws`.

    .. warning::
       This function is deprecated and will be removed in a future version.
       Use :class:`Draws` instead.

    This function issues a deprecation warning and returns an instance of
    :class:`Draws` with the provided arguments.

    :param \\*args: Positional arguments passed to :class:`Draws`.
    :param \\*\\*kwargs: Keyword arguments passed to :class:`Draws`.
    :return: An instance of :class:`Draws`.
    :rtype: Draws
    """
    warnings.warn(
        "'bioDraws' is deprecated and will be removed in a future version. "
        "Use 'Draws' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Draws(*args, **kwargs)


def bioNormalCdf(*args, **kwargs):
    """
    Deprecated wrapper for :class:`NormalCdf`.

    .. warning::
       This function is deprecated and will be removed in a future version.
       Use :class:`NormalCdf` instead.

    This function issues a deprecation warning and returns an instance of
    :class:`NormalCdf` with the provided arguments.

    :param \\*args: Positional arguments passed to :class:`Draws`.
    :param \\*\\*kwargs: Keyword arguments passed to :class:`Draws`.
    :return: An instance of :class:`NormalCdf`.
    """
    warnings.warn(
        "'bioNormalPdf' is deprecated and will be removed in a future version. "
        "Use 'NormalPdf' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return NormalCdf(*args, **kwargs)
