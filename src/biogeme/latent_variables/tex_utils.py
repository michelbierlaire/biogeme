from __future__ import annotations

"""Utilities for generating TeX-safe strings.

These helpers are shared by the LaTeX and HTML (MathJax) report generators.
Their single responsibility is to convert arbitrary strings into expressions
that can safely appear inside TeX math environments.

Michel Bierlaire
Fri Mar 13 2026, 15:24:16
"""


def tex_escape(text: str) -> str:
    """Escape characters that have special meaning in TeX.

    This function is intended for variable names, parameter names, or other
    identifiers that may contain characters such as underscores. It ensures
    the string can safely appear inside TeX math mode.

    :param text: Raw text to escape.
    :return: TeX-safe string.
    """
    if not text:
        return ""

    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def tex_identifier(text: str) -> str:
    """Format an identifier so it renders nicely in math mode.

    Identifiers such as parameter names or variable names are rendered using
    ``\\mathrm{}`` to avoid them being interpreted as mathematical symbols.

    :param text: Identifier to render.
    :return: TeX expression representing the identifier.
    """
    return rf"\mathrm{{{tex_escape(text)}}}"
