"""

biogeme.version
===============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Thu Dec  7 15:22:06 2023
"""

from biogeme.version import get_version, get_text, get_html, get_latex, __version__

# %%
# Obtain the version number
get_version()

# %%
# Package information in text format.
print(get_text())

# %%
# Package information in HTML format
print(get_html())

# %%
# Package information in LaTeX format
print(get_latex())

# %%
# Defines the `__version__` variable.
print(__version__)
