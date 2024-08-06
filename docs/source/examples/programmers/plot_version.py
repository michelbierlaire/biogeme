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

import biogeme.version as ver

# %%
# Obtain the version number
ver.get_version()

# %%
# Package information in text format.
print(ver.get_text())

# %%
# Package information in HTML format
print(ver.get_html())

# %%
# Package information in LaTeX format
print(ver.get_latex())

# %%
# Defines the `__version__` variable.
print(ver.__version__)
