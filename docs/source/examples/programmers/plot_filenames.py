"""

biogeme.filenames
=================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Wed Nov 22 13:47:01 2023
"""

from IPython.core.display_functions import display

from biogeme.filenames import get_new_file_name
from biogeme.version import get_text


# %%
# Version of Biogeme.
print(get_text())

# %%
# The role of this function is to obtain the name of a file that does not exist.
the_name = get_new_file_name('test', 'dat')
display(the_name)

# %%
# Now, let's create that file, and call the function again. A suffix
# with a number is appended to the name of the file, before its
# extension.
open(the_name, 'a').close()

# %%
the_name = get_new_file_name('test', 'dat')
display(the_name)

# %%
# If we do it again, the number is incremented.
open(the_name, 'a').close()

# %%
the_name = get_new_file_name('test', 'dat')
display(the_name)
