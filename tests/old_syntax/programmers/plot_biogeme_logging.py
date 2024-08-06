"""

biogeme.biogeme_logging
=======================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Wed Nov 22 13:52:20 2023
"""

# %%
import os
import biogeme.version as ver
import biogeme.biogeme_logging as blog

# %%
# Version of Biogeme.
print(ver.getText())

# %%
# In Python, the levels of reporting are:
#
#     - DEBUG
#     - INFO
#     - WARNING
#     - ERROR
#     - CRITICAL
#
# In Biogeme, we basically use the first three.

# %%
# If we request a specific level, all message from this level and all
# levels above are displayed. For example, if INFO is requested,
# everything except DEBUG will be diplayed.
logger = blog.get_screen_logger(level=blog.INFO)


# %%
logger.info('A test')

# %%
# If a debug message is generated, it is not displayed, as the INFO
# level has been requested above.

# %%
logger.debug('A debug message')

# %%
# But a warning message is displayed, as it comes higher in the hierarchy.

# %%
logger.warning('A warning message')

# %%
# It is also possible to log the messages on file.

# %%
THE_FILE = '_test.log'

# %%
# Let's first erase the file if it happens to exist.
try:
    os.remove(THE_FILE)
    print(f'File {THE_FILE} has been erased.')
except FileNotFoundError:
    print('File {THE_FILE} does not exist.')

# %%
file_logger = blog.get_file_logger(filename=THE_FILE, level=blog.DEBUG)

# %%
file_logger.debug('A debug message')
file_logger.warning('A warning message')
file_logger.info('A info message')

# %%
# Here is the content of the log file. Note that the message includes
# the filename, which is not informative in the context of this
# Notebook.

# %%
with open(THE_FILE, encoding='utf-8') as f:
    print(f.read())
