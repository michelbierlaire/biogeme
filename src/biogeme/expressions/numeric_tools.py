""" Function to check and process numeric expressions

:author: Michel Bierlaire
:date: Sat Sep  9 15:27:17 2023
"""
def is_numeric(obj):
    """Checks if an object is numeric
    :param obj: obj to be checked
    :type obj: Object

    """
    return isinstance(obj, (int, float, bool))

