""" Singleton class

:author: Michel Bierlaire
:date: Sat Jul 18 16:45:35 2020

"""


class Singleton(type):
    """
    A singleton is a class with only one instance
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
