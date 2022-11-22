""" Installation set up for Biogeme

:author: Michel Bierlaire
:date: Tue May 31 20:20:17 2022

"""
import os
import platform
from cysetuptools import setup

if platform.system() == "Darwin":
    os.environ["CC"] = "clang++"
    os.environ["CXX"] = "clang++"
else:
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"


setup()
