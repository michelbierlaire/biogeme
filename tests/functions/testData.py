"""
Data for the tests

:author: Michel Bierlaire
:data: Wed Apr 29 18:31:18 2020

"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import pandas as pd
import biogeme.database as db

df1 = pd.DataFrame({'Person': [1, 1, 1, 2, 2],
                    'Exclude': [0, 0, 1, 0, 1],
                    'Variable1': [1, 2, 3, 4, 5],
                    'Variable2': [10, 20, 30, 40, 50],
                    'Choice': [1, 2, 3, 1, 2],
                    'Av1': [0, 1, 1, 1, 1],
                    'Av2': [1, 1, 1, 1, 1],
                    'Av3': [0, 1, 1, 1, 1]})
myData1 = db.Database('test', df1)

df2 = pd.DataFrame({'Person': [1, 1, 1, 2, 2],
                    'Exclude': [0, 0, 1, 0, 1],
                    'Variable1': [10, 20, 30, 40, 50],
                    'Variable2': [100, 200, 300, 400, 500],
                    'Choice': [1, 2, 3, 1, 2],
                    'Av1': [0, 1, 1, 1, 1],
                    'Av2': [1, 1, 1, 1, 1],
                    'Av3': [0, 1, 1, 1, 1]})
myData2 = db.Database('test', df2)
