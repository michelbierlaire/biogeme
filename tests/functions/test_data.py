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

from copy import deepcopy
import numpy as np
import pandas as pd
import biogeme.database as db

df1 = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)


df2 = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [10, 20, 30, 40, 50],
        'Variable2': [100, 200, 300, 400, 500],
        'Choice': [2, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)

df3 = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Age': [40, 40, 40, 18, 18],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)


def getData(myid):
    data_frames = {
        1: deepcopy(df1),
        2: deepcopy(df2),
        3: deepcopy(df3),
    }
    return db.Database(f'test_{myid}', data_frames[myid])


input_flatten = pd.DataFrame(
    {
        'ID': [1, 1, 1, 2, 2],
        'Age': [23, 23, 23, 45, 45],
        'Cost': [34, 45, 12, 65, 34],
        'Name': ['Item3', 'Item4', 'Item7', 'Item3', 'Item7'],
    }
)

output_flatten_1 = pd.DataFrame(
    {
        'Age': [23, 45],
        'Item3_Cost': [34, 65],
        'Item4_Cost': [45, np.nan],
        'Item7_Cost': [12, 34],
    },
    index=[1, 2],
)

output_flatten_2 = pd.DataFrame(
    {
        'Age': [23, 45],
        '1_Cost': [34, 65],
        '1_Name': ['Item3', 'Item3'],
        '2_Cost': [45, 34],
        '2_Name': ['Item4', 'Item7'],
        '3_Cost': [12, np.nan],
        '3_Name': ['Item7', np.nan],
    },
    index=[1, 2],
)

output_flatten_3 = pd.DataFrame(
    {
        '1_Age': [23, 45],
        '1_Cost': [34, 65],
        '1_Name': ['Item3', 'Item3'],
        '2_Age': [23, 45],
        '2_Cost': [45, 34],
        '2_Name': ['Item4', 'Item7'],
        '3_Age': [23, np.nan],
        '3_Cost': [12, np.nan],
        '3_Name': ['Item7', np.nan],
    },
    index=[1, 2],
)

output_flatten_database_1 = pd.DataFrame(
    {
        '1_Age': [40, 18],
        '1_Exclude': [0, 0],
        '1_Variable1': [1, 4],
        '1_Variable2': [10, 40],
        '1_Choice': [1, 1],
        '1_Av1': [0, 1],
        '1_Av2': [1, 1],
        '1_Av3': [0, 1],
        '2_Age': [40, 18],
        '2_Exclude': [0, 1],
        '2_Variable1': [2, 5],
        '2_Variable2': [20, 50],
        '2_Choice': [2, 2],
        '2_Av1': [1, 1],
        '2_Av2': [1, 1],
        '2_Av3': [1, 1],
        '3_Age': [40, np.nan],
        '3_Exclude': [1, np.nan],
        '3_Variable1': [3, np.nan],
        '3_Variable2': [30, np.nan],
        '3_Choice': [3, np.nan],
        '3_Av1': [1, np.nan],
        '3_Av2': [1, np.nan],
        '3_Av3': [1, np.nan],
    },
    index=[1, 2],
)


output_flatten_database_2 = pd.DataFrame(
    {
        'Age': [40, 18],
        '1_Exclude': [0, 0],
        '1_Variable1': [1, 4],
        '1_Variable2': [10, 40],
        '1_Choice': [1, 1],
        '1_Av1': [0, 1],
        'Av2': [1, 1],
        '1_Av3': [0, 1],
        '2_Exclude': [0, 1],
        '2_Variable1': [2, 5],
        '2_Variable2': [20, 50],
        '2_Choice': [2, 2],
        '2_Av1': [1, 1],
        '2_Av3': [1, 1],
        '3_Exclude': [1, np.nan],
        '3_Variable1': [3, np.nan],
        '3_Variable2': [30, np.nan],
        '3_Choice': [3, np.nan],
        '3_Av1': [1, np.nan],
        '3_Av3': [1, np.nan],
    },
    index=[1, 2],
)


output_flatten_database_3 = pd.DataFrame(
    {
        'Age': [40, 18],
        '1_Exclude': [0, 0],
        '1_Variable1': [1, 4],
        '1_Variable2': [10, 40],
        '1_Choice': [1, 1],
        '1_Av1': [0, 1],
        '1_Av2': [1, 1],
        '1_Av3': [0, 1],
        '2_Exclude': [0, 1],
        '2_Variable1': [2, 5],
        '2_Variable2': [20, 50],
        '2_Choice': [2, 2],
        '2_Av1': [1, 1],
        '2_Av2': [1, 1],
        '2_Av3': [1, 1],
        '3_Exclude': [1, np.nan],
        '3_Variable1': [3, np.nan],
        '3_Variable2': [30, np.nan],
        '3_Choice': [3, np.nan],
        '3_Av1': [1, np.nan],
        '3_Av2': [1, np.nan],
        '3_Av3': [1, np.nan],
    },
    index=[1, 2],
)
