"""

Data definition for the simple tutorial
=======================================

Example extracted from Ben-Akiva and Lerman (1985)

Michel Bierlaire, EPFL
Sun Jun 15 2025, 07:18:39

"""

import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Variable

data = {
    'ID': pd.Series([i + 1 for i in range(21)]),
    'auto_time': pd.Series(
        [
            52.9,
            4.1,
            4.1,
            56.2,
            51.8,
            0.2,
            27.6,
            89.9,
            41.5,
            95.0,
            99.1,
            18.5,
            82.0,
            8.6,
            22.5,
            51.4,
            81.0,
            51.0,
            62.2,
            95.1,
            41.6,
        ]
    ),
    'transit_time': pd.Series(
        [
            4.4,
            28.5,
            86.9,
            31.6,
            20.2,
            91.2,
            79.7,
            2.2,
            24.5,
            43.5,
            8.4,
            84.0,
            38.0,
            1.6,
            74.1,
            83.8,
            19.2,
            85.0,
            90.1,
            22.2,
            91.5,
        ]
    ),
    'choice': pd.Series(
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    ),
}
pandas_dataframe = pd.DataFrame(data)
biogeme_database = Database('ben_akiva_lerman', pandas_dataframe)
auto_time = Variable('auto_time')
transit_time = Variable('transit_time')
choice = Variable('choice')
