"""

Using the estimated model
=========================
Once the model has been estimated, it can be applied to calculate the choice probability under several

Michel Bierlaire, EPFL
Sun Jun 15 2025, 08:30:48
"""

import pandas as pd
from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.models import logit
from biogeme.results_processing import EstimationResults
from tutorial_model import utilities

# %%
# Read the estimation results from a file
filename = 'saved_results/first_model.yaml'
estimation_results = EstimationResults.from_yaml_file(filename=filename)

scenarios = {
    'ID': pd.Series([1, 2]),
    'auto_time': pd.Series(
        [
            10.0,
            12.0,
        ]
    ),
    'transit_time': pd.Series(
        [
            4.4,
            13.0,
        ]
    ),
}
pandas_dataframe = pd.DataFrame(scenarios)
display(pandas_dataframe)

# %% Define the quantities to simulate

car_id = 0
transit_id = 1
proba_car = logit(utilities, None, car_id)
proba_transit = logit(utilities, None, transit_id)
simulate = {
    'Utility car': utilities[car_id],
    'Utility transit': utilities[transit_id],
    'Proba. car': proba_car,
    'Proba. transit': proba_transit,
}

# %%
# The data frame is used to initialize the Biogeme database.
scenarios_database = Database('ben_akiva_lerman_scenarios', pandas_dataframe)

# %%
# Now we can perform simulation
biogeme_object = BIOGEME(scenarios_database, simulate)
results = biogeme_object.simulate(the_beta_values=estimation_results.get_beta_values())

# %%
# The results are stored in a Pamdas dataframe, one row for each scenario in the original database.
display(results)
