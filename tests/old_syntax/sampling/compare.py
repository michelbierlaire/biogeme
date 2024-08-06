"""

Compare parameters
==================

Function to compare estimated parameters with true parameters.

:author: Michel Bierlaire
:date: Wed Nov  1 18:07:08 2023
"""
import pandas as pd
from biogeme.results import bioResults
from true_parameters import true_parameters


# %%
def compare(estimated_parameters: bioResults) -> tuple[pd.DataFrame, str]:
    """Compare estimated and true parameters

    :param estimated_parameters: estimation results
    """
    non_estimated = []
    data = []
    for name, value in true_parameters.items():
        try:
            est_value = estimated_parameters.at[name, 'Value']
            std_err = estimated_parameters.at[name, 'Rob. Std err']
            t_test = (value - est_value) / std_err
            # Append the data to the list instead of printing
            data.append(
                {
                    'Name': name,
                    'True Value': value,
                    'Estimated Value': est_value,
                    'T-Test': t_test,
                }
            )
        except KeyError:
            non_estimated.append(name)

    # Convert the list of dictionaries to a DataFrame

    msg = f'Parameters not estimated: {non_estimated}' if non_estimated else ''
    df = pd.DataFrame(data)
    return df, msg
