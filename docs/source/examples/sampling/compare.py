"""

Compare parameters
==================

Function to compare estimated parameters with true parameters.

Michel Bierlaire
Fri Jul 25 2025, 17:43:55
"""

import pandas as pd

from true_parameters import true_parameters


# %%
def compare(estimated_parameters: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compare estimated and true parameters

    :param estimated_parameters: estimation results
    """
    non_estimated = []
    data = []
    for name, value in true_parameters.items():
        try:
            est_value = estimated_parameters.loc[
                estimated_parameters['Name'] == name, 'Value'
            ].values[0]
            std_err = estimated_parameters.loc[
                estimated_parameters['Name'] == name, 'Robust std err.'
            ].values[0]
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
        except IndexError:
            non_estimated.append(name)

    # Convert the list of dictionaries to a DataFrame

    msg = f'Parameters not estimated: {non_estimated}' if non_estimated else ''
    df = pd.DataFrame(data)
    return df, msg
