# Declare a "true" model with latent variables.
from dataclasses import dataclass

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from icecream import ic

NUMBER_OF_OBSERVATIONS = 1000


@dataclass
class IndicatorsFromLatent:
    """Characterization of the model mapping the latent variables to the indicators"""

    mean: float
    std_err: float
    loading_x1: float
    loading_x2: float

    def draw_value(
        self, first_latent_variable: float, second_latent_variable: float
    ) -> float:
        """Draw a value of the indicator

        :return: simulated indicator
        """
        return (
            self.loading_x1 * first_latent_variable
            + self.loading_x2 * second_latent_variable
            + float(np.random.normal(self.mean, self.std_err, 1)[0])
        )


I_1 = IndicatorsFromLatent(mean=1, std_err=1, loading_x1=1.5, loading_x2=-1.2)
I_2 = IndicatorsFromLatent(mean=2, std_err=2, loading_x1=-2.5, loading_x2=1.2)
I_3 = IndicatorsFromLatent(mean=3, std_err=3, loading_x1=3.5, loading_x2=-1.2)
I_4 = IndicatorsFromLatent(mean=4, std_err=4, loading_x1=-4.5, loading_x2=1.2)
I_5 = IndicatorsFromLatent(mean=5, std_err=5, loading_x1=5.5, loading_x2=-1.2)
indicators = [I_1, I_2, I_3, I_4, I_5]

simulated_data = []
for obs in range(NUMBER_OF_OBSERVATIONS):
    # Draw x1 and x2 uniformly between -10 and 10
    # x1 = float(np.random.uniform(-10, 10, 1)[0])
    x1 = float(obs % 2)
    # x2 = float(np.random.uniform(-10, 10, 1)[0])
    x2 = float(obs % 3)
    # Generate the row using the function
    the_row = {
        f'I_{index+1}': indicator.draw_value(x1, x2)
        for index, indicator in enumerate(indicators)
    }
    ic(the_row)
    simulated_data.append(the_row)

# Create the dataframe
df = pd.DataFrame(simulated_data)

display(df)

df.to_csv('simulated_data.dat', index=False)
