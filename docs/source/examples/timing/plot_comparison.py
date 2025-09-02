    """

Comparison of execution times
=============================

Michel Bierlaire
Tue Jul 2 14:48:52 2024
"""

import pandas as pd
from IPython.core.display_functions import display

# %%
# Version 3.2.14 and 3.3.0 of Biogeme have been used to calculate a logit, a crossed-nested logit and a logit mixtures
# model on the same computer (MacBook Pro  Apple M2 Max 64 GB).


# %%
# Raw data
running_times = [
    # Logit 3.2.14
    ('logit', '3.2.14', 'Function only', None, 0.132),
    ('logit', '3.2.14', 'Function and gradient', None, 0.135),
    ('logit', '3.2.14', 'Function, gradient and hessian', None, 0.141),
    # Logit new
    ('logit', '3.3.0', 'Function only', 'with JIT', 0.0022),
    ('logit', '3.3.0', 'Function only', 'without JIT', 0.0161),
    ('logit', '3.3.0', 'Function and gradient', 'with JIT', 0.000819),
    ('logit', '3.3.0', 'Function and gradient', 'without JIT', 0.0445),
    ('logit', '3.3.0', 'Function, gradient and hessian', 'with JIT', 0.00258),
    ('logit', '3.3.0', 'Function, gradient and hessian', 'without JIT', 0.136),
    # CNL 3.2.14
    ('CNL', '3.2.14', 'Function only', None, 0.408),
    ('CNL', '3.2.14', 'Function and gradient', None, 0.362),
    ('CNL', '3.2.14', 'Function, gradient and hessian', None, 0.324),
    # CNL new
    ('CNL', '3.3.0', 'Function only', 'with JIT', 0.0034),
    ('CNL', '3.3.0', 'Function only', 'without JIT', 0.147),
    ('CNL', '3.3.0', 'Function and gradient', 'with JIT', 0.00848),
    ('CNL', '3.3.0', 'Function and gradient', 'without JIT', 0.525),
    ('CNL', '3.3.0', 'Function, gradient and hessian', 'with JIT', 0.121),
    ('CNL', '3.3.0', 'Function, gradient and hessian', 'without JIT', 1.79),
    # Mixtures 100 draws 3.2.14
    ('mixtures_100', '3.2.14', 'Function only', None, 1.38),
    ('mixtures_100', '3.2.14', 'Function and gradient', None, 1.65),
    ('mixtures_100', '3.2.14', 'Function, gradient and hessian', None, 2.12),
    # Mixtures 100 draws new
    ('mixtures_100', '3.3.0', 'Function only', 'with JIT', 0.0476),
    ('mixtures_100', '3.3.0', 'Function only', 'without JIT', 0.0503),
    ('mixtures_100', '3.3.0', 'Function and gradient', 'with JIT', 0.0185),
    ('mixtures_100', '3.3.0', 'Function and gradient', 'without JIT', 0.116),
    ('mixtures_100', '3.3.0', 'Function, gradient and hessian', 'with JIT', 0.16),
    ('mixtures_100', '3.3.0', 'Function, gradient and hessian', 'without JIT', 0.586),
    # Mixtures 1000 draws new
    ('mixtures_1000', '3.3.0', 'Function only', 'with JIT', 0.0836),
    ('mixtures_1000', '3.3.0', 'Function only', 'without JIT', 0.32),
    ('mixtures_1000', '3.3.0', 'Function and gradient', 'with JIT', 0.199),
    ('mixtures_1000', '3.3.0', 'Function and gradient', 'without JIT', 0.61),
    ('mixtures_1000', '3.3.0', 'Function, gradient and hessian', 'with JIT', 1.63),
    ('mixtures_1000', '3.3.0', 'Function, gradient and hessian', 'without JIT', 4.84),
]
df = pd.DataFrame(
    running_times, columns=['Model', 'Version', 'Computation', 'JIT', 'Time_sec']
)
display(df)


# %%
# Function evaluation
def generate_speedup_sentence(model_name, computation_type):
    old_time = df.query(
        'Model == @model_name and Version == "3.2.14" and Computation == @computation_type'
    )['Time_sec'].values[0]

    new_time = df.query(
        'Model == @model_name and Version == "3.3.0" and Computation == @computation_type and JIT == "with JIT"'
    )['Time_sec'].values[0]

    speedup = old_time / new_time
    return (
        f'For a {model_name} model, version 3.3.0 of Biogeme with JIT is '
        f'{speedup:.1f} times faster than version 3.2.14 for {computation_type.lower()}.'
    )


# %%
# Generate all speedup sentences
unique_models = df['Model'].unique()
unique_computations = df['Computation'].unique()

for model in unique_models:
    for computation in unique_computations:
        try:
            sentence = generate_speedup_sentence(model, computation)
            print(sentence)
        except IndexError:
            # Skip combinations where data is missing
            continue
