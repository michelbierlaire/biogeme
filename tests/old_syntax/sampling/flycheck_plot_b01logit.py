"""

Logit
=====

Estimation of a logit model using sampling of alternatives.

:author: Michel Bierlaire
:date: Wed Nov  1 17:39:47 2023
"""

import pandas as pd
from biogeme.sampling_of_alternatives import (
    SamplingContext,
    ChoiceSetsGeneration,
    GenerateModel,
    generate_segment_size,
)
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from compare import compare
from specification import V, combined_variables
from alternatives import (
    alternatives,
    ID_COLUMN,
    partitions,
)

# %%
logger = blog.get_screen_logger(level=blog.INFO)

# %%
# The data file contains several columns associated with synthetic
# choices. Here we arbitrarily select `logit_4`.
CHOICE_COLUMN = 'logit_4'

# %%
SAMPLE_SIZE = 10
PARTITION = 'asian'
MODEL_NAME = f'logit_{PARTITION}_{SAMPLE_SIZE}_alt'
FILE_NAME = f'{MODEL_NAME}.dat'
OBS_FILE = 'obs_choice.dat'

# %%
the_partition = partitions.get(PARTITION)
if the_partition is None:
    raise ValueError(f'Unknown partition: {PARTITION}')

# %%
segment_sizes = generate_segment_size(SAMPLE_SIZE, the_partition.number_of_segments())

# %%
observations = pd.read_csv(OBS_FILE)

# %%
context = SamplingContext(
    the_partition=the_partition,
    sample_sizes=segment_sizes,
    individuals=observations,
    choice_column=CHOICE_COLUMN,
    alternatives=alternatives,
    id_column=ID_COLUMN,
    biogeme_file_name=FILE_NAME,
    utility_function=V,
    combined_variables=combined_variables,
)

# %%
logger.info(context.reporting())

# %%
the_data_generation = ChoiceSetsGeneration(context=context)

# %%
the_model_generation = GenerateModel(context=context)

# %%
biogeme_database = the_data_generation.sample_and_merge(recycle=False)

# %%
logprob = the_model_generation.get_logit()

# %%
the_biogeme = bio.BIOGEME(biogeme_database, logprob)
the_biogeme.modelName = MODEL_NAME

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(SAMPLE_SIZE)})

# %%
# Estimate the parameters
results = the_biogeme.estimate(recycle=False)

# %%
print(results.short_summary())

# %%
estimated_parameters = results.getEstimatedParameters()
estimated_parameters

# %%
df, msg = compare(estimated_parameters)

# %%
print(df)

# %%
print(msg)
