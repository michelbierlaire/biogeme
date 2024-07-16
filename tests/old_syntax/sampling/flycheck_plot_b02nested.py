"""

Nested logit
============

Estimation of a nested logit model using sampling of alternatives.

:author: Michel Bierlaire
:date: Wed Nov  1 18:00:15 2023
"""
import pandas as pd
from biogeme.sampling_of_alternatives import (
    SamplingContext,
    ChoiceSetsGeneration,
    GenerateModel,
    generate_segment_size,
)
from biogeme.expressions import Beta
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from specification import V, combined_variables
from compare import compare
from alternatives import (
    alternatives,
    ID_COLUMN,
    partitions,
    asian,
    all_alternatives,
)

# %%
logger = blog.get_screen_logger(level=blog.INFO)

# %%
SAMPLE_SIZE = 20  # out of 100
SAMPLE_SIZE_MEV = 33  # out of 33
CHOICE_COLUMN = 'nested_0'
PARTITION = 'downtown'
MEV_PARTITION = 'uniform_asian'
MODEL_NAME = f'nested_{PARTITION}_{SAMPLE_SIZE}'
FILE_NAME = f'{MODEL_NAME}.dat'

# %%
the_partition = partitions.get(PARTITION)
if the_partition is None:
    raise ValueError(f'Unknown partition: {PARTITION}')

# %%
segment_sizes = generate_segment_size(SAMPLE_SIZE, the_partition.number_of_segments())

# %%
# We use all alternatives in the nest.
mev_partition = partitions.get(MEV_PARTITION)
if mev_partition is None:
    raise ValueError(f'Unknown partition: {MEV_PARTITION}')
mev_segment_sizes = [SAMPLE_SIZE_MEV]

# %%
observations = pd.read_csv('obs_choice.dat')

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
    mev_partition=mev_partition,
    mev_sample_sizes=mev_segment_sizes,
)

# %%
logger.info(context.reporting())

# %%
the_data_generation = ChoiceSetsGeneration(context=context)
the_model_generation = GenerateModel(context=context)

# %%
biogeme_database = the_data_generation.sample_and_merge(recycle=False)

# %%
# Definition of the nest.
mu_asian = Beta('mu_asian', 1.0, 1.0, None, 0)
nest_asian = OneNestForNestedLogit(
    nest_param=mu_asian, list_of_alternatives=asian, name='asian'
)
nests = NestsForNestedLogit(
    choice_set=all_alternatives,
    tuple_of_nests=(nest_asian,),
)

# %%
logprob = the_model_generation.get_nested_logit(nests)

# %%
the_biogeme = bio.BIOGEME(biogeme_database, logprob)
the_biogeme.modelName = MODEL_NAME

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(context.total_sample_size)})

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
