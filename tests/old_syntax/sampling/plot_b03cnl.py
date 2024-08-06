"""

Cross-nested logit
==================

Estimation of a cross-nested logit model using sampling of alternatives.

:author: Michel Bierlaire
:date: Wed Nov  1 18:00:33 2023
"""
import pandas as pd
from biogeme.sampling_of_alternatives import (
    SamplingContext,
    ChoiceSetsGeneration,
    GenerateModel,
    generate_segment_size,
)
from biogeme.expressions import Beta
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit
from specification import V, combined_variables
from compare import compare
from alternatives import (
    alternatives,
    ID_COLUMN,
    partitions,
    all_alternatives,
    asian_and_downtown,
    only_downtown,
    only_asian,
)

# %%
logger = blog.get_screen_logger(level=blog.INFO)

# %%
PARTITION = 'downtown'
MEV_PARTITION = 'uniform_asian_or_downtown'
SAMPLE_SIZE = 10  # out of 100 alternatives
SAMPLE_SIZE_MEV = 63  # out of 63 alternatives
CHOICE_COLUMN = 'cnl_3'
MODEL_NAME = f'cnl_{SAMPLE_SIZE}_{SAMPLE_SIZE_MEV}'
FILE_NAME = f'{MODEL_NAME}.dat'

# %%
the_partition = partitions.get(PARTITION)
if the_partition is None:
    raise ValueError(f'Unknown partition: {PARTITION}')

# %%
segment_sizes = list(
    generate_segment_size(SAMPLE_SIZE, the_partition.number_of_segments())
)

# %%
# We use all alternatives in the nest.
mev_partition = partitions.get(MEV_PARTITION)
if mev_partition is None:
    raise ValueError(f'Unknown partition: {MEV_PARTITION}')
mev_segment_sizes = [
    SAMPLE_SIZE_MEV,
]

# %%
# Nests

# %%
# Downtown
mu_downtown = Beta('mu_downtown', 1, 1, None, 0)
downtown_alpha_dict = {i: 0.5 for i in asian_and_downtown} | {
    i: 1 for i in only_downtown
}
downtown_nest = OneNestForCrossNestedLogit(
    nest_param=mu_downtown, dict_of_alpha=downtown_alpha_dict, name='downtown'
)

# %%
# Asian
mu_asian = Beta('mu_asian', 1, 1, None, 0)
asian_alpha_dict = {i: 0.5 for i in asian_and_downtown} | {i: 1.0 for i in only_asian}
asian_nest = OneNestForCrossNestedLogit(
    nest_param=mu_asian, dict_of_alpha=asian_alpha_dict, name='asian'
)

cnl_nests = NestsForCrossNestedLogit(
    choice_set=all_alternatives,
    tuple_of_nests=(downtown_nest, asian_nest),
)

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
    cnl_nests=cnl_nests,
)

# %%
logger.info(context.reporting())

# %%
the_data_generation = ChoiceSetsGeneration(context=context)
the_model_generation = GenerateModel(context=context)

# %%
biogeme_database = the_data_generation.sample_and_merge(recycle=False)

# %%
logprob = the_model_generation.get_cross_nested_logit()

# %%
the_biogeme = bio.BIOGEME(biogeme_database, logprob)
the_biogeme.modelName = MODEL_NAME

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(context.total_sample_size)})

# %%
# Estimate the parameters.
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
