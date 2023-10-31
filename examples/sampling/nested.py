import pandas as pd
from biogeme.sampling_of_alternatives import (
    SamplingContext,
    CrossVariableTuple,
    ChoiceSetsGeneration,
    GenerateModel,
    generate_segment_size,
)
from biogeme.expressions import Beta, Variable, log
from biogeme.partition import Partition
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from true_parameters import true_parameters
from alternatives import (
    alternatives,
    ID_COLUMN,
    asian,
    downtown,
    all_alternatives,
    complement,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.debug('DEBUGGING')

SAMPLE_SIZE = 20
SAMPLE_SIZE_MEV = len(asian)

CHOICE_COLUMN = 'nested_0'

PARTITION = 'downtown'

MODEL_NAME = f'nested_{PARTITION}_{SAMPLE_SIZE}'
FILE_NAME = f'{MODEL_NAME}.dat'

partition_asian = Partition([asian, complement(asian)])
partition_downtown = Partition([downtown, complement(downtown)])
partition_uniform = Partition([all_alternatives])

partitions = {
    'uniform': partition_uniform,
    'asian': partition_asian,
    'downtown': partition_downtown,
}

the_partition = partitions.get(PARTITION)
if the_partition is None:
    raise ValueError(f'Unknown partition: {PARTITION}')

segment_sizes = generate_segment_size(SAMPLE_SIZE, the_partition.number_of_segments())

# We use all alternatives in the nest
mev_partition = Partition([asian])
mev_segment_size = [SAMPLE_SIZE_MEV]

combined_variables = [
    CrossVariableTuple(
        'log_dist',
        log(
            (
                (Variable('user_lat') - Variable('rest_lat')) ** 2
                + (Variable('user_lon') - Variable('rest_lon')) ** 2
            )
            ** 0.5
        ),
    )
]


# Parameters to estimate
beta_rating = Beta('beta_rating', 0, None, None, 0)
beta_price = Beta('beta_price', 0, None, None, 0)
beta_chinese = Beta('beta_chinese', 0, None, None, 0)
beta_japanese = Beta('beta_japanese', 0, None, None, 0)
beta_korean = Beta('beta_korean', 0, None, None, 0)
beta_indian = Beta('beta_indian', 0, None, None, 0)
beta_french = Beta('beta_french', 0, None, None, 0)
beta_mexican = Beta('beta_mexican', 0, None, None, 0)
beta_lebanese = Beta('beta_lebanese', 0, None, None, 0)
beta_ethiopian = Beta('beta_ethiopian', 0, None, None, 0)
beta_log_dist = Beta('beta_log_dist', 0, None, None, 0)

V = (
    beta_rating * Variable('rating')
    + beta_price * Variable('price')
    + beta_chinese * Variable('category_Chinese')
    + beta_japanese * Variable('category_Japanese')
    + beta_korean * Variable('category_Korean')
    + beta_indian * Variable('category_Indian')
    + beta_french * Variable('category_French')
    + beta_mexican * Variable('category_Mexican')
    + beta_lebanese * Variable('category_Lebanese')
    + beta_ethiopian * Variable('category_Ethiopian')
    + beta_log_dist * Variable('log_dist')
)

observations = pd.read_csv('obs_choice.dat')

context = SamplingContext(
    the_partition=the_partition,
    sample_size=segment_sizes,
    individuals=observations,
    choice_column=CHOICE_COLUMN,
    alternatives=alternatives,
    id_column=ID_COLUMN,
    biogeme_file_name=FILE_NAME,
    utility_function=V,
    combined_variables=combined_variables,
    mev_partition=mev_partition,
    mev_sample_size=mev_segment_size,
)

the_data_generation = ChoiceSetsGeneration(context=context)
the_model_generation = GenerateModel(context=context)

print(f'Sample size: {context.sample_size}')
print(f'Second sample size: {context.second_sample_size}')
print(f'MEV sample size: {context.mev_sample_size}')
biogeme_database = the_data_generation.sample_and_merge(recycle=False)

mu_asian = Beta('mu_asian', 1.0, 1.0, None, 0)
nest_asian = OneNestForNestedLogit(
    nest_param=mu_asian, list_of_alternatives=asian, name='asian'
)
nests = NestsForNestedLogit(
    choice_set=all_alternatives,
    tuple_of_nests=(nest_asian,),
)
logprob = the_model_generation.get_nested_logit(nests)

logger.info(f'Sample size: {context.sample_size}')
the_biogeme = bio.BIOGEME(biogeme_database, logprob)
the_biogeme.modelName = MODEL_NAME

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(context.sample_size)})

# Estimate the parameters
results = the_biogeme.estimate(recycle=False)
print(results.short_summary())
estimated_parameters = results.getEstimatedParameters()
print(estimated_parameters)
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
df = pd.DataFrame(data)
print(df)
if non_estimated:
    print('Parameters not estimated: ', non_estimated)
