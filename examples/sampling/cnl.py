import pandas as pd
from biogeme.sampling_of_alternatives import (
    SamplingContext,
    CrossVariableTuple,
    ChoiceSetsGeneration,
    GenerateModel,
    StratumTuple,
    generate_segment_size,
)
from biogeme.expressions import Beta, Variable, log
from biogeme.partition import Partition
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit
from true_parameters import true_parameters
from alternatives import (
    alternatives,
    ID_COLUMN,
    asian,
    downtown,
    asian_or_downtown,
    asian_and_downtown,
    all_alternatives,
    complement,
    only_asian,
    only_downtown,
)
logger = blog.get_screen_logger(level=blog.INFO)

SAMPLE_SIZE = 10
SAMPLE_SIZE_MEV = int(len(asian_or_downtown)/2)

CHOICE_COLUMN = 'cnl_3'

MODEL_NAME = f'cnl_{SAMPLE_SIZE}_{SAMPLE_SIZE_MEV}'
FILE_NAME = f'{MODEL_NAME}.dat'

PARTITION = 'downtown'

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

segment_sizes = list(
    generate_segment_size(SAMPLE_SIZE, the_partition.number_of_segments())
)

# We use all alternatives in the nest
mev_partition = Partition([asian_or_downtown])
mev_segment_size = [SAMPLE_SIZE_MEV,]


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

# Nests

# Downtown
mu_downtown = Beta('mu_downtown', 1, 1, None, 0)
downtown_alpha_dict = {i: 0.5 for i in downtown}
downtown_nest = OneNestForCrossNestedLogit(
    nest_param=mu_downtown, dict_of_alpha=downtown_alpha_dict, name='downtown'
)

# Asian
mu_asian = Beta('mu_asian', 1, 1, None, 0)
asian_alpha_dict = {i: 0.5 for i in asian}
asian_nest = OneNestForCrossNestedLogit(
    nest_param=mu_asian, dict_of_alpha=asian_alpha_dict, name='asian'
)

cnl_nests = NestsForCrossNestedLogit(
    choice_set=all_alternatives,
    tuple_of_nests=(downtown_nest, asian_nest),
)

observations = pd.read_csv('obs_choice.dat')

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
    mev_sample_size=mev_segment_size,
    cnl_nests=cnl_nests,
)

print(context.reporting())

the_data_generation = ChoiceSetsGeneration(context=context)
the_model_generation = GenerateModel(context=context)

biogeme_database = the_data_generation.sample_and_merge(recycle=False)

mu_asian = Beta('mu_asian', 1.0, 1.0, None, 0)
nest_asian = mu_asian, asian
nests = (nest_asian,)
logprob = the_model_generation.get_cross_nested_logit()

the_biogeme = bio.BIOGEME(biogeme_database, logprob)
the_biogeme.modelName = MODEL_NAME

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(context.total_sample_size)})

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
