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
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit
from true_parameters import true_parameters
from alternatives import (
    alternatives,
    ID_COLUMN,
    asian,
    all_alternatives,
    complement,
)

SAMPLE_SIZE = 20
SAMPLE_SIZE_MEV = len(asian) + 1

CHOICE_COLUMN = 'choice_nested'

MODEL_NAME = f'nested_cnl_{SAMPLE_SIZE}_{SAMPLE_SIZE_MEV}'
FILE_NAME = f'{MODEL_NAME}.dat'

segment_sizes = generate_segment_size(SAMPLE_SIZE, 2)

partition_asian = (
    StratumTuple(subset=asian, sample_size=segment_sizes[0]),
    StratumTuple(subset=complement(asian), sample_size=segment_sizes[1]),
)


partition_asian_mev = (
    StratumTuple(subset=asian, sample_size=SAMPLE_SIZE_MEV - 1),
    StratumTuple(subset=complement(asian), sample_size=1),
)

the_first_partition = partition_asian
the_second_partition = partition_asian_mev

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

# Asian
mu_asian = Beta('mu_asian', 1, 1, None, 0)
asian_alpha_dict = {i: 1 for i in asian}
asian_nest = OneNestForCrossNestedLogit(
    nest_param=mu_asian, dict_of_alpha=asian_alpha_dict, name='asian'
)

cnl_nests = NestsForCrossNestedLogit(
    choice_set=all_alternatives,
    tuple_of_nests=(asian_nest,),
)

observations = pd.read_csv('obs_choice.dat')

context = SamplingContext(
    partition=the_first_partition,
    individuals=observations,
    choice_column=CHOICE_COLUMN,
    alternatives=alternatives,
    id_column=ID_COLUMN,
    biogeme_file_name=FILE_NAME,
    utility_function=V,
    combined_variables=combined_variables,
    second_partition=the_second_partition,
    cnl_nests=cnl_nests,
)

logger = blog.get_screen_logger(level=blog.INFO)

the_data_generation = ChoiceSetsGeneration(context=context)
the_model_generation = GenerateModel(context=context)

print(f'Sample size: {context.sample_size}')
print(f'Second sample size: {context.second_sample_size}')
print(f'MEV sample size: {context.mev_sample_size}')
biogeme_database = the_data_generation.sample_and_merge(overwrite=True)
logprob = the_model_generation.get_cross_nested_logit()

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
