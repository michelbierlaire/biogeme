"""Script generating the synthetic choices

"""
import os
import numpy as np
import pandas as pd
import biogeme.biogeme as bio
from biogeme.expressions import Variable, log
from biogeme.database import Database
from biogeme.models import loglogit, lognested, logcnl
from biogeme.nests import (
    OneNestForNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
)
from alternatives import (
    asian,
    downtown,
    asian_and_downtown,
    only_asian,
    only_downtown,
    all_alternatives,
)
from true_parameters import true_parameters

NUMBER_OF_DRAWS = 5

DATA_FILE_NAME = 'result.csv.gz'

LOGIT_FILE_NAME = 'logit.csv.gz'
NESTED_FILE_NAME = 'nested.csv.gz'
CNL_FILE_NAME = 'cnl.csv.gz'
LOGIT_DRAWS = 'logit_draws.csv.gz'
NESTED_DRAWS = 'nested_draws.csv.gz'
CNL_DRAWS = 'cnl_draws.csv.gz'
RESTAURANTS_FILE_NAME = 'restaurants.dat'
OBS_FILE_NAME = 'individuals.dat'
# OBS_FILE_NAME = 'tiny.dat'

CHOICE_FILE_NAME = 'obs_choice.dat'
# CHOICE_FILE_NAME = 'tiny_choice.dat'


def choice_from_logp(logp: np.ndarray) -> int:
    """Draw a choice given the vector of log probabilities

    :param: array with the log of probabilities
    """
    the_sum = sum(np.exp(logp))
    if not np.isclose(the_sum, 1):
        raise ValueError(f'The probabilities do NOT sum up to one: {the_sum}.')

    # Subtracting the max value for numerical stability
    adjusted_probs = np.exp(logp - np.max(logp))

    # Normalize the probabilities
    normalized_probs = adjusted_probs / adjusted_probs.sum()

    if not np.isclose(sum(normalized_probs), 1):
        raise ValueError('The probabilities do NOT sum up to one.')

    return np.random.choice(len(logp), p=normalized_probs)


generate_data = not os.path.exists(DATA_FILE_NAME)
restaurants = pd.read_csv(RESTAURANTS_FILE_NAME)
obs_choice = pd.read_csv(OBS_FILE_NAME)

nbr_of_alternatives = restaurants.shape[0]
nbr_of_observations = obs_choice.shape[0]

if generate_data:
    # Prepare the expanded data and the new column names
    expanded_data = []
    new_columns = []

    for _, row in restaurants.iterrows():
        restaurant_id = row['ID']
        for col in restaurants.columns:
            new_columns.append(f'{col}_{int(restaurant_id)}')
            expanded_data.append(row[col])

    # Repeat the expanded data as many times as there are rows in obs_choice
    expanded_data = [expanded_data] * nbr_of_observations

    # Create a dataframe from the expanded data
    expanded_df = pd.DataFrame(expanded_data, columns=new_columns)

    # Concatenate the expanded dataframe with obs_choice
    result = pd.concat([obs_choice.reset_index(drop=True), expanded_df], axis=1)

    # Calculate the Euclidean distance
    for i in range(nbr_of_alternatives):
        rest_lat = result[f'rest_lat_{i}']
        rest_lon = result[f'rest_lon_{i}']
        user_lat = result['user_lat']
        user_lon = result['user_lon']
        distance = np.sqrt((rest_lat - user_lat) ** 2 + (rest_lon - user_lon) ** 2)
        result[f'distance_to_rest_{i}'] = distance

    result.to_csv(DATA_FILE_NAME, compression='gzip', index=False)

    print(f'File {DATA_FILE_NAME} has been created.')
else:
    print(f'File {DATA_FILE_NAME} already exist.')

df = pd.read_csv(DATA_FILE_NAME)
database = Database('flat_data', df)

beta_rating = true_parameters['beta_rating']
beta_price = true_parameters['beta_price']
beta_chinese = true_parameters['beta_chinese']
beta_japanese = true_parameters['beta_japanese']
beta_korean = true_parameters['beta_korean']
beta_indian = true_parameters['beta_indian']
beta_french = true_parameters['beta_french']
beta_mexican = true_parameters['beta_mexican']
beta_lebanese = true_parameters['beta_lebanese']
beta_ethiopian = true_parameters['beta_ethiopian']
beta_log_dist = true_parameters['beta_log_dist']

print('Building the utility function')
V = {
    i: (
        beta_rating * Variable(f'rating_{i}')
        + beta_price * Variable(f'price_{i}')
        + beta_chinese * Variable(f'category_Chinese_{i}')
        + beta_japanese * Variable(f'category_Japanese_{i}')
        + beta_korean * Variable(f'category_Korean_{i}')
        + beta_indian * Variable(f'category_Indian_{i}')
        + beta_french * Variable(f'category_French_{i}')
        + beta_mexican * Variable(f'category_Mexican_{i}')
        + beta_lebanese * Variable(f'category_Lebanese_{i}')
        + beta_ethiopian * Variable(f'category_Ethiopian_{i}')
        + beta_log_dist * log(Variable(f'distance_to_rest_{i}'))
    )
    for i in range(nbr_of_alternatives)
}

print('Probability models')

if os.path.exists(LOGIT_FILE_NAME):
    print(f'File {LOGIT_FILE_NAME} already exists')
else:
    print('Simulation: logit')
    logit_prob = {f'P[{i}]': loglogit(V, None, i) for i in range(nbr_of_alternatives)}

    logit_biosim = bio.BIOGEME(database, logit_prob)
    logit_biosim.modelName = 'logit'
    logit_prob_values = logit_biosim.simulate(theBetaValues={})
    logit_prob_values.to_csv(LOGIT_FILE_NAME, compression='gzip', index=False)
    print(f'File {LOGIT_FILE_NAME} has been created')


if os.path.exists(NESTED_FILE_NAME):
    print(f'File {NESTED_FILE_NAME} already exists')
else:
    print('Simulation: nested logit')

    mu_asian = true_parameters['mu_asian']
    nest_asian = OneNestForNestedLogit(
        nest_param=mu_asian, list_of_alternatives=asian, name='asian'
    )
    nests = NestsForNestedLogit(
        choice_set=all_alternatives,
        tuple_of_nests=(nest_asian,),
    )

    nested_prob = {
        f'P[{i}]': lognested(V, None, nests, i) for i in range(nbr_of_alternatives)
    }

    nested_biosim = bio.BIOGEME(database, nested_prob)
    nested_biosim.modelName = 'nested'
    nested_prob_values = nested_biosim.simulate(theBetaValues={})
    nested_prob_values.to_csv(NESTED_FILE_NAME, compression='gzip', index=False)
    print(f'File {NESTED_FILE_NAME} has been created')

if os.path.exists(CNL_FILE_NAME):
    print(f'File {CNL_FILE_NAME} already exists')
else:
    print('Simulation: cross-nested logit')

    mu_downtown = true_parameters['mu_downtown']
    downtown_alpha_dict = {i: 0.5 for i in asian_and_downtown} | {
        i: 1 for i in only_downtown
    }
    downtown_nest = OneNestForCrossNestedLogit(
        nest_param=mu_downtown, dict_of_alpha=downtown_alpha_dict, name='downtown'
    )

    # Asian
    mu_asian = true_parameters['mu_asian']
    asian_alpha_dict = {i: 0.5 for i in asian_and_downtown} | {
        i: 1.0 for i in only_asian
    }
    asian_nest = OneNestForCrossNestedLogit(
        nest_param=mu_asian, dict_of_alpha=asian_alpha_dict, name='asian'
    )

    cnl_nests = NestsForCrossNestedLogit(
        choice_set=all_alternatives,
        tuple_of_nests=(downtown_nest, asian_nest),
    )

    cnl_prob = {
        f'P[{i}]': logcnl(V, None, cnl_nests, i) for i in range(nbr_of_alternatives)
    }

    cnl_biosim = bio.BIOGEME(database, cnl_prob)
    cnl_biosim.modelName = 'cnl'
    cnl_prob_values = cnl_biosim.simulate(theBetaValues={})
    cnl_prob_values.to_csv(CNL_FILE_NAME, compression='gzip', index=False)
    print(f'File {CNL_FILE_NAME} has been created')


def draw_samples(row):
    return np.random.choice(range(len(row)), size=NUMBER_OF_DRAWS, p=row)


print('Generating synthetic choices')
if os.path.exists(LOGIT_DRAWS):
    print(f'File {LOGIT_DRAWS} already exists')
else:
    df = pd.read_csv(LOGIT_FILE_NAME)
    prob = np.exp(df)
    logit_samples = prob.apply(draw_samples, axis=1, result_type='expand')
    logit_samples.to_csv(LOGIT_DRAWS, compression='gzip', index=False)
    print(f'File {LOGIT_DRAWS} has been created')

if os.path.exists(NESTED_DRAWS):
    print(f'File {NESTED_DRAWS} already exists')
else:
    df = pd.read_csv(NESTED_FILE_NAME)
    prob = np.exp(df)
    nested_samples = prob.apply(draw_samples, axis=1, result_type='expand')
    nested_samples.to_csv(NESTED_DRAWS, compression='gzip', index=False)
    print(f'File {NESTED_DRAWS} has been created')

if os.path.exists(CNL_DRAWS):
    print(f'File {CNL_DRAWS} already exists')
else:
    df = pd.read_csv(CNL_FILE_NAME)
    prob = np.exp(df)
    cnl_samples = prob.apply(draw_samples, axis=1, result_type='expand')
    cnl_samples.to_csv(CNL_DRAWS, compression='gzip', index=False)
    print(f'File {CNL_DRAWS} has been created')

logit = pd.read_csv(LOGIT_DRAWS)
nested = pd.read_csv(NESTED_DRAWS)
cnl = pd.read_csv(CNL_DRAWS)

logit.columns = [f'logit_{col}' for col in logit.columns]
nested.columns = [f'nested_{col}' for col in nested.columns]
cnl.columns = [f'cnl_{col}' for col in cnl.columns]
result = pd.concat([obs_choice, logit, nested, cnl], axis=1)
result.to_csv(CHOICE_FILE_NAME, index=False)
print(f'File {CHOICE_FILE_NAME} has been created')
