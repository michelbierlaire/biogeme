""" Contains the constants used in various files

"""

N_ALT = 10
INSTANCES = 10

partitions = ('pure', 'asian', 'downtown', 'both')

# True parameters of the choice model for the synthetic choice
scale = 2.0
true_parameters = {
    'beta_rating': 1.5 * scale,
    'beta_price': -0.8 * scale,
    'beta_chinese': 1.5 * scale,
    'beta_japanese': 2.5 * scale,
    'beta_korean': 1.5 * scale,
    'beta_indian': 2 * scale,
    'beta_french': 1.5 * scale,
    'beta_mexican': 2.5 * scale,
    'beta_lebanese': 1.5 * scale,
    'beta_ethiopian': 1 * scale,
    'beta_log_dist': -1.2 * scale,
    'true_mu_asian': 2.0,
    'true_mu_downtown': 2.0,
}
