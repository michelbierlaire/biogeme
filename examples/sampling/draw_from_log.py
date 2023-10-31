import numpy as np


def choice_from_logp(logp):
    the_sum = sum(np.exp(logp))
    if not np.isclose(the_sum, 1):
        raise ValueError(f'The probabilities do NOT sum up to one: {the_sum}.')

    # Subtracting the max value for numerical stability
    adjusted_probs = np.exp(logp - np.max(logp))

    # Normalize the probabilities
    normalized_probs = adjusted_probs / adjusted_probs.sum()

    return np.random.choice(len(logp), p=normalized_probs)


# Example usage:

probs = np.array([0.1, 0.2, 0.7])
log_probs = np.log(probs)

# Simulate 10000 draws
num_draws = 10000
samples = [choice_from_logp(log_probs) for _ in range(num_draws)]

# Calculate empirical probabilities
empirical_probs = np.bincount(samples, minlength=len(log_probs)) / num_draws

print("Target Probabilities:", np.exp(log_probs))
print("Empirical Probabilities:", empirical_probs)
