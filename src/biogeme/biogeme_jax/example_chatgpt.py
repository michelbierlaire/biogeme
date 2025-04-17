import numpy as np
import pandas as pd
import biogeme_jax
import biogeme_jax.numpy as jnp
from biogeme_jax import jacfwd, jacrev, vmap
from scipy.optimize import minimize
import time

# ---------------- Step 1: Generate Large Dataset Efficiently ----------------
num_samples = 1_000_000  # Large dataset

# Define variable names explicitly
feature_names = [
    "income",
    "age",
    "distance",
    "time",
    "category_A",
    "category_B",
    "category_C",
]  # Example named features

# Generate categorical variable with 3 categories
categories = ["A", "B", "C"]
data = pd.DataFrame(
    {
        "income": np.random.uniform(20_000, 100_000, num_samples),
        "age": np.random.uniform(18, 80, num_samples),
        "distance": np.random.uniform(1, 50, num_samples),
        "time": np.random.uniform(5, 120, num_samples),
        "category": np.random.choice(categories, num_samples),
        "choice": np.random.randint(0, 2, num_samples),  # Target variable
    }
)

# One-hot encode the categorical variable
category_dummies = pd.get_dummies(data["category"], prefix="category")
data = pd.concat([data.drop(columns=["category"]), category_dummies], axis=1)

# Ensure all features are numeric
for col in data.columns:
    if data[col].dtype not in [np.float64, np.int64, bool]:
        raise ValueError(
            f"Column {col} has an invalid dtype: {data[col].dtype}. Expected numerical values."
        )

# Convert boolean columns to float (for JAX compatibility)
data = data.astype(float)

# Convert DataFrame to JAX array
X_jax = jnp.array(data.drop(columns=["choice"]).to_numpy())  # Only numeric features
Y = jnp.array(data["choice"].to_numpy())  # Convert choice variable to JAX array
feature_names = list(data.drop(columns=["choice"]).columns)  # Updated feature names


# ---------------- Step 2: Define External Logit Model with Named Features ----------------
def external_logit_model(params, row):
    """
    External logit model function applied to a single row, accessing columns by name.

    :param params: Model parameters
    :param row: JAX array representing a single row of data
    :return: Logit probability for the given row
    """
    beta0 = params[0]
    betas = dict(
        zip(feature_names, params[1:])
    )  # Create dictionary mapping feature names to parameters
    row_dict = dict(
        zip(feature_names, row)
    )  # Convert row array to dictionary for named access

    utility = (
        beta0
        + betas["income"]
        * jnp.log1p(row_dict["income"])  # Example: log-transformed income
        + betas["age"] * jnp.sqrt(row_dict["age"])  # Example: age transformation
        + betas["distance"]
        * jnp.exp(-row_dict["distance"] / 10)  # Example: exponential decay on distance
        + betas["time"]
        * (row_dict["time"] ** 2)
        / 1000  # Example: quadratic effect on time
        + sum(
            betas[col] * row_dict[col]
            for col in feature_names
            if col.startswith("category_")
        )  # Categorical effects
    )

    return 1 / (1 + jnp.exp(-utility))  # Sigmoid function


# Vectorize the function to apply it to all rows efficiently
vectorized_logit_model = vmap(
    lambda params, row: external_logit_model(params, row), in_axes=(None, 0)
)


# ---------------- Step 3: Define Log-Likelihood Function ----------------
def loglikelihood_and_derivatives(params, logit_model, X):
    """
    Computes the log-likelihood, gradient, and Hessian in a single pass.

    :param params: Model parameters
    :param logit_model: External logit model function (NumPy-based, nonlinear allowed)
    :param X: JAX array of features
    :return: Tuple (log-likelihood value, gradient, Hessian)
    """
    p = vectorized_logit_model(params, X)  # Apply model to all rows

    loglik_value = -jnp.sum(Y * jnp.log(p) + (1 - Y) * jnp.log(1 - p))

    def loglik_func(params):
        p = vectorized_logit_model(params, X)
        return -jnp.sum(Y * jnp.log(p) + (1 - Y) * jnp.log(1 - p))

    grad_value = jacrev(loglik_func)(params)  # Reverse-mode autodiff for gradient
    hessian_value = jacfwd(jacrev(loglik_func))(
        params
    )  # Forward-over-reverse Hessian computation

    return loglik_value, grad_value, hessian_value


# Compile with JIT for efficiency
loglikelihood_and_derivatives_jit = biogeme_jax.jit(
    loglikelihood_and_derivatives, static_argnums=(1,)
)

# Initialize parameters (one per named feature + intercept)
params_init = np.zeros(
    len(feature_names) + 1
)  # Includes beta0 and coefficients for named variables


# ---------------- Step 4: Optimize Using BFGS with Performance Profiling ----------------
def objective_function(params):
    start = time.time()
    loglik_value, _, _ = loglikelihood_and_derivatives_jit(
        params, external_logit_model, X_jax
    )
    end = time.time()
    print(f"Log-likelihood computation time: {end - start:.4f} seconds")
    return loglik_value


def objective_gradient(params):
    start = time.time()
    _, grad_value, _ = loglikelihood_and_derivatives_jit(
        params, external_logit_model, X_jax
    )
    end = time.time()
    print(f"Gradient computation time: {end - start:.4f} seconds")
    return grad_value


start_optimization = time.time()
result = minimize(
    objective_function, x0=params_init, jac=objective_gradient, method="BFGS"
)
end_optimization = time.time()
print(f"Total optimization time: {end_optimization - start_optimization:.4f} seconds")

# ---------------- Step 5: Print Results with Profiling ----------------
start_final_eval = time.time()
loglik_value, grad_value, hessian_value = loglikelihood_and_derivatives_jit(
    result.x, external_logit_model, X_jax
)
end_final_eval = time.time()
print(f"Final evaluation time: {end_final_eval - start_final_eval:.4f} seconds")

print("Optimization successful:", result.success)
print("Final log-likelihood:", loglik_value)
print("Estimated parameters:")
print(result.x)
