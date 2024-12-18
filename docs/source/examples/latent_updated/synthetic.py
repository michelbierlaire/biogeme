import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from factor_analyzer import FactorAnalyzer

# Set random seed for reproducibility
# np.random.seed(42)

### Step 1: Create a dataset with known factor values

# Assume we have 3 factors and 5 observed variables
n_samples = 100  # Number of observations
n_factors = 3  # Number of latent factors
n_variables = 5  # Number of observed variables

# Generate random factor values (latent variables)
true_factors = np.random.randn(n_samples, n_factors)

### Step 2: Postulate a "true" factor loading matrix

# Create a true factor loading matrix (5 observed variables x 3 factors)
true_loadings = np.array(
    [
        [0.8, 0.5, 0.0],
        [0.6, 0.4, 0.3],
        [0.0, 0.6, 0.7],
        [0.0, 0.3, 0.8],
        [0.7, 0.0, 0.6],
    ]
)

# Generate synthetic observed data using the factor scores and factor loadings
# Add some Gaussian noise to simulate real-world data (unexplained variance)
error_variance = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Error terms for each variable
errors = np.random.randn(n_samples, n_variables) * np.sqrt(error_variance)
synthetic_data = np.dot(true_factors, true_loadings.T) + errors

### Step 3: Perform factor analysis and estimate loadings

# Normalize the data (just like you would do in a real factor analysis)
scaler = StandardScaler()
normalized_synthetic_data = scaler.fit_transform(synthetic_data)

# Perform factor analysis (without rotation to match the true loadings directly)
fa = FactorAnalyzer(n_factors=n_factors, rotation=None)
fa.fit(normalized_synthetic_data)

# Get the estimated factor loadings (no rotation)
estimated_loadings = fa.loadings_

### Step 4: Align estimated loadings with true loadings using Procrustes rotation

# Perform Procrustes rotation to align estimated loadings with true loadings
R, scale = orthogonal_procrustes(estimated_loadings, true_loadings)

# Apply the rotation to the estimated loadings
aligned_loadings = np.dot(estimated_loadings, R)

# Compare aligned loadings with the true loadings
print("True Factor Loadings:")
print(true_loadings)

print("\nAligned Estimated Factor Loadings (After Procrustes Rotation):")
print(aligned_loadings)

### Step 5: Generate another synthetic dataset based on new factor values

# Generate new factor values (for validation)
new_factors = np.random.randn(n_samples, n_factors)

# Generate new synthetic data using the new factor values
new_errors = np.random.randn(n_samples, n_variables) * np.sqrt(error_variance)
new_synthetic_data = np.dot(new_factors, true_loadings.T) + new_errors

# Normalize the new synthetic data using the same scaler
new_normalized_synthetic_data = scaler.transform(new_synthetic_data)

### Step 6: Apply factor reconstruction method to recover the factors

# Reconstruct the factors from the new synthetic data using the aligned loadings
# Use the equation: Z = X * L * (L^T * L)^{-1}
loadings_cov = np.dot(aligned_loadings.T, aligned_loadings)
reconstructed_factors = np.dot(
    new_normalized_synthetic_data, np.dot(aligned_loadings, np.linalg.inv(loadings_cov))
)

# Compare the original new factors with the reconstructed factors
print("\nOriginal New Factors (Latent Variables):")
print(new_factors[:5])  # Show the first 5 rows for comparison

print("\nReconstructed Factors (Latent Variables):")
print(reconstructed_factors[:5])  # Show the first 5 rows for comparison

relative_difference = np.abs((new_factors - reconstructed_factors) / new_factors)
print('\nRelative difference')
print(relative_difference.mean())
