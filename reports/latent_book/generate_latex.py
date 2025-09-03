"""This script generates LaTeX code from estimation results"""

from biogeme.results_processing import (
    EstimateVarianceCovariance,
    EstimationResults,
    compare_parameters,
    get_latex_estimated_parameters,
    get_latex_general_statistics,
)

# MIMIC model
print('=' * 20 + ' MIMIC ' + '=' * 20)
mimic_file = 'saved_results/b01_mimic.yaml'

mimic_results = EstimationResults.from_yaml_file(filename=mimic_file)

mimic_statistics = get_latex_general_statistics(estimation_results=mimic_results)
print(mimic_statistics)
parameter_names = {
    r'delta_1': '$\\delta_1$',
    r'delta_2': r'$\delta_2$',
}
mimic_parameters = get_latex_estimated_parameters(
    estimation_results=mimic_results,
    variance_covariance_type=EstimateVarianceCovariance.ROBUST,
    renaming_parameters=parameter_names,
)

print(mimic_parameters)

# Choice model
print('=' * 20 + ' Choice only ' + '=' * 20)

choice_file = 'saved_results/b02_choice_only.yaml'

choice_results = EstimationResults.from_yaml_file(filename=choice_file)

choice_statistics = get_latex_general_statistics(estimation_results=choice_results)

print(choice_statistics)

choice_parameters = get_latex_estimated_parameters(
    estimation_results=choice_results,
    variance_covariance_type=EstimateVarianceCovariance.ROBUST,
)
print(choice_parameters)

# Choice model with latent variables: sequential estimation
print('=' * 20 + ' Choice with latent variables: sequential ' + '=' * 20)

sequential_file = 'saved_results/b03_sequential.yaml'

sequential_results = EstimationResults.from_yaml_file(filename=sequential_file)

sequential_statistics = get_latex_general_statistics(
    estimation_results=sequential_results
)

print(sequential_statistics)

sequential_parameters = get_latex_estimated_parameters(
    estimation_results=sequential_results,
    variance_covariance_type=EstimateVarianceCovariance.ROBUST,
)
print(sequential_parameters)

# Choice model with latent variables: simultaneous estimation
print('=' * 20 + ' Choice with latent variables: simultaneous ' + '=' * 20)

simultaneous_file = 'saved_results/b03_simultaneous.yaml'

simultaneous_results = EstimationResults.from_yaml_file(filename=simultaneous_file)

simultaneous_statistics = get_latex_general_statistics(
    estimation_results=simultaneous_results
)

print(simultaneous_statistics)

simultaneous_parameters = get_latex_estimated_parameters(
    estimation_results=simultaneous_results,
    variance_covariance_type=EstimateVarianceCovariance.BHHH,
)
print(simultaneous_parameters)

### Compare parameters
comparison = compare_parameters(
    {
        'Choice only': choice_results,
        'Sequential': sequential_results,
        'Simultaneous': simultaneous_results,
    }
)
print(comparison)
