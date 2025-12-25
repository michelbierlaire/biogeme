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
mimic_file = 'saved_results/b02_mimic_ml.yaml'

mimic_results = EstimationResults.from_yaml_file(filename=mimic_file)
print(mimic_results.get_beta_values())
mimic_statistics = get_latex_general_statistics(estimation_results=mimic_results)
print(mimic_statistics)
parameter_names = {
    'likert_delta_0_log': r'$\ln(\delta^{\text{likert}}_0)$',
    'likert_delta_1_log': r'$\ln(\delta^{\text{likert}}_1)$',
    'cars_delta_0_log': r'$\ln(\delta^{\text{cars}}_0)$',
    'cars_delta_1_log': r'$\ln(\delta^{\text{cars}}_1)$',
}
mimic_parameters = get_latex_estimated_parameters(
    estimation_results=mimic_results,
    variance_covariance_type=EstimateVarianceCovariance.BHHH,
    renaming_parameters=parameter_names,
)

print(mimic_parameters)

# Choice model
print('=' * 20 + ' Choice only ' + '=' * 20)

choice_file = 'saved_results/b01_choice_only_ml.yaml'

choice_results = EstimationResults.from_yaml_file(filename=choice_file)

choice_statistics = get_latex_general_statistics(estimation_results=choice_results)

print(choice_statistics)

choice_parameters = get_latex_estimated_parameters(
    estimation_results=choice_results,
    variance_covariance_type=EstimateVarianceCovariance.BHHH,
)
print(choice_parameters)

# Choice model with latent variables: simultaneous estimation
print('=' * 20 + ' Choice with latent variables: simultaneous ' + '=' * 20)

simultaneous_file = 'saved_results/b03_hybrid_ml.yaml'

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
        'MIMIC': mimic_results,
        'Simultaneous': simultaneous_results,
    }
)
print(comparison)
