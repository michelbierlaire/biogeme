from biogeme.latent_variables import (
    EstimationMode,
    LatentVariable,
    Normalization,
    OrderedMimic,
    StructuralEquation,
)

from config import Config
from latent_variables import (
    car_explanatory_variables,
    car_likert_indicators,
    car_name,
    env_name,
    environment_explanatory_variables,
    environment_likert_indicators,
)
from likert_indicators import likert_indicators, likert_types


def generate_mimic_model(config: Config):
    bayesian_estimation = config.estimation == "bayes"
    estimation_mode = (
        EstimationMode.BAYESIAN
        if bayesian_estimation
        else EstimationMode.MAXIMUM_LIKELIHOOD
    )

    mimic_model = OrderedMimic(
        estimation_mode=estimation_mode,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
    )

    car_lv = LatentVariable(
        name=car_name,
        structural_equation=StructuralEquation(
            name=car_name,
            explanatory_variables=car_explanatory_variables,
        ),
        indicators=car_likert_indicators,
        normalization=Normalization(indicator='Envir01', coefficient=-1),
    )
    mimic_model.add_latent_variable(car_lv)

    env_lv = LatentVariable(
        name=env_name,
        structural_equation=StructuralEquation(
            name=env_name,
            explanatory_variables=environment_explanatory_variables,
        ),
        indicators=environment_likert_indicators,
        normalization=Normalization(indicator='Envir02', coefficient=1),
    )
    mimic_model.add_latent_variable(env_lv)
    return mimic_model
