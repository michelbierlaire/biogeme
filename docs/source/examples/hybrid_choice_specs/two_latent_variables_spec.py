"""
Latent variables (SPEC ONLY)

Pure specification: what depends on what.

Michel Bierlaire
:Wed Mar 04 2026, 13:48:57
"""

from biogeme.latent_variables import (
    LatentVariable,
    PositiveParameterSpec,
    StructuralEquation,
)

DEFAULT_SIGMA_START = 10.0

car_centric_attitude = LatentVariable(
    name="car_centric_attitude",
    structural_equation=StructuralEquation(
        name="car_centric_attitude",
        explanatory_variables=[
            "high_education",
            "top_manager",
            "age_30_less",
            "ScaledIncome",
            "car_oriented_parents",
            "high_education",
            "low_education",
            "male",
            "used_to_go_to_school_by_car",
        ],
    ),
    indicators={
        "Envir01",
        "Envir02",
        "Envir06",
        "Mobil03",
        "Mobil05",
        "Mobil08",
        "Mobil09",
        "Mobil10",
        "LifSty07",
    },
    structural_sigma=PositiveParameterSpec(start=DEFAULT_SIGMA_START),
)

environmental_attitude = LatentVariable(
    name="environmental_attitude",
    structural_equation=StructuralEquation(
        name="environmental_attitude",
        explanatory_variables=[
            "childSuburb",
            "ScaledIncome",
            "city_center_as_kid",
            "artisans",
            "high_education",
            "low_education",
            "single",
        ],
    ),
    indicators={
        "Envir01",
        "Envir02",
        "Envir03",
        "Envir04",
        "Envir05",
        "Envir06",
        "Mobil12",
        "LifSty01",
    },
    structural_sigma=PositiveParameterSpec(start=DEFAULT_SIGMA_START),
)

latent_variables = [car_centric_attitude, environmental_attitude]
