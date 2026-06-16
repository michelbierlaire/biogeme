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
    name='car_centric_attitude',
    structural_equation=StructuralEquation(
        name='car_centric_attitude',
        intercept=True,
        explanatory_variables=[
            'top_manager',
            'car_oriented_parents',
            'high_education',
            'low_education',
            'used_to_go_to_school_by_car',
        ],
    ),
    indicators={
        'Envir01',
        'Envir02',
        'Envir06',
        'Mobil03',
        'Mobil05',
        'Mobil08',
        'Mobil09',
        'Mobil10',
        'LifSty07',
    },
    structural_sigma=PositiveParameterSpec(start=DEFAULT_SIGMA_START),
)

latent_variables = [car_centric_attitude]
