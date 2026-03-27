from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum

import pytest
from biogeme.latent_variables.model_spec import (
    IndicatorMeasurementSpec,
    LatentVariable,
    LikertIndicator,
    LikertType,
    MeasurementConfiguration,
    MeasurementModel,
    PositiveParameterSpec,
    StructuralEquation,
)


def test_measurement_model_is_str_enum_with_expected_values() -> None:
    assert issubclass(MeasurementModel, str)
    assert issubclass(MeasurementModel, Enum)

    assert MeasurementModel.GAUSSIAN.value == 'gaussian'
    assert MeasurementModel.ORDERED_PROBIT.value == 'ordered_probit'
    assert MeasurementModel.ORDERED_LOGIT.value == 'ordered_logit'


def test_measurement_model_can_be_constructed_from_value() -> None:
    assert MeasurementModel('gaussian') is MeasurementModel.GAUSSIAN
    assert MeasurementModel('ordered_probit') is MeasurementModel.ORDERED_PROBIT
    assert MeasurementModel('ordered_logit') is MeasurementModel.ORDERED_LOGIT


def test_positive_parameter_spec_defaults() -> None:
    spec = PositiveParameterSpec()

    assert spec.start is None
    assert spec.lower_bound == 0.0


def test_positive_parameter_spec_custom_values() -> None:
    spec = PositiveParameterSpec(start=1.25, lower_bound=0.5)

    assert spec.start == 1.25
    assert spec.lower_bound == 0.5


def test_positive_parameter_spec_accepts_none_lower_bound() -> None:
    spec = PositiveParameterSpec(start=2.0, lower_bound=None)

    assert spec.start == 2.0
    assert spec.lower_bound is None


def test_positive_parameter_spec_is_frozen_and_slotted() -> None:
    spec = PositiveParameterSpec()

    with pytest.raises(FrozenInstanceError):
        spec.start = 3.0  # type: ignore[misc]

    assert hasattr(PositiveParameterSpec, '__slots__')
    assert '__dict__' not in dir(spec)


def test_structural_equation_fields_are_preserved() -> None:
    explanatory_variables = ('income', 'age')
    equation = StructuralEquation(
        name='LV_TRAVEL',
        explanatory_variables=explanatory_variables,
    )

    assert equation.name == 'LV_TRAVEL'
    assert equation.explanatory_variables is explanatory_variables


def test_structural_equation_accepts_any_iterable_for_explanatory_variables() -> None:
    explanatory_variables = iter(['x1', 'x2'])
    equation = StructuralEquation(
        name='LV_A',
        explanatory_variables=explanatory_variables,
    )

    assert equation.explanatory_variables is explanatory_variables
    assert list(equation.explanatory_variables) == ['x1', 'x2']


def test_structural_equation_is_frozen_and_slotted() -> None:
    equation = StructuralEquation(name='LV_X', explanatory_variables=[])

    with pytest.raises(FrozenInstanceError):
        equation.name = 'LV_Y'  # type: ignore[misc]

    assert hasattr(StructuralEquation, '__slots__')
    assert '__dict__' not in dir(equation)


def test_latent_variable_defaults_structural_sigma_to_none() -> None:
    equation = StructuralEquation(name='LV1', explanatory_variables=['x'])
    indicators = ('ind1', 'ind2')

    latent = LatentVariable(
        name='LV1',
        structural_equation=equation,
        indicators=indicators,
    )

    assert latent.name == 'LV1'
    assert latent.structural_equation is equation
    assert latent.indicators is indicators
    assert latent.structural_sigma is None


def test_latent_variable_accepts_custom_structural_sigma() -> None:
    equation = StructuralEquation(name='LV2', explanatory_variables=['x1', 'x2'])
    sigma = PositiveParameterSpec(start=1.0, lower_bound=0.1)

    latent = LatentVariable(
        name='LV2',
        structural_equation=equation,
        indicators=['i1'],
        structural_sigma=sigma,
    )

    assert latent.name == 'LV2'
    assert latent.structural_equation is equation
    assert list(latent.indicators) == ['i1']
    assert latent.structural_sigma is sigma


def test_latent_variable_is_frozen_and_slotted() -> None:
    latent = LatentVariable(
        name='LV3',
        structural_equation=StructuralEquation(name='LV3', explanatory_variables=[]),
        indicators=[],
    )

    with pytest.raises(FrozenInstanceError):
        latent.name = 'LV4'  # type: ignore[misc]

    assert hasattr(LatentVariable, '__slots__')
    assert '__dict__' not in dir(latent)


def test_likert_type_fields_are_preserved() -> None:
    likert_type = LikertType(
        type_name='agreement_5',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[3],
    )

    assert likert_type.type_name == 'agreement_5'
    assert likert_type.symmetric is True
    assert likert_type.categories == [1, 2, 3, 4, 5]
    assert likert_type.neutral_labels == [3]


def test_likert_type_accepts_empty_neutral_labels() -> None:
    likert_type = LikertType(
        type_name='binary',
        symmetric=False,
        categories=[0, 1],
        neutral_labels=[],
    )

    assert likert_type.type_name == 'binary'
    assert likert_type.symmetric is False
    assert likert_type.categories == [0, 1]
    assert likert_type.neutral_labels == []


def test_likert_type_is_frozen_and_slotted() -> None:
    likert_type = LikertType(
        type_name='scale',
        symmetric=False,
        categories=[1, 2],
        neutral_labels=[],
    )

    with pytest.raises(FrozenInstanceError):
        likert_type.type_name = 'other'  # type: ignore[misc]

    assert hasattr(LikertType, '__slots__')
    assert '__dict__' not in dir(likert_type)


def test_likert_indicator_fields_are_preserved() -> None:
    indicator = LikertIndicator(
        name='ind_safety',
        statement='Public transport is safe.',
        type_name='agreement_5',
    )

    assert indicator.name == 'ind_safety'
    assert indicator.statement == 'Public transport is safe.'
    assert indicator.type_name == 'agreement_5'


def test_likert_indicator_is_frozen_and_slotted() -> None:
    indicator = LikertIndicator(
        name='ind1',
        statement='Statement',
        type_name='type1',
    )

    with pytest.raises(FrozenInstanceError):
        indicator.statement = 'Other statement'  # type: ignore[misc]

    assert hasattr(LikertIndicator, '__slots__')
    assert '__dict__' not in dir(indicator)


def test_indicator_measurement_spec_defaults_measurement_sigma_to_none() -> None:
    spec = IndicatorMeasurementSpec(
        indicator_name='ind_cost',
        measurement_model=MeasurementModel.GAUSSIAN,
    )

    assert spec.indicator_name == 'ind_cost'
    assert spec.measurement_model is MeasurementModel.GAUSSIAN
    assert spec.measurement_sigma is None


def test_indicator_measurement_spec_accepts_custom_measurement_sigma() -> None:
    sigma = PositiveParameterSpec(start=0.8, lower_bound=0.01)

    spec = IndicatorMeasurementSpec(
        indicator_name='ind_time',
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        measurement_sigma=sigma,
    )

    assert spec.indicator_name == 'ind_time'
    assert spec.measurement_model is MeasurementModel.ORDERED_LOGIT
    assert spec.measurement_sigma is sigma


@pytest.mark.parametrize(
    ('model',),
    [
        (MeasurementModel.GAUSSIAN,),
        (MeasurementModel.ORDERED_PROBIT,),
        (MeasurementModel.ORDERED_LOGIT,),
    ],
)
def test_indicator_measurement_spec_supports_all_measurement_models(
    model: MeasurementModel,
) -> None:
    spec = IndicatorMeasurementSpec(
        indicator_name='ind',
        measurement_model=model,
    )

    assert spec.measurement_model is model


def test_indicator_measurement_spec_is_frozen_and_slotted() -> None:
    spec = IndicatorMeasurementSpec(
        indicator_name='ind',
        measurement_model=MeasurementModel.GAUSSIAN,
    )

    with pytest.raises(FrozenInstanceError):
        spec.indicator_name = 'other'  # type: ignore[misc]

    assert hasattr(IndicatorMeasurementSpec, '__slots__')
    assert '__dict__' not in dir(spec)


def test_measurement_configuration_preserves_iterable_reference() -> None:
    specifications = (
        IndicatorMeasurementSpec(
            indicator_name='ind1',
            measurement_model=MeasurementModel.GAUSSIAN,
        ),
        IndicatorMeasurementSpec(
            indicator_name='ind2',
            measurement_model=MeasurementModel.ORDERED_PROBIT,
        ),
    )

    config = MeasurementConfiguration(specifications=specifications)

    assert config.specifications is specifications


def test_measurement_configuration_accepts_generator_iterable() -> None:
    def make_specs():
        yield IndicatorMeasurementSpec(
            indicator_name='ind1',
            measurement_model=MeasurementModel.GAUSSIAN,
        )
        yield IndicatorMeasurementSpec(
            indicator_name='ind2',
            measurement_model=MeasurementModel.ORDERED_LOGIT,
        )

    specs = make_specs()
    config = MeasurementConfiguration(specifications=specs)

    assert config.specifications is specs
    collected = list(config.specifications)
    assert [spec.indicator_name for spec in collected] == ['ind1', 'ind2']
    assert [spec.measurement_model for spec in collected] == [
        MeasurementModel.GAUSSIAN,
        MeasurementModel.ORDERED_LOGIT,
    ]


def test_measurement_configuration_is_frozen_and_slotted() -> None:
    config = MeasurementConfiguration(specifications=[])

    with pytest.raises(FrozenInstanceError):
        config.specifications = []  # type: ignore[misc]

    assert hasattr(MeasurementConfiguration, '__slots__')
    assert '__dict__' not in dir(config)


def test_dataclass_equality_for_spec_objects() -> None:
    assert PositiveParameterSpec(start=1.0, lower_bound=0.0) == PositiveParameterSpec(
        start=1.0,
        lower_bound=0.0,
    )
    assert StructuralEquation(
        name='LV',
        explanatory_variables=['x1'],
    ) == StructuralEquation(
        name='LV',
        explanatory_variables=['x1'],
    )
    assert LatentVariable(
        name='LV',
        structural_equation=StructuralEquation(name='LV', explanatory_variables=[]),
        indicators=['i1'],
        structural_sigma=PositiveParameterSpec(start=1.0, lower_bound=0.0),
    ) == LatentVariable(
        name='LV',
        structural_equation=StructuralEquation(name='LV', explanatory_variables=[]),
        indicators=['i1'],
        structural_sigma=PositiveParameterSpec(start=1.0, lower_bound=0.0),
    )
    assert LikertType(
        type_name='t',
        symmetric=True,
        categories=[1, 2, 3],
        neutral_labels=[2],
    ) == LikertType(
        type_name='t',
        symmetric=True,
        categories=[1, 2, 3],
        neutral_labels=[2],
    )
    assert LikertIndicator(
        name='ind',
        statement='A statement',
        type_name='t',
    ) == LikertIndicator(
        name='ind',
        statement='A statement',
        type_name='t',
    )
    assert IndicatorMeasurementSpec(
        indicator_name='ind',
        measurement_model=MeasurementModel.GAUSSIAN,
        measurement_sigma=PositiveParameterSpec(start=1.0, lower_bound=0.0),
    ) == IndicatorMeasurementSpec(
        indicator_name='ind',
        measurement_model=MeasurementModel.GAUSSIAN,
        measurement_sigma=PositiveParameterSpec(start=1.0, lower_bound=0.0),
    )
    assert MeasurementConfiguration(
        specifications=[
            IndicatorMeasurementSpec(
                indicator_name='ind',
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        ]
    ) == MeasurementConfiguration(
        specifications=[
            IndicatorMeasurementSpec(
                indicator_name='ind',
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        ]
    )
