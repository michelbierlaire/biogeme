# normalization/__init__.py
"""
Normalization package.

This package defines:
- semantic parameter references (`ParameterRef` and concrete targets),
- normalization plans (sets of explicit fixings),
- plan validation against a model specification.

It intentionally contains no Biogeme expression-building logic.

Michel Bierlaire
Wed Mar 04 2026
"""

from .parameter_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    ParameterRef,
    StructuralCoefficient,
    StructuralSigma,
    ThresholdFirst,
)
from .plan import ConflictPolicy, Fixing, NormalizationPlan
from .validation import Diagnostic, DiagnosticLevel, validate_plan
