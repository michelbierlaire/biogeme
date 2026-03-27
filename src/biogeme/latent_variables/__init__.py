"""Fresh latent-variables package centered on a resolved semantic model.

Public workflow
---------------
1. Define pure specifications in :mod:`model_spec`.
2. Define an optional normalization plan in :mod:`normalization_plan`.
3. Validate and resolve the model with :func:`resolve_model`.
4. Choose an output:
   - live Biogeme expressions with :func:`build_biogeme_model`,
   - pedagogical runnable Python code with :func:`generate_python_code`,
   - a LaTeX scientific report with :func:`generate_latex_report`,
   - a static HTML report with :func:`generate_html_report`.
"""

from .biogeme_builder import BuiltBiogemeModel, build_biogeme_model
from .context import BuildContext, EstimationMode, PositivityMode
from .html_report import generate_html_report, generate_html_report, save_html_report
from .io import save_text
from .latex_report import (
    generate_latex_report,
    generate_latex_report,
    save_latex_report,
)
from .model_spec import (
    IndicatorMeasurementSpec,
    LatentVariable,
    LikertIndicator,
    LikertType,
    MeasurementConfiguration,
    MeasurementModel,
    PositiveParameterSpec,
    StructuralEquation,
)
from .normalization_plan import ConflictPolicy, Fixing, NormalizationPlan
from .normalization_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    ParameterRef,
    StructuralCoefficient,
    StructuralSigma,
    ThresholdDelta,
    ThresholdFirst,
)
from .python_generator import (
    generate_python_code,
    generate_python_code,
    save_python_code,
)
from .resolved import ResolvedModel
from .resolver import resolve_model
