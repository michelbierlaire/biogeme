"""
Build context.

The BuildContext centralizes build-time concerns:
- estimation-mode dependent defaults (ML vs Bayesian),
- draw type,
- factories for positive parameters and sigmas,
- naming policy used by builders,
- OrderedProbit numerical settings.

Michel Bierlaire
Thu Mar 05 2026, 11:34:55

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .naming import DefaultNamingPolicy, NamingPolicy
from .positive_parameter import (
    PositiveParameterFactory,
    SigmaFactory,
    make_positive_parameter_factory,
    make_sigma_factory,
)


class EstimationMode(str, Enum):
    """Estimation mode controlling builder defaults."""

    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN = "bayesian"


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Configuration passed to builders."""

    mode: EstimationMode
    draw_type: str
    sigma_factory: SigmaFactory
    positive_parameter_factory: PositiveParameterFactory
    naming: NamingPolicy

    ordinal_eps: float = 1e-12
    ordinal_enforce_order: bool = True

    @staticmethod
    def from_mode(
        mode: EstimationMode,
        *,
        draw_type: str | None = None,
        use_log_positive_parameters: bool | None = None,
        naming: NamingPolicy | None = None,
        ordinal_eps: float = 1e-12,
        ordinal_enforce_order: bool = True,
    ) -> "BuildContext":
        """Factory with sensible defaults for ML vs Bayesian."""

        if draw_type is None:
            draw_type = (
                "NORMAL_MLHS_ANTI"
                if mode == EstimationMode.MAXIMUM_LIKELIHOOD
                else "Normal"
            )

        if use_log_positive_parameters is None:
            use_log_positive_parameters = mode == EstimationMode.MAXIMUM_LIKELIHOOD

        if naming is None:
            naming = DefaultNamingPolicy()

        sigma_factory = make_sigma_factory(use_log=use_log_positive_parameters)
        positive_factory = make_positive_parameter_factory(
            use_log=use_log_positive_parameters
        )

        return BuildContext(
            mode=mode,
            draw_type=draw_type,
            sigma_factory=sigma_factory,
            positive_parameter_factory=positive_factory,
            naming=naming,
            ordinal_eps=ordinal_eps,
            ordinal_enforce_order=ordinal_enforce_order,
        )
