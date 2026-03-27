from __future__ import annotations

"""Build-time context shared by resolution and outputs."""

from dataclasses import dataclass
from enum import Enum

from .naming import DefaultNamingPolicy, NamingPolicy


class EstimationMode(str, Enum):
    MAXIMUM_LIKELIHOOD = 'maximum_likelihood'
    BAYESIAN = 'bayesian'


class PositivityMode(str, Enum):
    LOG_EXP = 'log_exp'
    LOWER_BOUND = 'lower_bound'


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Configuration influencing resolution and output generation.

    :param estimation_mode: Maximum likelihood or Bayesian mode.
    :param draw_type: Draw type used for latent variables.
    :param positivity_mode: Positive-parameter parameterization.
    :param naming: Naming policy.
    :param ordinal_eps: Lower clipping bound for ordinal probabilities.
    :param ordinal_enforce_order: Whether to enforce ordered cutpoints.
    """

    estimation_mode: EstimationMode
    draw_type: str
    positivity_mode: PositivityMode
    naming: NamingPolicy
    ordinal_eps: float = 1e-12
    ordinal_enforce_order: bool = True

    @staticmethod
    def default(estimation_mode: EstimationMode) -> 'BuildContext':
        draw_type = 'NORMAL_MLHS_ANTI' if estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD else 'Normal'
        positivity_mode = (
            PositivityMode.LOG_EXP if estimation_mode == EstimationMode.MAXIMUM_LIKELIHOOD else PositivityMode.LOWER_BOUND
        )
        return BuildContext(
            estimation_mode=estimation_mode,
            draw_type=draw_type,
            positivity_mode=positivity_mode,
            naming=DefaultNamingPolicy(),
        )
