from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from biogeme.constants import LOG_LIKE
from biogeme.database import ContiguousPanelMap, build_contiguous_panel_map

from .base_expressions import Expression, ExpressionOrNumeric
from .bayesian import Dimension, PymcModelBuilderType
from .individual_draws import individual_draws

logger = logging.getLogger(__name__)


class PanelLogLikelihood(Expression):
    """
    Aggregate per-observation **log-probabilities** into per-individual log-likelihoods.

    This expression assumes its child evaluates, for a given dataframe, to a 1-D
    tensor of shape ``(obs,)`` containing the **log-probability** of each
    observation. It then sums these log-probabilities within each individual
    (panel) and returns a 1-D tensor of shape ``(indiv,)``.

    Notes
    -----
    - Intended for Bayesian estimation with PyMC, which operates in log-space.
    - The panel/individual id column name is taken from ``self.panel_index_name``
      if available, otherwise it defaults to ``'ID'``.
    - A coord named :data:`Dimension.INDIVIDUALS` is created on the active PyMC
      model if it does not yet exist, to label the individuals' axis.

    :param child: Expression that returns per-observation **log-probabilities**
        when evaluated by the PyMC builder.
    """

    def __init__(self, child: ExpressionOrNumeric) -> None:
        super().__init__()
        if isinstance(child, Expression):
            self.child: Expression = child
        else:
            # Numeric constants are allowed by the Expression API; they will
            # typically be wrapped upstream. We keep type hints explicit.
            self.child = child  # type: ignore[assignment]

        individual_draws(expr=self.child)
        self.children.append(self.child)
        self.panel_id = None

    def deep_flat_copy(self) -> "PanelLogLikelihood":
        """
        Return a deep/flat copy of the expression.

        :return: A structurally independent copy whose child is a deep/flat copy.
        """
        copy_child: Expression = self.child.deep_flat_copy()
        return PanelLogLikelihood(child=copy_child)

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"PanelLogLikelihood({self.child})"

    def __repr__(self) -> str:
        """Unambiguous representation for debugging."""
        return f"PanelLogLikelihood({repr(self.child)})"

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Build a PyMC evaluation closure that returns per-individual log-likelihoods,
        using a precomputed ContiguousPanelMap to aggregate rows belonging to the
        same individual via cumulative-sum + index differences.
        """
        child_builder: PymcModelBuilderType = (
            self.child.recursive_construct_pymc_model_builder()
        )

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            # Per-observation log-likelihood (shape: (N_obs,))
            logp_obs: pt.TensorVariable = child_builder(dataframe=dataframe)
            pm.Deterministic(f"{LOG_LIKE}_obs", logp_obs, dims=Dimension.OBS)

            # Build the panel map; validate contiguity
            panel_map: ContiguousPanelMap = build_contiguous_panel_map(
                dataframe, panel_column=self.panel_id
            )
            n_indiv = int(panel_map.unique_ids.size)

            # Segment-sum via cumsum + differences at indptr
            # indptr is length K+1 with [start_0, start_1, ..., N]
            indptr_pt = pt.as_tensor_variable(
                panel_map.indptr.astype(np.int64)
            )  # (K+1,)

            # cumsum over observations; pad a leading zero
            s = pt.cumsum(logp_obs)  # (N_obs,)
            # make a scalar zero with the same dtype as logp_obs
            zero = pt.zeros_like(logp_obs[:1]).sum()  # scalar 0.0, dtype matches
            s_pad = pt.concatenate([zero[None], s])  # (N_obs + 1,)

            # ll_indiv[k] = s_pad[indptr[k+1]] - s_pad[indptr[k]]
            ll_indiv = s_pad[indptr_pt[1:]] - s_pad[indptr_pt[:-1]]  # (K,)

            pm.Deterministic(f"{LOG_LIKE}_panel", ll_indiv, dims=Dimension.INDIVIDUALS)
            return ll_indiv

        return builder
