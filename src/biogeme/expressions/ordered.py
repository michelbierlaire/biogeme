from __future__ import annotations

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pytensor.tensor.math import softplus as pt_softplus

from .base_expressions import Expression
from .bayesian import PymcModelBuilderType
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType
from ..exceptions import BiogemeError


class OrderedBase(Expression):
    """
    Base class for ordered-response models (logit and probit).

    This class implements the common logic for ordered discrete-choice models.
    Given a latent variable :math:`\\eta_n` and ordered cutpoints
    :math:`\\tau_1 < \\tau_2 < \\dots < \\tau_{K-1}`, the probability of observing
    category :math:`y_k` is

    .. math::

        P(y = y_k \\mid \\eta, \\tau)
        = \\mathrm{CDF}(\\tau_k - \\eta) - \\mathrm{CDF}(\\tau_{k-1} - \\eta),

    where :math:`\\tau_0 = -\\infty` and :math:`\\tau_K = +\\infty`.

    Subclasses must implement the appropriate cumulative distribution
    function (CDF), either logistic (logit) or Gaussian (probit).

    This class returns per-observation **probabilities**. To obtain per-observation
    log-likelihoods, use the corresponding log-variants (e.g., :class:`OrderedLogLogit`).

    :param eta: Expression defining the latent variable :math:`\\eta_n`.
    :param cutpoints: List of expressions defining the cutpoints (length K-1).
    :param y: Expression defining the observed categorical response.
    :param categories: Ordered list of category labels (e.g. ``[1, 2, 3, 4, 5]``).
        If ``None``, defaults to ``[0, 1, ..., K-1]``.
    :param enforce_order: If ``True``, ensures that cutpoints are monotonically
        increasing, using a softplus transform (JAX) or sorting (PyTensor).
    :param eps: Lower bound for probabilities to avoid numerical issues.
    :param neutral_labels: Labels that may appear in the data and must be
        treated as “always valid”; their contribution is probability 1.
        Useful to avoid crashes on placeholder/missing/special codes.

    **Examples**

    Ordered Logit with 5 Likert levels (valid labels 1..5) and two neutral codes 98/99:

    .. code-block:: python

        income = Variable('Income')
        age = Variable('Age')
        beta_income = Beta('beta_income', 0, None, None, 0)
        beta_age = Beta('beta_age', 0, None, None, 0)

        eta = beta_income * income + beta_age * age

        # Four thresholds for five ordered responses
        tau1 = Beta('tau1', -1, None, None, 0)
        tau2 = Beta('tau2',  0, None, None, 0)
        tau3 = Beta('tau3',  1, None, None, 0)
        tau4 = Beta('tau4',  2, None, None, 0)

        y = Variable('Satisfaction')  # coded as 1, 2, 3, 4, 5, 98, 99

        model = OrderedLogit(
            eta=eta,
            cutpoints=[tau1, tau2, tau3, tau4],
            y=y,
            categories=[1, 2, 3, 4, 5],
            neutral_labels=[98, 99],
        )

        prob_vec = model.recursive_construct_pymc_model_builder()

    Ordered Probit with the same structure:

    .. code-block:: python

        model = OrderedProbit(
            eta=eta,
            cutpoints=[tau1, tau2, tau3, tau4],
            y=y,
            categories=[1, 2, 3, 4, 5],
            neutral_labels=[98, 99],
        )
    """

    def __init__(
        self,
        eta: Expression,
        cutpoints: list[Expression],
        y: Expression,
        categories: list[float] | tuple[float, ...] | None = None,
        neutral_labels: list[float] | tuple[float, ...] | None = None,
        enforce_order: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eta = validate_and_convert(eta)
        self.cutpoints = [validate_and_convert(c) for c in cutpoints]
        self.y = validate_and_convert(y)
        self.enforce_order = bool(enforce_order)
        self.eps = float(eps)

        # Determine K and validate 'categories'
        K = len(self.cutpoints) + 1
        if categories is None:
            self.categories = np.arange(K, dtype=np.int32)
        else:
            cats = np.asarray(categories)
            if cats.ndim != 1 or cats.size != K:
                raise BiogemeError(
                    f"'categories' must be a 1-D sequence of length K={K}; got shape {cats.shape}."
                )
            self.categories = cats.astype(np.float64)

        # Neutral/skip labels
        if neutral_labels is None or len(neutral_labels) == 0:
            self.neutral_labels = np.asarray([], dtype=np.float64)
            self._has_neutrals = False
        else:
            self.neutral_labels = np.asarray(neutral_labels, dtype=np.float64)
            self._has_neutrals = True

        # Register children
        self.children += [self.eta, self.y] + self.cutpoints

    # --------------------------------------------------------------------------
    #                                JAX builder
    # --------------------------------------------------------------------------
    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Build a JAX-compatible function returning per-observation probabilities.

        :param numerically_safe: Whether to use numerically stable operators.
        :return: A callable taking model parameters and one observation, and returning
            the probability of the observed category.
        """
        import jax
        import jax.numpy as jnp

        Km1 = len(self.cutpoints)
        K = Km1 + 1
        eps = self.eps
        cats = jnp.asarray(self.categories)
        neutrals = jnp.asarray(self.neutral_labels)

        def get_val(expr, θ, row, draws, rvars):
            fn = expr.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
            return fn(θ, row, draws, rvars)

        def order_cuts(raw_tau: jnp.ndarray) -> jnp.ndarray:
            """Ensure monotonic cutpoints using softplus increments (differentiable)."""
            if (not self.enforce_order) or (raw_tau.size == 0):
                return raw_tau
            tau0 = raw_tau[0]
            deltas = jax.nn.softplus(jnp.diff(raw_tau))
            return jnp.concatenate([jnp.array([tau0]), tau0 + jnp.cumsum(deltas)])

        CDF = self._cdf_jax

        def the_jax(θ, one_row, draws, rvars):
            # Scalars for a single observation 'one_row'
            eta = get_val(self.eta, θ, one_row, draws, rvars)
            raw = jnp.array(
                [get_val(c, θ, one_row, draws, rvars) for c in self.cutpoints]
            )
            tau = order_cuts(raw)

            # Observed label and matches
            y_val = get_val(self.y, θ, one_row, draws, rvars)
            match_cat = cats == y_val
            has_cat = jnp.any(match_cat)
            kpos = jnp.argmax(
                match_cat
            )  # 0..K-1 (undefined if has_cat=False, masked later)

            has_neutral = jnp.logical_and(neutrals.size > 0, jnp.any(neutrals == y_val))

            # Category probabilities via CDF differences
            if K == 1:
                probs = jnp.array([1.0])
            else:
                p0 = CDF(tau[0] - eta)
                mids = [
                    CDF(tau[k] - eta) - CDF(tau[k - 1] - eta) for k in range(1, Km1)
                ]
                pK = 1.0 - CDF(tau[-1] - eta)
                probs = jnp.concatenate(
                    [jnp.array([p0]), *(jnp.array([m]) for m in mids), jnp.array([pK])]
                )

            probs = jnp.clip(probs, eps, 1.0 - eps)
            p_valid = probs[kpos]

            p = jnp.where(has_neutral, 1.0, jnp.where(has_cat, p_valid, eps))
            return p

        return the_jax

    # --------------------------------------------------------------------------
    #                              PyMC / PyTensor builder
    # --------------------------------------------------------------------------
    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Build a PyTensor-compatible function returning per-observation probabilities.

        :return: Callable taking a pandas DataFrame and returning a PyTensor
            variable of probabilities of the observed categories.
        """
        K = len(self.cutpoints) + 1
        eps = pt.as_tensor_variable(self.eps)
        cats_const = pt.constant(np.asarray(self.categories))
        neutrals_const = pt.constant(np.asarray(self.neutral_labels))

        eta_b = self.eta.recursive_construct_pymc_model_builder()
        cut_b = [c.recursive_construct_pymc_model_builder() for c in self.cutpoints]
        y_b = self.y.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            eta = eta_b(dataframe)  # (N,)

            # Raw cutpoints stacked: (N, K-1) or broadcasted to that
            if len(cut_b):
                raw = pt.stack(
                    [cb(dataframe) for cb in cut_b], axis=1
                )  # (N, K-1) or (K-1,)
                # Ensure 2D shape (N, K-1) for downstream ops
                if raw.ndim == 1:
                    raw = pt.shape_padaxis(raw, 0) + pt.zeros_like(eta)[:, None]
                if self.enforce_order:
                    # Monotone cutpoints via softplus increments (same logic as JAX):
                    # tau0 = raw[:, 0]; deltas = softplus(diff(raw)); tau = concat([tau0, tau0 + cumsum(deltas)])
                    tau0 = raw[:, 0:1]
                    deltas = pt_softplus(raw[:, 1:] - raw[:, :-1])
                    tau = pt.concatenate(
                        [tau0, tau0 + pt.cumsum(deltas, axis=1)], axis=1
                    )
                else:
                    tau = raw
            else:
                tau = pt.zeros((eta.shape[0], 0), dtype=eta.dtype)  # K == 1

            CDF = self._cdf_pt

            # Probabilities matrix (N, K)
            if K == 1:
                probs = pt.ones((eta.shape[0], 1), dtype=eta.dtype)
            else:
                c = tau
                p0 = CDF(c[:, 0] - eta)  # (N,)
                mids = [
                    CDF(c[:, k] - eta) - CDF(c[:, k - 1] - eta) for k in range(1, K - 1)
                ]
                pK = 1.0 - CDF(c[:, -1] - eta)
                probs = pt.stack([p0, *mids, pK], axis=1)  # (N, K)

            probs = pt.clip(probs, eps, 1.0 - eps)

            # Map observed labels to positions 0..K-1 using provided categories
            y_val = y_b(dataframe)  # (N,)
            matches = pt.eq(y_val[:, None], cats_const[None, :])  # (N, K)
            idx = pt.argmax(matches, axis=1)  # (N,)
            any_match = pt.any(matches, axis=1)  # (N,)

            rows = pt.arange(y_val.shape[0])
            chosen = probs[rows, idx]  # (N,)
            p_valid = chosen

            # Neutral labels detection
            if self._has_neutrals:
                neutral_matches = pt.eq(
                    y_val[:, None], neutrals_const[None, :]
                )  # (N, |neutrals|)
                has_neutral = pt.any(neutral_matches, axis=1)  # (N,)
            else:
                has_neutral = pt.zeros_like(any_match)

            p = pt.where(has_neutral, 1.0, pt.where(any_match, p_valid, eps))
            return p

        return builder

    # --------------------------------------------------------------------------
    #                       Abstract CDF hooks for subclasses
    # --------------------------------------------------------------------------
    def _cdf_jax(self, z):
        """Subclass hook for the cumulative distribution function in JAX."""
        raise NotImplementedError

    def _cdf_pt(self, z):
        """Subclass hook for the cumulative distribution function in PyTensor."""
        raise NotImplementedError


class OrderedLogit(OrderedBase):
    """Ordered response model using the logistic cumulative distribution function.

    Returns per-observation probabilities.
    """

    def _cdf_jax(self, z):
        import jax

        return jax.nn.sigmoid(z)

    def _cdf_pt(self, z):
        return 1.0 / (1.0 + pt.exp(-z))

    def __repr__(self):
        return f'OrderedLogit({repr(self.eta)})'


class OrderedProbit(OrderedBase):
    """Ordered response model using the standard normal cumulative distribution function.

    Returns per-observation probabilities.
    """

    def _cdf_jax(self, z):
        import jax.numpy as jnp
        import jax.lax as lax

        return 0.5 * (1.0 + lax.erf(z / jnp.sqrt(2.0)))

    def _cdf_pt(self, z):
        return 0.5 * (1.0 + pt.erf(z / pt.sqrt(pt.as_tensor_variable(2.0))))

    def __repr__(self):
        return f'OrderedProbit({repr(self.eta)})'


class OrderedLogLogit(OrderedLogit):
    """Ordered response model using logistic CDF, returning per-observation log-likelihoods."""

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        import jax.numpy as jnp

        base_fn = super().recursive_construct_jax_function(numerically_safe)

        def wrapper(*args, **kwargs):
            p = base_fn(*args, **kwargs)
            return jnp.log(jnp.clip(p, self.eps, 1.0))

        return wrapper

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        base_builder = super().recursive_construct_pymc_model_builder()
        eps = pt.as_tensor_variable(self.eps)

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            p = base_builder(dataframe)
            return pt.log(pt.clip(p, eps, 1.0))

        return builder


class OrderedLogProbit(OrderedProbit):
    """Ordered response model using probit CDF, returning per-observation log-likelihoods."""

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        import jax.numpy as jnp

        base_fn = super().recursive_construct_jax_function(numerically_safe)

        def wrapper(*args, **kwargs):
            p = base_fn(*args, **kwargs)
            return jnp.log(jnp.clip(p, self.eps, 1.0))

        return wrapper

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        base_builder = super().recursive_construct_pymc_model_builder()
        eps = pt.as_tensor_variable(self.eps)

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            p = base_builder(dataframe)
            return pt.log(pt.clip(p, eps, 1.0))

        return builder
