Numerically safe likelihood evaluation
======================================

Biogeme can prioritize numerical robustness by enabling
``numerically_safe``::

    the_biogeme = BIOGEME(
        database,
        log_probability,
        numerically_safe=True,
    )

The same setting can be placed in the ``Specification`` section of
``biogeme.toml``::

    [Specification]
    numerically_safe = true

The default is ``false`` for performance and backward compatibility. Safe
mode is recommended when utilities may be extreme or when availability can
make a nest empty. The setting is propagated to likelihood values, gradients,
Hessians, BHHH matrices, simulation, bootstrap estimation, cross-validation,
and Monte Carlo diagnostics. The PyMC/PyTensor likelihood builders use the
finite safe formulation unconditionally.

Unavailable alternatives and empty nests
-----------------------------------------

An availability expression equal to zero makes its alternative unavailable;
any nonzero value makes it available. An unavailable alternative contributes
no probability mass. A nest is active only if it contains at least one
available alternative with a positive membership. Empty nests are excluded
from the effective denominator.

If the recorded chosen alternative is unavailable, the observation is
inconsistent with the model specification. Safe mode returns the finite log
probability sentinel ``-1e30`` for that observation. This keeps automatic
derivatives and result serialization finite, but it does not repair the data:
users should inspect the choice and availability definitions. The same
sentinel is returned if every alternative is unavailable or if the chosen
alternative has no positive CNL membership.

Finite masking strategy
-----------------------

The safe logit, nested-logit, dense CNL, and sparse CNL evaluators remain in
the log domain. They determine active alternatives and nests directly from
availability and membership masks, and use the finite sentinel ``-1e30`` for
inactive log terms. Empty-nest log sums are replaced by a finite neutral value
before they are multiplied by a nest coefficient.

Using ``where(condition, valid_value, fallback)`` is not sufficient if either
value contains undefined arithmetic. Automatic-differentiation systems may
evaluate or trace both branches, so expressions such as ``0 * -inf`` or
``-inf - -inf`` can still yield NaN derivatives. Biogeme therefore makes both
branches finite before applying a conditional mask.

Safe and unsafe paths
---------------------

When every relevant alternative and nest is active and utilities are in a
regular numerical range, safe and unsafe evaluations are mathematically
equivalent up to floating-point precision. The unsafe path retains the faster
direct formulation and may produce infinities or NaNs for extreme utilities
or empty nests. Safe mode incurs additional masks and log-domain operations.

Availability and positive CNL membership define discrete regimes. Derivatives
are calculated within the regime selected at the evaluation point. At an
allocation parameter equal to zero, the corresponding CNL edge is inactive
and its local derivative contribution is zero. This convention provides
finite values and derivatives at the boundary; users should not interpret it
as a derivative across a change in model topology.

Second derivatives
------------------

The safe paths support analytical JAX gradients, Hessians, and BHHH matrices,
including observations that empty one or several nests. Hessian memory
management is a separate concern: large models may still require the chunked
analytical Hessian configuration described by the estimation parameters.
Numerical safety prevents undefined mask arithmetic; it does not reduce the
size of the differentiated model.
