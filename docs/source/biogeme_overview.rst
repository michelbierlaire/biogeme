Biogeme code: how the pieces fit together
==========================================

The generated pages under :doc:`code/biogeme_api` are the complete API
reference. They are intentionally close to the Python source. This page
provides the conceptual map needed before reading those detailed pages.

The main modelling path
-----------------------

A Biogeme model normally moves through the following layers:

1. **Data**: a :class:`~biogeme.database.Database` wraps a Pandas data frame
   and provides the observations, availability conditions, and panel
   structure.
2. **Expressions**: :class:`~biogeme.expressions.Variable` and
   :class:`~biogeme.expressions.Beta` objects represent data columns and
   parameters. Arithmetic on these objects builds an expression tree rather
   than immediately producing a numeric result.
3. **Model probability**: functions such as
   :func:`~biogeme.models.logit.loglogit`, nested logit, and cross-nested logit
   combine expressions into a log-likelihood contribution.
4. **Estimation controller**: :class:`~biogeme.biogeme.BIOGEME` connects the
   database and model expression to an estimation algorithm, configuration,
   starting values, and output files.
5. **Results**: estimation returns a results object that can be inspected in
   Python or exported through the result-processing helpers.

The corresponding code is usually recognizable in an example::

    database = Database('my_database', pandas_dataframe)
    beta_time = Beta('beta_time', 0.0, None, None, 'Utility')
    utility = beta_time * Variable('travel_time')
    probability = loglogit({1: utility}, Variable('choice'))
    biogeme = BIOGEME(database, probability)
    results = biogeme.estimate()

The exact model specification is application-dependent. The tutorial examples
in :doc:`examples` show complete runnable versions of this pattern.

How the layers interact
-----------------------

Expression objects are evaluated by a calculator backend. The standard
NumPy-based path is used for many estimation and simulation tasks; JAX and
PyMC/PyTensor backends are available for models that require automatic
differentiation or Bayesian estimation. The expression tree is the shared
model description, while each backend supplies its own numerical evaluator.

The :class:`~biogeme.biogeme.BIOGEME` object is deliberately the orchestration
layer, not the place where every model formula is implemented. Model formulas
live in :mod:`biogeme.models`, expression classes live in
:mod:`biogeme.expressions`, and data handling lives in
:mod:`biogeme.database`. Keeping those responsibilities separate makes it
possible to use the same specification for estimation, simulation,
elasticities, validation, and Bayesian workflows.

Where to look next
------------------

* Start with :doc:`examples` for end-to-end model specifications.
* Read :doc:`code/biogeme/biogeme` for the estimation controller.
* Read :doc:`code/biogeme/database/index` for data and panel handling.
* Read :doc:`code/biogeme/expressions/index` for the expression hierarchy.
* Read :doc:`code/biogeme/models/index` for probability-model implementations.
* Read :doc:`code/biogeme/results_processing/index` for reporting and export.
* Read :doc:`code/biogeme/bayesian_estimation/index` for Bayesian-specific
  results and sampling components.

The generated API pages document all public members with docstrings. They do
not replace this overview: when a module's purpose or the relationship between
several modules matters, the narrative documentation and the examples are the
best starting point.
