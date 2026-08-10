Monte Carlo draw-stability diagnostic
=====================================

The post-estimation diagnostic evaluates the model criterion and gradient at
the already estimated parameter vector using fresh draw designs. It does not
run the optimizer, re-estimate the model, or calculate a Hessian. The normal
estimation YAML is never modified: checkpoints and the explanatory report use
the companion suffixes ``_monte_carlo_diagnostic.yaml`` and
``_monte_carlo_diagnostic.md``.

Run after estimation
--------------------

The diagnostic is explicit and disabled by default::

   results = the_biogeme.estimate_or_load()
   diagnostic = the_biogeme.check_monte_carlo_stability(
       estimation_results=results
   )

The returned object exposes ``execution_status``, ``diagnostic_conclusion``,
``recommendation``, and the paths of both output files. A compatible checkpoint
is resumed by default. Pass ``resume=False`` to replace a previous diagnostic
run with a fresh one.

Standalone postprocessing
-------------------------

A postprocessing script should reconstruct exactly the same database and model,
then load the estimation result directly. Checking the path first ensures that
a missing result fails clearly instead of starting an estimation::

   from pathlib import Path

   from biogeme.results_processing import EstimationResults

   # Reconstruct database, log_likelihood, and the_biogeme here.
   the_biogeme.model_name = 'my_model'

   estimation_file = Path('saved_results/my_model.yaml')
   if not estimation_file.is_file():
       raise FileNotFoundError(
           f'No completed estimation result exists: {estimation_file}'
       )

   results = EstimationResults.from_yaml_file(filename=estimation_file)
   diagnostic = the_biogeme.check_monte_carlo_stability(
       estimation_results=results,
       output_directory='saved_results',
   )

Configuration and stopping
--------------------------

The settings are in the ``MonteCarlo`` section of ``biogeme.toml``. They
control draw-count factors, independent replications, tolerances, the maximum
draw count, and runtime forecasting. The first Ctrl-C requests a graceful stop
after the active evaluation finishes and is checkpointed. A second Ctrl-C may
terminate immediately. An interrupted run can still produce a conclusive
recommendation when sufficient evaluations have completed.

The complete list of settings and their defaults is provided in
:doc:`code/toml`.
