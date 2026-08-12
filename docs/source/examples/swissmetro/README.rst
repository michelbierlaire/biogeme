Biogeme examples for the Swissmetro data
****************************************

You find here several examples of models that can be estimated and simulated with Biogeme.

The example :ref:`plot_b28_parameter_overrides` introduces explicit parameter
overrides.  It shows how to replace a ``Beta`` by a fixed ``Beta`` or by a
numeric expression before constructing the ``BIOGEME`` object.

The example :ref:`plot_b27_monte_carlo_diagnostic` illustrates the
post-estimation Monte Carlo draw-stability diagnostic.  It evaluates the
objective and gradient at the fixed estimates with fresh draw designs of
increasing size.  The diagnostic can be interrupted and resumed; it writes a
raw YAML checkpoint and an American-English Markdown report separately from
the ordinary estimation result.
