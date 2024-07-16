Configuration parameters
========================

Biogeme can be configured using a parameter file. By default, the name is supposed to be ``biogeme.toml``. If such a
file does not exist, Biogeme will create one containing the default values. The following table provides a description
of all parameters that can be configured.
    

.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Default
     - Section
     - Type
   * - largest_neighborhood
     - int: size of the largest neighborhood copnsidered by the Variable Neighborhood Search (VNS) algorithm.
     - 20
     - AssistedSpecification
     - int
   * - maximum_attempts
     - int: an attempts consists in selecting a solution in the Pareto set, and trying to improve it. The parameter imposes an upper bound on the total number of attempts, irrespectively if they are successful or not.
     - 100
     - AssistedSpecification
     - int
   * - maximum_number_parameters
     - int: maximum number of parameters allowed in a model. Each specification with a higher number is deemed invalid and not estimated.
     - 50
     - AssistedSpecification
     - int
   * - number_of_neighbors
     - int: maximum number of neighbors that are visited by the VNS algorithm.
     - 20
     - AssistedSpecification
     - int
   * - version
     - Version of Biogeme that created the TOML file. Do not modify this value.
     - 3.2.14a1
     - Biogeme
     - str
   * - bootstrap_samples
     - int: number of re-estimations for bootstrap sampling.
     - 100
     - Estimation
     - int
   * - large_data_set
     - If the number of observations is larger than this value, the data set is deemed large, and the default estimation algorithm will not use second derivatives.
     - 100000
     - Estimation
     - int
   * - max_number_parameters_to_report
     - int: maximum number of parameters to report during the estimation.
     - 15
     - Estimation
     - int
   * - maximum_number_catalog_expressions
     - If the expression contains catalogs, the parameter sets an upper bound of the total number of possible combinations that can be estimated in the same loop.
     - 100
     - Estimation
     - int
   * - optimization_algorithm
     - str: optimization algorithm to be used for estimation. Valid values: ['automatic', 'scipy', 'LS-newton', 'TR-newton', 'LS-BFGS', 'TR-BFGS', 'simple_bounds', 'simple_bounds_newton', 'simple_bounds_BFGS']
     - automatic
     - Estimation
     - str
   * - save_iterations
     - bool: If True, the current iterate is saved after each iteration, in a file named ``__[modelName].iter``, where ``[modelName]`` is the name given to the model. If such a file exists, the starting values for the estimation are replaced by the values saved in the file.
     - True
     - Estimation
     - bool
   * - number_of_draws
     - int: Number of draws for Monte-Carlo integration.
     - 100
     - MonteCarlo
     - int
   * - seed
     - int: Seed used for the pseudo-random number generation. It is useful only when each run should generate the exact same result. If 0, a new seed is used at each run.
     - 0
     - MonteCarlo
     - int
   * - number_of_threads
     - int: Number of threads/processors to be used. If the parameter is 0, the number of available threads is calculated using cpu_count().
     - 0
     - MultiThreading
     - int
   * - generate_html
     - bool: "True" if the HTML file with the results must be generated.
     - True
     - Output
     - bool
   * - generate_pickle
     - bool: "True" if the pickle file with the results must be generated.
     - True
     - Output
     - bool
   * - identification_threshold
     - float: if the smallest eigenvalue of the second derivative matrix is lesser or equal to this parameter, the model is considered not identified. The corresponding eigenvector is then reported to identify the parameters involved in the issue.
     - 1e-05
     - Output
     - float
   * - only_robust_stats
     - bool: "True" if only the robust statistics need to be reported. If "False", the statistics from the Rao-Cramer bound are also reported.
     - True
     - Output
     - bool
   * - enlarging_factor
     - If an iteration is very successful, the radius of the trust region is multiplied by this factor
     - 10
     - SimpleBounds
     - float
   * - infeasible_cg
     - If True, the conjugate gradient algorithm may generate infeasible solutions until termination.  The result will then be projected on the feasible domain.  If False, the algorithm stops as soon as an infeasible iterate is generated
     - False
     - SimpleBounds
     - bool
   * - initial_radius
     - Initial radius of the trust region
     - 1
     - SimpleBounds
     - float
   * - max_iterations
     - int: maximum number of iterations
     - 1000
     - SimpleBounds
     - int
   * - second_derivatives
     - float: proportion (between 0 and 1) of iterations when the analytical Hessian is calculated
     - 1.0
     - SimpleBounds
     - float
   * - steptol
     - The algorithm stops when the relative change in x is below this threshold. Basically, if p significant digits of x are needed, steptol should be set to 1.0e-p.
     - 1e-05
     - SimpleBounds
     - float
   * - tolerance
     - float: the algorithm stops when this precision is reached
     - 0.0001220703125
     - SimpleBounds
     - float
   * - missing_data
     - number: If one variable has this value, it is assumed that a data is missing and an exception will be triggered.
     - 99999
     - Specification
     - int
   * - dogleg
     - bool: choice of the method to solve the trust region subproblem. True: dogleg. False: truncated conjugate gradient.
     - True
     - TrustRegion
     - bool

The structure of the ``biogeme.toml`` file is as follows.

.. literalinclude:: biogeme.toml
   :language: none
   :linenos:

.. toctree::
  :maxdepth: 2
  :caption: Configuration parameters
