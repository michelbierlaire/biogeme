# Biogeme 3.3.5

Numerically safe likelihood evaluation has been strengthened for logit,
nested logit, cross-nested logit, and sparse cross-nested logit models.
Unavailable alternatives and empty nests now use finite log-domain masks in
all differentiated expressions when numerical safety is enabled, preventing
NaN gradients, Hessians, and BHHH matrices. The performance-oriented JAX paths
also use finite logit masks, Boolean availability semantics, explicit choice
validation, and the common finite invalid-choice sentinel. Their nested and
cross-nested logit calculations now tolerate availability-induced empty nests
without non-finite derivatives, while retaining direct arithmetic for ordinary
models. NumPy, JAX, and PyMC/PyTensor paths now share the same finite
invalid-choice behavior, and regression coverage includes the Swissmetro
singleton-nest availability case.

# Biogeme 3.3.4

Biogeme 3.3.4 introduces a new expression-based computational backend for
nested logit, cross-nested logit, and sampled choice models. These dedicated
expressions preserve the model structure during JAX compilation, provide more
efficient function, gradient, and Hessian evaluations, and support numerically
stable log-domain calculations. The existing public model interfaces remain
available for backward compatibility.

The new sparse cross-nested logit implementation, available through
`sparse_cnl`, avoids evaluating structurally zero allocation terms while
preserving all parameter-dependent memberships. The original `cnl`
implementation remains unchanged, and Biogeme reports when the sparsity
pattern suggests that the sparse implementation may be advantageous.

The calculation of analytical Hessians has been substantially improved.
Biogeme now supports exact chunked Hessian calculations based on JAX
Hessian-vector products, configurable parameter and observation blocks, and an
automatic memory-aware strategy that accounts for the effective memory
available to the process, including resource limits imposed by systems such as
Slurm. These options are exposed through the estimation parameters
`analytical_hessian_mode`, `hessian_parameter_block_size`,
`hessian_observation_batch_size`, and `hessian_memory_fraction`. Chunked
calculations report their progress and estimated completion time.

Estimation results are now written as recoverable checkpoints throughout the
estimation and post-estimation phases. If the calculation of gradients, BHHH
matrices, Hessians, or bootstrap results is interrupted or fails, Biogeme can
resume the missing phases without repeating the optimization. Result
serialization is also performed atomically, reducing the risk of corrupt YAML
files.

A new post-estimation Monte Carlo draw-stability diagnostic evaluates the
objective and gradient at fixed estimates using fresh draw designs and
increasing numbers of draws. It supports independent replications, runtime
budgets, interruption and resumption, and configurable stability tolerances.
Raw diagnostic results and an explanatory Markdown report are stored separately
from the ordinary estimation results, allowing users to determine whether
additional simulation draws are required.

Finally, robustness and reproducibility have been improved through expanded
regression tests, JAX benchmarks, better handling of bootstrap execution in
restricted environments, and enhanced support for recent Python versions.
Biogeme 3.3.4 requires Python 3.12 or later and is tested with Python 3.14.

# Biogeme 3.3.3

The main new feature introduced in Biogeme 3.3.3 is a completely redesigned
framework for the specification and estimation of hybrid choice models, that is,
discrete choice models involving latent variables. The new implementation provides
a high-level object-oriented interface that allows analysts to define latent variables,
structural equations, measurement equations, normalization rules, and choice models in
a modular and transparent way. This substantially reduces the amount of code required to
specify complex models, improves readability and maintainability, and facilitates the development and comparison of alternative model specifications. The framework supports a wide range of configurations, from simple MIMIC models to full hybrid choice models integrating multiple latent constructs and measurement systems.
In addition, several improvements have been made to the Bayesian estimation framework
introduced in version 3.3.2. These include a more robust estimation workflow,
improved management of estimation results through YAML summaries and NetCDF files,
and enhanced diagnostic and post-processing capabilities. Reporting facilities have also
been strengthened, with better support for parameter grouping and more consistent generation
of HTML, LaTeX, and pandas summaries.
Finally, compatibility has been updated for recent Python releases, including Python 3.14,
and development workflows based on uv are now fully supported.


# Biogeme 3.3.2

The main new features introduced in Biogeme 3.3.2 are the integration
of PyMC for Bayesian estimation and the introduction of a set
of dedicated Python classes designed to facilitate the specification
and estimation of hybrid choice models. The PyMC interface enables
full Bayesian inference for complex models involving latent variables,
providing access to posterior distributions, credible intervals, and
diagnostic tools for convergence and model assessment. In parallel,
the new object-oriented specification framework allows analysts to
define structural equations, measurement equations, normalization
rules, and choice models in a modular and transparent way. This
significantly reduces boilerplate code, improves readability and
maintainability of model specifications, and makes it easier to
combine or compare different model variants (such as choice-only,
MIMIC, and full hybrid choice models) within a unified and consistent
framework.


# Biogeme 3.3.1

Biogeme 3.3.1 is a major release. In earlier versions, the computation of the likelihood function and its derivatives
was handled by Cythonbiogeme, a C++-based engine. Starting with this release, Biogeme now relies on JAX, a
high-performance numerical computing library for Python developed by Google Research. This transition brings
substantial performance improvements, particularly for mixture models that require Monte Carlo integration.
Although the transition has been designed to be seamless, some minor adjustments to existing models may still be necessary.


# Biogeme 3.2.14

<p>
In this release, various improvements have been made, including code
reorganization and documentation, bug fixes, and new functionalities. In particular, the name of several objects and functions have been modified for a better compliance with the Python recommendations. The old syntax has been maintained, but is tagged as deprecated.
</p>
<ul>
    <li>The implementation of the arithmetic expressions (cythonbiogeme) has been optimized for better numerical stability. See the <a href="https://transp-or.epfl.ch/documents/technicalReports/Bier24.pdf">technical report</a> for details.</li>
    <li>The management of the parameters has been simplified. Indeed, it can be done either using the <code>biogeme.toml</code> file, or directlywhen constructing the BIOGEME object.</li>
    <li>The Multiple Discrete Continuous Extreme Value (MDCEV) model has been validated. It is possible to estimate its parameters, and to use the estimated model for forecasting. See the <a href="https://transp-or.epfl.ch/documents/technicalReports/BierWang24.pdf">technical report</a> for details.</li>
    <li>The files preparing the data for Swissmetro, Optima and the MDCEV data set are included in the distribution.</li>
</ul>


# Biogeme 3.2.13

<dl>
<dt>MDCEV</dt>
<dd>The Multiple Discrete Continuous Extreme Value model has been implemented. The code is still experimental, and the documentation is not ready yet.
</dd>
<dt>Local-sensitivity hashing</dt>
<dd>The data reduction method introduced by <a href='https://transp-or.epfl.ch/documents/technicalReports/OrteLappBier2023.pdf'>Ortelli et al. (2023)</a> has been implemented. It has not yet been integrated in the optimization framework.</dd>
<dt>Nests definition</dt>
<dd>The definition of the nests for the nested logit and the cross-nested logit models has been improved, using specific objects. The calculation of the correlation structure among the alternatives is now performed by those objects, and not anymore by the <samp>bioResults</samp> object as in previous versions.</dd>
<dt>Sampling of alternatives</dt>
<dd>The methods for the sampling of alternatives have been completely reimplemented. A report with a complete documentation will be available soon.</dd>
<dt>Examples</dt>
<dd>The structure of the examples has been revisited. They are now integrated in the Sphinx documentation, and available both as Python scripts and Jupyter notebooks. <a href='sphinx/auto_examples/index.html'>Click here.</a></dd>
<dt>Non convergence</dt>
<dd>The reporting has been improved when the algorithm does not converge. </dd>
<dt>Logging</dt>
<dd>The logging module has been renamed from <code>biogeme.logging</code> into <code>biogeme.biogeme_logging.py</code>. It was necessary because of the ambiguity with the <code>logging</code> module from Python.
</dd>
<dt>File organization</dt>
<dd>Several scripts have been reorganized into modules. This improves the code readability and should be transparent for the user.</dd>


</dl>


# Biogeme 3.2.12

<p>
This release mainly implements some re-organization of the code and bugs fixes. In particular, the generic optimization algorithms are now distributed in a different package, called biogeme_optimization.
</p>


# Biogeme 3.2.11

<dl>

<dt>Sampling of alternatives</dt>
<dd>It is now possible to estimate logit, nested logit and cross-nested logit models using only a sample of alternatives. </dd>

<dt>Assisted specification</dt>
<dd>The assisted specification algorithm has been completely redesigned. The concept of <samp>Catalog</samp> has been introduced to allow the modeler to suggest several versions of the model specification. The possible versions can either be fully enumerated (if their number allows for it) or can be algorithmically investigated. </dd>

<dt>Pareto optimality</dt>
<dd>It is possible to extract the Pareto optimal models from a list of estimation results.</dd>


<dt>TOML file for the definition of the parameters</dt>
<dd>A commented parameter file is now available to modify the various parameters of Biogeme. A  version of the file with default values of the parameters is created the first time Biogeme is executed in a directory. Note that parameters can still be defined directly from the Python script. It particularly simplifies the definitions of the parameters controlling the optimization algorithms. </dd>


<dt>Explicit definition of the Beta parameters for simulation</dt>
<dd>The <samp>simulate</samp> function now requires an explicit definition for the value of the parameters. The initial values can be retrieved from the <samp>get_beta_values</samp> function of a Biogeme expression. The estimated values can be retrieved from the <samp>getBetaValues</samp> function of the <samp>bioResult</samp> object.</dd>

<dt>Use of the standard Python logging system</dt>
<dd>The <samp>messaging</samp> module used to control the verbosity of Biogeme is now obsolete. Biogeme implements the standard Python logging system. If you do not know what it is, Biogeme includes a simple <samp>logging</samp> module, that provides simple access to the logging system.
</dd>

<dt>Naming conventions</dt>
<dd>Some object/functions/variables have been renamed to comply better with the common Python practice. For example, the exception <samp>biogemeError</samp>, defined in the <samp>exceptions</samp> module is now called <samp>BiogemeError</samp>. </dd>

<dt>Removed functions from the <samp>database</samp> module</dt>
<dd>The functions <samp>sumFromDatabase</samp> and <samp>sampleWithoutReplacement</samp> are no longer available.</dd>
<dt>New expression: <samp>logzero</samp></dt>
<dd><samp>logzero(x)</samp> returns the logarithm of x if x is not zero, and zero otherwise.</dd>



</dl>


# Biogeme 3.2.10

<p><strong>Note</strong>: versions 3.2.9 and 3.2.10 are
	  identical. Therefore, version 3.2.9 has been removed from
	  the official distribution platform. </p>
	  <dl>
	    <dt>New syntax for <code>DefineVariable</code></dt>
	    <dd>
	      <p><code>DefineVariable</code> actually defines a new column in the
		database. The old syntax was:</p>
	      <p><code>myvar = DefineVariable('myvar', x * y + 2,
		  database)</code></p>
	      <p>The new syntax is:</p>
	      <p><code>myvar = database.DefineVariable('myvar', x * y +
		  2)</code></p>
	    </dd>
	    <dt>Likelihood ratio test</dt>
	    <dd>It is now possible to perform a likelihood ratio test
	    directly from the estimation results. See the documentation. It relies on a function that can
	    be used in more general context.</dd>
	    <dt>Comparing several models</dt>
	    <dd>It is now possible to compile the estimation results
	    from several models into a single data frame. See the documentation. </dd>
	    <dt>Automatic segmentation</dt>
	    <dd>It is now possible to define a parameter such that it
	    has a different value for each segment in the
	    population. </dd>
	    <dt>Simulation of panel data</dt>
	    <dd>It is now possible to use Biogeme in simulation mode
	    for panel data. See the following
	      example.</dd>
	    <dt>Flattening panel data</dt>
	    <dd>This new feature transforms a database organized in
	    panel mode (that is, one row per observation) into a
	    database organized in normal mode (that is, one row per
	    individual, and the observations of each individual
	    across columns). See documentation
	    the documentation.</dd>
	    <dt>Covariance and correlation matrix of the nested and
	      the cross-nested logit
	      models</dt>
	    <dd>These new functions calculate the covariance and the correlation matrix
	    of the error terms of a cross-nested logit model from the
	    estimated parameters. See
	    documentation.</dd>
	    <dt>Recycling estimation results</dt>
	    <dd>It is now possible to skip estimation and read the
	    estimation results from the pickle file by setting the
	    parameter <code>recycle=True</code>. See the online documentation.</dd>
	    <dt>The feature removing unused variables has been
	      canceled.</dt>
	    <dd>The parameters <code>removeUnusedVariables</code>
	    and <code>displayUsedVariables</code> in the BIOGEME
	      constructor have been removed.</dd>
	    <dt>More functionalities for the mathematical expressions.</dt>
	    <dd>The expressions have now been designed to also be
	    available outside of the BIOGEME class. A detailed
	    illustration of the functionalities is available
	    [<a href="https://github.com/michelbierlaire/biogeme/blob/master/examples/notebooks/biogeme.expressions.ipynb">Click
		here</a>].</dd>
	    <dt>New syntax for the assisted specification algorithm</dt>
	    <dd>The new syntax involves <code>NamedTuple</code> to make the code
	      more readable. Refer to the examples, such as
	      <code>optima.py</code>.</dd>

	  </dl>


# Biogeme 3.2.8

<p>Note that version 3.2.7 and 3.2.8 are almost
	  identical. The description belows compares to version 3.2.6.
	  <dl>
	    <dt>Assisted specification</dt>
	    <dd>The asssisted specification algorithm
	    by <a href="https://dx.doi.org/10.1016/j.jocm.2021.100285">Ortelli
	      et al. (2021)</a> is now available. </dd>
	    <dt>Optimization</dt>
	    <dd>The optimization algorithms have been organized into
	      two modules. The
	      module <code>algorithms.py</code>
	      contains generic optimization algorithms. The
	      module <code>optimization.py</code>
	      contains the functions that can be called directly by
	      Biogeme. The old example is no longer distributed.</dd>
	    <dt>CFSQP</dt>
	    <dd>The CFSQP algorithm has been removed from the
	    distribution. </dd>
	    <dt>Null log likelihood</dt>
	    <dd>The log likelihood is calculated. The null model
	      predicts equal probability for each alternative.</dd>
	    <dt>Saved iterations</dt>
	    <dd>Iterations are saved in a file with
	      extension <code>.iter</code>. If the file exists, Biogeme
	      will initialize the parameters from this .py, and
	      ignore the starting values provided. To turn this feature
	      off, set <code>biogeme.saveIterations=False</code></dd>
	    <dt>Random starting values</dt>
	    <dd>It is possible to modify the initial values of the parameters in all formulas,
              using randomly generated values. The value is drawn from a
              uniform distribution on the interval defined by the
              bounds (by default [-100, 100].)
              See the current Biogeme documentation.
	      <dt>Sensitivity analysis</dt>
	    <dd>The betas for sensitivity analysis are now generated
	      by bootstrapping.
		      See the current results documentation.</dd>
	    <dt>Box-Cox</dt>
	    <dd>The implementation of
		      the Box-Cox transform was incorrect and has been corrected.
	      <dt>Validation</dt>
	    <dd>The out-of-sample validation has been
		      improved. It has to be combined with the <code>split</code> function
	      of the database object.
	      <dt>Statistics about chosen alternatives</dt>
	    <dd>It is now possible to calculate the number of time
	      each alternative is chosen and available in the
		      sample.</dd>
	    <dt>Validity check for the nests</dt>
	    <dd> The validity of the specification of the nests
	      for nested and cross nested logit models is new
	      checked.</dd>
	    <dt>ALOGIT file</dt>
	    <dd>Output .py in F12 format compatible with ALOGIT can
	      now be
		      produced.</dd>
	    <dt>Likelihood ratio test</dt>
	    <dd>A function to perform the likelihood ratio test has
	      been
			  implemented.</dd>
	  </dl>


# Biogeme 3.2.6

<dl>
	    <dt>Optimization</dt>
	    <dd>New optimization algorithms are
	      available for estimation See the documentation of
	      the <code>estimate</code>
	      function, and
	      the <code>optimization</code> module.</dd>
	    <dt>Stochastic log likelihood</dt>
	    <dd>It is now possible to calculate the log likelihood
	      function on a sample (a batch) of the full data
	      file. This is particularly useful with large
	      databases. It can be used in the implementation of a
	      stochastic gradient algorithm, for instance.</dd>
	    <dt>User's notes</dt>
	    <dd>It is possible to include your own notes in the HTML
	      file using the <code>user_notes</code> parameter of the
	      <code>biogeme</code>
	      object. See the current documentation. The old example is no longer distributed.</dd>
	    <dt>Scaling</dt>
	    <dd>It is possible to have Biogeme suggesting the scales
	      of the variables in the database using
	      the <code>suggestScales</code> parameter of
	      the <code>biogeme</code>
	      object. See the current documentation.</dd>
	    <dt>Estimation</dt>
	    <dd>A new function <code>quickEstimate</code> performs
	      the estimation of the parameters, and skips the
	      calculation of the
	      statistics. </dd>
	    <dt>Validation</dt>
	    <dd>A new function in the <code>database</code> module allows to split the database in order to
	      prepare an estimation and a validation sets, for
	      out-of-sample
	      validation. It
	      is used by the new function <code>validate</code> in the
	      <code>biogeme</code> module. The old example is no longer distributed.</dd>
	    <dt>Messages</dt>
	    <dd>A new function allows to extract all the messages
	      generated during a
	      run. See the current documentation. It
	      is also possible to make the logger temporarily silent
	      using the functions <code>temporarySilence</code> and <code>resume</code>.</dd>
	  </dl>
