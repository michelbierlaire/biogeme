faq = {}

faq['What was new in Biogeme 3.2.12?'] = """
<p>
This release mainly implements some re-organization of the code and bugs fixes. In particular, the generic optimization algorithms are now distributed in a different package, called biogeme_optimization.
</p>
"""

faq['What was new in Biogeme 3.2.11?'] = """
<dl>

<dt>Sampling of alternatives</dt>
<dd>It is now possible to estimate logit, nested logit and cross-nested logit models using only a sample of alternatives. </dd>

<dt>Assisted specification</dt>
<dd>The assisted specification algorithm has been completely redesigned. The concept of <a href="sphinx/catalog.html#biogeme.catalog.Catalog"><samp>Catalog</samp></a> has been introduced to allow the modeler to suggest several versions of the model specification. The possible versions can either be fully enumerated (if their number allows for it) or can be algorithmically investigated. </dd>

<dt>Pareto optimality</dt>
<dd>It is possible to extract the Pareto optimal models from a list of estimation results.</dd>


<dt>TOML file for the definition of the parameters</dt>
<dd>A commented parameter file is now available to modify the various parameters of Biogeme. A  version of the file with default values of the parameters is created the first time Biogeme is executed in a directory. Note that parameters can still be defined directly from the Python script. It particularly simplifies the definitions of the parameters controlling the optimization algorithms. </dd>


<dt>Explicit definition of the Beta parameters for simulation</dt>
<dd>The <samp>simulate</samp> function now requires an explicit definition for the value of the parameters. The initial values can be retrieved from the <samp>get_beta_values</samp> function of a Biogeme expression. The estimated values can be retrieved from the <samp>getBetaValues</samp> function of the <samp>bioResult</samp> object.</dd>

<dt>Use of the standard Python logging system</dt>
<dd>The <samp>messaging</samp> module used to control the verbosity of Biogeme is now obsolete. Biogeme implements the standard Python logging system. If you do not know what it is, Biogeme includes a simple <a href="sphinx/logging.html"><samp>logging</samp></a> module, that provides simple access to the logging system.
</dd>

<dt>Naming conventions</dt>
<dd>Some object/functions/variables have been renamed to comply better with the common Python practice. For example, the exception <samp>biogemeError</samp>, defined in the <samp>exceptions</samp> module is now called <samp>BiogemeError</samp>. </dd>

<dt>Removed functions from the <samp>database</samp> module</dt>
<dd>The functions <samp>sumFromDatabase</samp> and <samp>sampleWithoutReplacement</samp> are no longer available.</dd>
<dt>New expression: <samp>logzero</samp></dt>
<dd><samp>logzero(x)</samp> returns the logarithm of x if x is not zero, and zero otherwise.</dd>



</dl>
"""

faq['What was new in Biogeme 3.2.10?'] = """
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
	    directly from the estimation results. <a href="sphinx/results.html#biogeme.results.bioResults.likelihood_ratio_test">See
		documentation here. </a> It relies on a function that can
	    be used in more general context. <a href="sphinx/tools.html#biogeme.tools.likelihood_ratio_test"> See
	    documentation here.<a></dd>
	    <dt>Comparing several models</dt>
	    <dd>It is now possible to compile the estimation results
	    from several models into a single data frame. <a href="sphinx/results.html#biogeme.results.compileEstimationResults">See
		documentation here. </a> </dd>
	    <dt>Automatic segmentation</dt>
	    <dd>It is now possible to define a parameter such that it
	    has a different value for each segment in the
	    population. See the example <a href="examples/swissmetro/01logitBis.py"><code>01logitBis.py</code></a>.</dd>
	    <dt>Simulation of panel data</dt>
	    <dd>It is now possible to use Biogeme in simulation mode
	    for panel data. See the following
	      example: <a href="examples/swissmetro/13panel_simul.py"><code>13panel_simul.py</code></a>.</dd>
	    <dt>Flattening panel data</dt>
	    <dd>This new feature transforms a database organized in
	    panel mode (that is, one row per observation) into a
	    database organized in normal mode (that is, one row per
	    individual, and the observations of each individual
	    across columns). See documentation
	    <a href="sphinx/database.html#biogeme.database.Database.generateFlatPanelDataframe">here</a> and <a href="sphinx/tools.html#biogeme.tools.flatten_database">here</a></dd>
	    <dt>Covariance and correlation matrix of the nested and
	      the cross-nested logit
	      models</dt>
	    <dd>These new functions calculate the covariance and the correlation matrix
	    of the error terms of a cross-nested logit model from the
	    estimated parameters. See
	    documentation <a href="sphinx/tools.html#biogeme.tools.calculate_correlation">here</a>,
	    <a href="sphinx/tools.html#biogeme.tools.correlation_nested">here</a>, <a href="sphinx/tools.html#biogeme.tools.covariance_cross_nested">here</a> and <a href="sphinx/tools.html#biogeme.tools.correlation_cross_nested">here</a>.</dd>
	    <dt>Recycling estimation results</dt>
	    <dd>It is now possible to skip estimation and read the
	    estimation results from the pickle file by setting the
	    parameter <code>recycle=True</code>. See the online
	    documentation [<a href="sphinx/biogeme.html#biogeme.biogeme.BIOGEME.estimate">here</a>].</dd>
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
	      <a href="examples/assisted/optima.py"
		 target="_blank"><code>optima.py</code></a>.</dd>
	    
	  </dl>

"""
faq['What was new in Biogeme 3.2.8?'] = """
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
	      module <a href="sphinx/algorithms.html"><code>algorithms.py</code></a>
	      contains generic optimization algorithms. The
	      module <a href="sphinx/optimization.html"><code>optimization.py</code></a>
	      contains the functions that can be called directly by
	      Biogeme [<a href="sphinx/biogeme.html?highlight=estimate#biogeme.biogeme.BIOGEME.estimate">Click here for the documentation of
	      the <code>estimate</code>
	      function</a>]. [<a href="examples/swissmetro/01logit_allAlgos.py">Click
	      here for an example.</a>]</dd>
	    <dt>CFSQP</dt>
	    <dd>The CFSQP algorithm has been removed from the
	    distribution. </dd>
	    <dt>Null log likelihood</dt>
	    <dd>The log likelihood is calculated. The null model
	      predicts equal probability for each alternative.</dd>
	    <dt>Saved iterations</dt>
	    <dd>Iterations are saved in a file with
	      extension <code>.iter</code>. If the file exists, Biogeme
	      will initialize the parameters from this files, and
	      ignore the starting values provided. To turn this feature
	      off, set <code>biogeme.saveIterations=False</code></dd>
	    <dt>Random starting values</dt>
	    <dd>It is possible to modify the initial values of the parameters in all formulas,
              using randomly generated values. The value is drawn from a
              uniform distribution on the interval defined by the
              bounds (by default [-100, 100].)
              [<a href="sphinx/biogeme.html?highlight=setrandominitvalue#biogeme.biogeme.BIOGEME.setRandomInitValues">Click
              here for the documentation</a>].
	      <dt>Sensitivity analysis</dt>
	    <dd>The betas for sensitivity analysis are now generated
	      by bootstrapping.
	      [<a href="sphinx/results.html?highlight=sensitivityanalysis#biogeme.results.bioResults.getBetasForSensitivityAnalysis">Click
	      here for the documentation</a>].</dd>
	    <dt>Box-Cox</dt>
	    <dd>The implementation of
	      the <a href="sphinx/models.html?highlight=boxcox#biogeme.models.boxcox">Box-Cox
		transform</a> was  incorrect and has been corrected.
	      <dt>Validation</dt>
	    <dd>The out-of-sample validation has been
	      improved. [<a href="sphinx/biogeme.html?highlight=validate#biogeme.biogeme.BIOGEME.validate">Click
	      here for the documentation</a>]. It
	      has to be compined with the <a href="sphinx/database.html?highlight=split#biogeme.database.Database.split"><code>split</code></a> function
	      of the database object.
	      <dt>Statistics about chosen alternatives</dt>
	    <dd>It is now possible to calculate the number of time
	      each alternative is chosen and available in the
	      sample. [<a href="sphinx/database.html?highlight=choiceavailability#biogeme.database.Database.choiceAvailabilityStatistics">Click
	      here for the documentation</a>].</dd>
	    <dt>Validity check for the nests</dt>
	    <dd> The validity of the specification of the nests
	      for nested and cross nested logit models is new
	      checked.</dd>
	    <dt>ALOGIT file</dt>
	    <dd>Output files in F12 format compatible with ALOGIT can
	      now be
	      produced. [<a href="sphinx/results.html?highlight=f12#biogeme.results.bioResults.writeF12">Click
	      here for the documentation</a>. </dd>
	    <dt>Likelihood ratio test</dt>
	    <dd>A function to perform the likelihood ratio test has
	      been
	      implemented. [<a href="sphinx/tools.html?highlight=likelihood_ratio#biogeme.tools.likelihood_ratio_test">Click
	      here for the documentation</a>].</dd>
	  </dl>

"""

faq['What was new in Biogeme 3.2.6?'] = """
	  <dl>
	    <dt>Optimization</dt>
	    <dd>New optimization algorithms are
	      available for estimation See the documentation of
	      the <a href="sphinx/biogeme.html?highlight=algorithm#biogeme.biogeme.BIOGEME.estimate"><code>estimate</code></a>
	      function, and
	      the <a href="sphinx/biogeme.html#module-biogeme.optimization"><code>optimization</code>
		module</a>. See also an
	      <a href="examples/swissmetro/01logit_allAlgos.py">example.</a></dd>
	    <dt>Stochastic log likelihood</dt>
	    <dd>It is now possible to calculate the log likelihood
	      function on a sample (a batch) of the full data
	      file. This is particularly useful with large
	      databases. It can be used in the implementation of a
	      stochastic gradient algorithm, for instance. See <a href="sphinx/biogeme.html?highlight=calculatelikelihood#biogeme.biogeme.BIOGEME.calculateLikelihoodAndDerivatives">documentation</a>.</dd>
	    <dt>User's notes</dt>
	    <dd>It is possible to include your own notes in the HTML
	      file using the <code>userNotes</code> parameter of the
	      <code>biogeme</code>
	      object. See <a href="sphinx/biogeme.html?highlight=usernotes#biogeme.biogeme.BIOGEME"
			     target="_blank">documentation</a>. See
	      <a href="examples/swissmetro/01logitBis.py">example</a>.</dd>
	    <dt>Scaling</dt>
	    <dd>It is possible to have Biogeme suggesting the scales
	      of the variables in the database using
	      the <code>suggestScales</code> parameter of
	      the <code>biogeme</code>
	      object. See <a href="sphinx/biogeme.html?highlight=suggestscales#biogeme.biogeme.BIOGEME.__init__"
			     target="_blank">documentation</a>.</dd>
	    <dt>Estimation</dt>
	    <dd>A new function <code>quickEstimate</code> performs
	      the estimation of the parameters, and skips the
	      calculation of the
	      statistics. See <a href="sphinx/biogeme.html?highlight=quickestimate#biogeme.biogeme.BIOGEME.quickEstimate">documentation</a>. </dd>
	    <dt>Validation</dt>
	    <dd>A new function in the <code>database</code> module allows to split the database in order to
	      prepare an estimation and a validation sets, for
	      out-of-sample
	      validation. See <a href="sphinx/biogeme.html?highlight=split#biogeme.database.Database.split">documentation</a>. It
	      is used by the new function <code>validate</code> in the
	      <code>biogeme</code> module. See <a href="sphinx/biogeme.html?highlight=validate#biogeme.biogeme.BIOGEME.validate">documentation</a>. See <a href="examples/swissmetro/b04validation.py">example</a>.</dd>
	    <dt>Messages</dt>
	    <dd>A new function allows to extract all the messages
	      generated during a
	      run. See <a href="sphinx/biogeme.html?highlight=allmessages#biogeme.messaging.bioMessage.allMessages">documentation</a>. See
	      <a href
		 ="https://github.com/michelbierlaire/biogeme/blob/master/examples/notebooks/biogeme.messaging.ipynb">example</a>. It
	      is also possible to make the logger temporarily silent
	      using the functions <a href="sphinx/biogeme.html?highlight=temporarysilent#biogeme.messaging.bioMessage.temporarySilence"><code>temporarySilence</code></a> and <a href="sphinx/biogeme.html?highlight=resume#biogeme.messaging.bioMessage.resume"><code>resume</code>.</a></dd>
	  </dl>

"""

faq[
    'Why is the file headers.py not generated?'
] = """
<p>In order to comply better with good programming practice in
Python, the syntax to import the variable names from the data
file has been modified since version 3.2.5. The file
<code>headers.py</code> is not generated anymore.
The best practice is to declare every variable explicity:
</p>
<p>
<pre>
PURPOSE = Variable('PURPOSE')
CHOICE = Variable('CHOICE')
GA = Variable('GA')
TRAIN_CO = Variable('TRAIN_CO')
CAR_AV = Variable('CAR_AV')
SP = Variable('SP')
TRAIN_AV = Variable('TRAIN_AV')
TRAIN_TT = Variable('TRAIN_TT')
</pre>
</p>
<p>
If, for any reason, this explicit declaration is not
desired, it is possible to replace the statement
</p>
<p><code>from headers import *</code></p>
<p>by</p>
<p><code>
globals().update(database.variables)
</code>
</p>
<p>where <code>database</code>  is the object containing the
database, created as follows:</p>
<p>  <code>
import biogeme.database as db<br>
df = pd.read_csv('swissmetro.dat', '\\t')<br>
database = db.Database('swissmetro', df)
</code></p>
<p>Also, in order to avoid any ambiguity, the operators used by
Biogeme must be explicitly imported. For instance:</p>
<p>
<code>
from biogeme.expressions import Beta, bioDraws, PanelLikelihoodTrajectory, MonteCarlo, log
</code>
</p>
<p>
Note that it is also possible to import all of them using the
following syntax</p>
<p>  <code>
from biogeme.expressions import *
</code></p>
<p>  although this is not a good Python programming practice. </p>
"""
faq[
    'What initial values should I select for the parameters?'
] = """
If you have the results of a previous estimation, it may be
a good idea to use the estimated values as a starting point
for the estimation of similar models. If not, it depends
on the nature of the parameters:
<ul>
<li>If the parameter is a coefficient (traditionally
denoted by &beta;), the value 0 is
appropriate.</li>
<li>If the parameter is a nest parameter of a nested or
cross-nested logit model (traditionally
denoted by &mu;), the value 1 is
appropriate. Make sure to define the lower bound of the
parameter to 1.</li>
<li>If the parameter is the nest membership coefficient of
a cross-nested logit model (traditionally
denoted by &alpha;), the value 0.5 is
appropriate. Make sure to define the lower bound to 0 and
the upper bound to 1.</li>
<li>If the parameter captures the membership to a class of
a latent class model, the value 0.5 is appropriate. Make
sure to define the lower bound to 0 and
the upper bound to 1.</li>
<li>If the parameter is the scale of an error component in
a mixture of logit model (traditionally
denoted by &sigma;), the value must be sufficient
large so that the likelihood of each observation is not
too close to zero. It is suggested to try first with the
value one. If there are numerical issues, try a larger
value, such as 10. See Section 7 in the report
<a href="http://transp-or.epfl.ch/documents/technicalReports/Bier18b.pdf" target="_blank">
Estimating choice models  with latent variables
with PandasBiogeme</a> for a detailed discussion. </li>

</ul>
"""
faq[
    'Can I save intermediate iterations during the estimation?'
] = """
<p>
Yes. It is actually the default behavior. At each
iteration, Biogeme creates a
file <code>__myModel.iter</code>. This file will be read the
next time Biogeme tries to estimate the same model. If you want to turn this
feature off, set the BIOGEME class
variable <code>saveIterations</code> to <code>False</code>.
</p>
"""
faq[
    'Does Biogeme provide support for out-of-sample validation?'
] = """
<p>
Yes.  See
example <code><a href="https://github.com/michelbierlaire/biogeme/blob/master/examples/swissmetro/b04validation.py">b04validation.py</a> on Github.
"""

faq[
    'The init loglikelihood is <code>-1.797693e+308</code> and '
    'no iteration is performed. What should I do?'
] = """
<p>If the model returns a probability 0 for the chosen
alternative for at least one observation in the sample, then
the likelihood is 0, and the log likelihood is minus
infinity. For the sake of robustness, Biogeme assigns the
value <code>-1.797693e+308</code> to the log likelihood in
this context. 
</p>
<p>A possible reason is when the initial value of a scale
parameter is too close to zero. 
</p>
<p>But there are many other possible reasons. The best way
to investigate the source of the problem is to use Biogeme
in simulation mode, and report the probability of the chosen
alternative for each observation. Once you have identified
the problematic entries, it is easier to investigate the
reason why the model returns a probability of zero.
</p>
"""

faq['ImportError: DLL load failed while importing cythonbiogeme: The specified module could not be found'] = """
The issue is that in Python 3.8 and older on Windows, DLLs are loaded from trusted locations only  (<a href ="https://docs.python.org/3/whatsnew/3.8.html#ctypes" target="_blank">see this</a>).  It is necessary to add the path of the DLLs. Here is a way proposed by Facundo Storani, University of Salerno:
<ul>
<li>Search the DLLs folder of anaconda3. It may be similar to: <code>C:\\Users\\[USER_NAME]\\anaconda3\\DLLs or C:\\ProgramData\\Anaconda3\\DLLs</code>.</li>
<li>Click the Start button, type "environment properties" into the search bar and hit Enter. </li>
<li>In the System Properties window, click "Environment Variables." </li>
<li>Select "Path" on the users' list, and modify. </li>
<li>Add the path of the dlls folder to the list. It may be similar to:
<code>C:\\Users\\[USER_NAME]\\anaconda3\\DLLs</code> or <code>C:\\ProgramData\\Anaconda3\\DLLs</code></code>.</li>
</ul> (credit: Facundo Storani)
"""

faq['Why is Cython library not found?'] = """
On Mac OSX, the
following error is sometimes
generated: <pre>
ImportError:
dlopen(/Users/~/anaconda3/lib/python3.6/site-packages/biogeme/cbiogeme.cpython-36m-darwin.so,
2): Symbol not found:
__ZNSt15__exception_ptr13exception_ptrD1Ev
</pre>
<p>It
is likely to be due to a conflict of versions of Python
packages. The
best way to deal with it is to reinstall Biogeme using the
following steps:
<ul>
<li>First, make sure that you have the latest version of pip:
<pre>
pip install --upgrade pip
</pre>
</li>
<li>Uninstall biogeme:
<pre>
pip uninstall biogeme
</pre>
</li>
<li>Install cython:
<pre>
pip install —-upgrade cython
</pre>
</li>
<li>Reinstall biogeme, without using the cache:
<pre>
pip install biogeme -—no-cache-dir
</pre>
</ul>
If it does not work, try first to install gcc:
<pre>
conda install gcc
</pre>
If it does not work, try creating a new conda environment: 
<pre>
conda create -n python310 python=3.10 pip
conda activate python310
pip install biogeme
</pre>
If it does not work... I don't know :-(
"""

faq['Why is it trying to compile during installation?'] = """
On Mac OSX and Windows, the procedure is designed to install
from binaries, not sources. If you get messages that look like the
following, it means that pip is trying to compile from sources. And
it will most certainly fail as the environment must be properly configured.
<pre>
Running setup.py install for biogeme ... error
Complete output from command
c:\\users\\willi\\anaconda3\\python.exe -u -c "import setuptools,
tokenize;
__file__='C:\\Users\\willi\\AppData\\Local\\Temp\\pip-install-iaflhasr\\biogeme\\setup.py';
f=getattr(tokenize, 'open', open)(__file__);
code=f.read().replace('\\r\\n', '\\n');
f.close();
exec(compile(code, __file__, 'exec'))" install --record C:\\Users\\willi\\AppData\\Local\\Temp\\pip-record-v6_zn0ff\\install-record.txt --single-version-externally-managed --compile:
Using Cython
Please put "# distutils: language=c++" in your .pyx or .pxd file(s)
running install
</pre>
It means that there is no binaries available for your version of
Python. To check which versions are supported, go to the repository
<p>
<a href="https://pypi.org/project/biogeme/">pypi.org/project/biogeme/</a>
</p>
<p>For instance, the following files are available for version 3.2.10:
<pre>biogeme-3.2.10.tar.gz</pre>
<pre>biogeme-3.2.10-cp310-cp310-win_amd64.whl</pre>
<pre>biogeme-3.2.10-cp310-cp310-macosx_10_9_x86_64.whl</pre>
<pre>biogeme-3.2.10-cp39-cp39-win_amd64.whl</pre>
<pre>biogeme-3.2.10-cp39-cp39-macosx_10_9_x86_64.whl</pre>
<pre>biogeme-3.2.10-cp38-cp38-win_amd64.whl</pre>
<pre>biogeme-3.2.10-cp38-cp38-macosx_10_9_x86_64.whl</pre>
<pre>biogeme-3.2.10-cp37-cp37m-win_amd64.whl</pre>
<pre>biogeme-3.2.10-cp37-cp37m-macosx_10_9_x86_64.whl</pre>
<pre>biogeme-3.2.10-cp36-cp36m-macosx_10_9_x86_64.whl</pre>
</ul>
  It means that you can use Python 3.7, 3.8 and 3.9 on both platforms,
while the version for Python 3.6 is only available on MacOSX.
</p>
"""
