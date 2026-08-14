faq = {}

faq['Why is the file headers.py not generated?'] = """
<p>In order to comply better with good programming practice in
Python, the syntax to import the variable names from the data
file has been modified since version 3.2.5. The file
<code>headers.py</code> is not generated anymore.
The best practice is to declare every variable explicitly:
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
<p>  although this is not recommended. </p>
"""
faq['What initial values should I select for the parameters?'] = """
If you have the results of a previous estimation, it may be
a good idea to use the estimated values as a starting point
for the estimation of similar models. If not, it depends
on the nature of the parameters:
<ul>
<li>If the parameter is a coefficient (traditionally
denoted by &Beta;), the value 0 is
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
a latent_old class model, the value 0.5 is appropriate. Make
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
Estimating choice models  with latent_old variables
with PandasBiogeme</a> for a detailed discussion. </li>
</ul>
Note that if a file <code>__mymodel.iter</code> exists, where <code>mymodel</code> is the name of the model
to be estimated, the initial values of the parameters are read from this file.
"""
faq['How do I cancel the calculation of second derivatives during estimation?'] = """
<p>Change the optimization algorithm to '<code>simple_bounds_BFGS</code>'. It can be done in the .toml file, or
in the biogeme object: </p>
<p><code>biogeme.optimization_algorithm = 'simple_bounds_BFGS'</code></p>
"""
faq['Can I save intermediate iterations during the estimation?'] = """
<p>
Yes. It is actually the default behavior. At each
iteration, Biogeme creates a
file <code>__myModel.iter</code>. This file will be read the
next time Biogeme tries to estimate the same model. If you want to turn this
feature off, set the parameter <code>save_iterations</code> to <code>False</code> in the <code>biogeme.toml</code>
file. See the <a href="sphinx/code/toml.html">documentation</a> for details.
</p>
"""
faq['Does Biogeme provide support for out-of-sample validation?'] = """
<p>
Yes.  See
<a href="sphinx/auto_examples/swissmetro/plot_b04_validation.html">this example</a>.
"""

faq[
    'The init loglikelihood is <code>-inf</code> and '
    'no iteration is performed. What should I do?'
] = """
<p>If the model returns a probability 0 for the chosen
alternative for at least one observation in the sample, then
the likelihood is 0, and the log likelihood is minus
infinity.
</p>
<p>A possible reason is when the initial value of a scale
parameter is too close to zero.
</p>
<p>There may be several other reasons for the issue. The most effective method to identify the problem’s source is to
use Biogeme in simulation mode and report the probability of the chosen alternative for each observation. Once the
problematic entries are identified, it becomes easier to investigate why the model returns a probability of zero.
</p>
"""

faq[
    'ImportError: DLL load failed while importing cythonbiogeme: The specified module could not be found'
] = """
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
best way to deal with it is to reinstall Biogeme in a clean environment using the
following steps:
<ul>
<li>Leave the current environment by typing <code>deactivate</code>.</li>
<li>Create a new environment:
<pre>
virtualenv -p python3.12 env_biogeme
</pre>
</li>
<li>Activate the new environment. On MacOSx:
<pre>
source env_biogeme/bin/activate
</pre>
On Windows:
<pre>
.\\env_biogeme\\Scripts\\activate
</pre>
</li>
<li>Make sure that you have the latest version of pip:
<pre>
pip install --upgrade pip
</pre>
or
<pre>
python -m pip install --upgrade pip
</pre>
</li>
<li>Install biogeme:
<pre>
pip install biogeme
</pre>
</ul>
"""

faq[
    'Why is it trying to compile during installation when installing CythonBiogeme?'
] = """
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
<a href="https://pypi.org/project/cythonbiogeme/">pypi.org/project/cythonbiogeme/</a>
</p>
<p>For instance, the following files are available for CythonBiogeme 1.0.4:

<pre>cythonbiogeme-1.0.4.tar.gz</pre>
<pre>cythonbiogeme-1.0.4-cp312-cp312-win_amd64.whl</pre>
<pre>cythonbiogeme-1.0.4-cp312-cp312-macosx_10_9_universal2.whl</pre>
<pre>cythonbiogeme-1.0.4-cp311-cp311-win_amd64.whl</pre>
<pre>cythonbiogeme-1.0.4-cp311-cp311-macosx_10_9_universal2.whl</pre>
<pre>cythonbiogeme-1.0.4-cp310-cp310-win_amd64.whl</pre>
<pre>cythonbiogeme-1.0.4-cp310-cp310-macosx_10_9_universal2.whl</pre>
</ul>
  It means that you can use Python 3.10, 3.11 and 3.12 on both platforms.
</p>
"""
