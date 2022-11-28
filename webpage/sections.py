"""
Dictionary. Key: title of the card. Value: list of paragraphs.
"""
about = {}
install = {}
documentation = {}
archives = {}
resources = {}

about['Biogeme'] = (
"""
Biogeme is a open
source <a href="https://www.python.org/"
target="_blank">Python</a> package designed for the
maximum likelihood estimation of parametric models
in general, with a special emphasis on discrete
choice models. It relies on the
package <a href="https://pandas.pydata.org/"
target="_blank">Python Data Analysis Library called
Pandas</a>.
""",
"""
It is developed and maintained by <a href="https://people.epfl.ch/michel.bierlaire"
target="_blank">Prof. Michel
Bierlaire</a>, <a href="https://transp-or.epfl.ch/"
targert="_blank">Transport and Mobility
Laboratory</a>, <a href="https://www.epfl.ch" target="_blank">Ecole
Polytechnique F&eacute;d&eacute;rale de Lausanne</a>, Switzerland.
""",
"""
Biogeme used to be a stand alone software package,
written in C++. All the material related to the previous
versions of Biogeme are available on
the <a href="http://transp-or.epfl.ch/pythonbiogeme/"
target="_blank">old webpage</a>.
"""
)

about['What\'s new in Biogeme 3.2.10?'] = (
"""
<strong>Note</strong>: versions 3.2.9 and 3.2.10 are
identical. Therefore, version 3.2.9 has been removed from
	  the official distribution platform.
""",
"""    
<dl>
<dt>New syntax for <samp>DefineVariable</samp></dt>
<dd>
<p><samp>DefineVariable</samp> actually defines a new column in the
database. The old syntax was:
<pre>
myvar = DefineVariable('myvar', x * y + 2, database)
</pre>
The new syntax is:
<pre>
myvar = database.DefineVariable('myvar', x * y + 2)
</pre>
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
population. See the example <a href="examples/swissmetro/01logitBis.py"><samp>01logitBis.py</samp></a>.</dd>
<dt>Simulation of panel data</dt>
<dd>It is now possible to use Biogeme in simulation mode
for panel data. See the following
example: <a href="examples/swissmetro/13panel_simul.py"><samp>13panel_simul.py</samp></a>.</dd>
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
parameter <samp>recycle=True</samp>. See the online
documentation [<a href="sphinx/biogeme.html#biogeme.biogeme.BIOGEME.estimate">here</a>].</dd>
<dt>The feature removing unused variables has been
canceled.</dt>
<dd>The parameters <samp>removeUnusedVariables</samp>
and <samp>displayUsedVariables</samp> in the BIOGEME
constructor have been removed.</dd>
<dt>More functionalities for the mathematical expressions.</dt>
<dd>The expressions have now been designed to also be
available outside of the BIOGEME class. A detailed
illustration of the functionalities is available
[<a href="https://github.com/michelbierlaire/biogeme/blob/master/examples/notebooks/biogeme.expressions.ipynb">Click
here</a>].</dd>
<dt>New syntax for the assisted specification algorithm</dt>
<dd>The new syntax involves <samp>NamedTuple</samp> to make the code
more readable. Refer to the examples, such as
<a href="examples/assisted/optima.py"
target="_blank"><samp>optima.py</samp></a>.</dd>

</dl>
""",
)

about['Conditions of use'] = (
"""
BIOGEME is distributed free of charge. We ask each user
<ul>
<li>to register
to <a href="https://groups.google.com/d/forum/biogeme"
target="_blank">Biogeme users group</a>, and </li>
<li>to mention explicitly the use of the package when
publishing results, using the following reference:
<p><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier20.pdf"
target="_blank">Bierlaire, M. (2020). A
short introduction to
PandasBiogeme. Technical report TRANSP-OR
200605. Transport and Mobility Laboratory,
ENAC, EPFL.</a>
""",
)

about['Author'] = (
"""
 Biogeme has been developed by
	    <a href="http://people.epfl.ch/michel.bierlaire"
	       target="_blank">Michel
	      Bierlaire</a>, <a href="http://www.epfl.ch" target="_blank">Ecole Polytechnique
	      F&eacute;d&eacute;rale de Lausanne</a>, Switzerland.
""",
)


about['Acknowledgments'] = (
"""
I would like to thank the following persons who played various
roles in the development of Biogeme along the years. The list is
certainly not complete, and I apologize for those who are omitted:
	    Alexandre Alahi,
	    Nicolas Antille,
	    Gianluca Antonini,
	    Cristian Arteaga,
	    Kay Axhausen,
	    John Bates,
	    Denis Bolduc,
	    David Bunch,
	    Andrew Daly,
	    Anna Fernandez Antolin,
	    Mamy Fetiarison,
	    Mogens Fosgerau,
	    Emma Frejinger,
	    Carmine Gioia,
	    Marie-H&eacute;l&egrave;ne Godbout,
	    Stephane Hess,
	    Tim Hillel,
	    Richard Hurni,
	    Eva Kazagli,
	    Jasper Knockaert,
	    Xinjun Lai,
	    Gael Lederrey,
	    Virginie Lurkin,
	    Nicholas Molyneaux,
	    Nicola Ortelli,
	    Carolina Osorio,
	    Meritxell Pacheco Paneque,
	    Thomas Robin,
	    Pascal Scheiben,
	    Matteo Sorci,
	    Ewout ter Hoeven,
	    Michael Th&eacute;mans,
	    Joan Walker.
""",
"""
I would like to give special thanks to Moshe Ben-Akiva
and Daniel McFadden for their friendship, and for the immense
influence that they had and still have on my work.
"""
)

install['Install Python'] = (
"""
Biogeme is an open source Python package, that relies on the version 3
of Python. Make sure that Python 3.x is installed on your
computer. If you have never used Python before, you may
want to consider a complete platform such
as <a href="https://www.anaconda.com/" target="_blank">Anaconda</a>.
""",
"""
If Python is already installed on your computer, verify
the version. Two versions of Python are distributed:
version 2 and version 3. Biogeme  works only
with version 3.
"""
)

install['Installing PandasBiogeme on MaxOSX'] = (
"""
<p class='text-center'>
<object width="425" height="344"><param name="movie" value="http://www.youtube.com/v/Swg1FUK-QaU&hl=en&fs=1"></param><param name="allowFullScreen" value="true"></param><embed src="http://www.youtube.com/v/Swg1FUK-QaU&hl=en&fs=1" type="application/x-shockwave-flash" allowfullscreen="true" width="425" height="344"></embed></object></p>
""",
)

install['Installing PandasBiogeme on Windows'] = (
"""
<p class=text-center>
<object width="425" height="344"><param name="movie" value="http://www.youtube.com/v/1TlNrhS2rFo&hl=en&fs=1"></param><param name="allowFullScreen" value="true"></param><embed src="http://www.youtube.com/v/1TlNrhS2rFo&hl=en&fs=1" type="application/x-shockwave-flash" allowfullscreen="true" width="425" height="344"></embed></object>
</p>
""",
)

install['Install Biogeme from pip'] = (
"""
Biogeme is distributed using the
<a href="https://pypi.org/project/pip/" target="_blank">pip</a>
package manager. There are several tutorials available on the internet
such
as <a href="https://packaging.python.org/tutorials/installing-packages/"
target="_blank">this one</a> or
<a href="https://www.youtube.com/watch?v=U2ZN104hIcc"
target="_blank">this one</a>.
""",
"""
The command to install is simply
<pre>
pip install biogeme
</pre>
""",
"""
Depending on your OS and the version of
Python, pip will either directly install
the executable (it is called a "wheel"), or attempt to compile the package from
sources.
""",
"""
In the latter case, it requires a proper
environment to compile C++ code. In general, it is readily available on Linux, and
MacOSX (if <a href="https://developer.apple.com/xcode/"
target="_blank">Xcode</a> has been installed). It may be more
complicated on Windows.
"""
)

install['Biogeme on Github'] = (
"""
The source code of Biogeme is available on
<a href="https://github.com/michelbierlaire/biogeme" target="_blank">GitHub</a>.
There are several tutorials available on the internet
such
as <a href="https://guides.github.com/activities/hello-world/"
target="_blank">this one</a> or
<a href="https://youtu.be/HkdAHXoRtos"
target="_blank">this one</a>.
""",
"""
The command to install Biogeme from source is
<pre>pip install -ve .</pre>
""",
"""
that must be executed in the directory containing the
file setup.py.
""",
"""
Note that it requires a proper
environment to compile C++ code. In general, it is readily available on Linux, and
MacOSX (if <a href="https://developer.apple.com/xcode/"
target="_blank">Xcode</a> has been
installed).
""",
"""
On Windows,
<ol>
<li>Install <a href="https://www.msys2.org">MSYS2</a>.</li>
<li>Add c:\\msys64\\mingw64\\bin in the Windows PATH.</li>
<li>Install using the following command:
<pre>pip install --global-option build_ext --global-option --compiler=mingw32</pre></p></li>
"""
)

install['Check the installation'] = (
"""
To verify if biogeme is correctly installed, you can print
the version of Biogeme. To do so,  execute the
following commands in Python:
<ul>
<li>Import the package: <pre>import biogeme.version as ver</pre></li>
<li>Print the version information: <pre>print(ver.getText())</pre>
</ul>
The result should look like the following:
<pre>
Python 3.10.4 (main, Mar 31 2022, 03:38:35) [Clang 12.0.0 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import biogeme.version as ver
>>> print(ver.getText())
biogeme 3.2.9 [2022-08-19]
Version entirely written in Python
Home page: http://biogeme.epfl.ch
Submit questions to https://groups.google.com/d/forum/biogeme
Michel Bierlaire, Transport and Mobility Laboratory, Ecole Polytechnique Fédérale de Lausanne (EPFL)
</pre>
""",
)

documentation['My first choice model with PandasBiogeme'] = (
"""
<p class='text-center'>
<object width="425" height="344">
<param name="movie" value="https://youtu.be/vS-Sg0htQP4"></param>
<param name="allowFullScreen" value="true"></param>
<embed src="https://youtu.be/vS-Sg0htQP4" type="application/x-shockwave-flash" allowfullscreen="true" width="425" height="344"></embed></object></p>
""",
)


documentation['Code documentation'] = (
"""
<a href="sphinx/index.html" target="_blank">Click here for the documentation of the source of Biogeme</a>. It has been
generated with the <a href="http://www.sphinx-doc.org"
target="_blank">Python Documentation Generator Sphinx</a>.
""",
)

documentation['Technical reports'] = (
"""
The following technical reports will walk through concrete examples to
get familiar with the software.
""",
"""
<ul>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/Bier20.pdf"
target="_blank">A short introduction to PandasBiogeme</a>.</li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf" target="_blank">Calculating indicators with PandasBiogeme.</a></li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier19.pdf"
target="_blank">Monte-Carlo integration with PandasBiogeme.</a></li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier18b.pdf"
target="_blank">Estimating choice models with latent
variables with PandasBiogeme.</a></li>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/BierOrte22.pdf" target="_blank">Assisted specification with Biogeme.</a></li>
</ul>
""",
)


resources['Videos'] = (
"""
<a href="https://youtu.be/vS-Sg0htQP4" target="_blank">
My first choice model with PandasBiogeme
</a>
""",
"""
<a href="https://youtu.be/GmNodW_pUHk" target="_blank">
To be or not to be significant
</a>
""",
"""
<a href="https://youtu.be/_w7RxZIUBqI" target="_blank">
Survival of the fittest... or not
</a>
""",
"""
<a href="https://youtu.be/hZsdMpNC-30" target="_blank">
Aggregate elasticities
</a>
"""
)

resources['EPFL Winter Course'] = (
"""
<a href="https://transp-or-academia.epfl.ch/dca" target="_blank">Click here for information about the course</a>
""",
"""
EPFL proposes a 5-day short course entitled <em>"Discrete Choice
Analysis: Predicting Individual Behavior and Market Demand"</em>. It is
organized every year in March (occasionally in February).
""",
"""
Content:
<ol>
<li><em>Fundamental methodology</em>, e.g. the foundations of individual choice modeling, random utility models, discrete choice models (binary, multinomial, nested, cross-nested logit models, MEV models, probit models, and hybrid choice models such as logit kernel and mixed logit);</li>
<li><em>Data collection issues</em>, e.g. choice-based samples, enriched samples, stated preferences surveys, conjoint analysis, panel data;</li>
<li><em>Model design issues</em>, e.g. specification of utility functions, generic and alternative specific variables, joint discrete/continuous models, dynamic choice models;</li>
<li><em>Model estimation issues</em>, e.g. statistical estimation,
testing procedures, software packages, estimation with individual
and grouped data, Bayesian estimation;</li>
<li><em>Forecasting techniques</em>, e.g. aggregate predictions, sample enumeration, micro-simulation, elasticities, pivot-point predictions and transferability of parameters;</li>
<li><em>Examples and case studies</em>, including marketing (e.g., brand choice), housing (e.g., residential location), telecommunications (e.g., choice of residential telephone service), energy (e.g., appliance type), transportation (e.g., mode of travel).</li>
</ol>
""",
"""
<table>
<tr valign="top"><td>Lecturers:</td><td><a href="https://cee.mit.edu/people_individual/moshe-e-ben-akiva/"
target="_blank">Prof. Moshe
Ben-Akiva</a></td><td>Massachusetts Institute of Technology,
Cambridge, Ma (USA)</td></tr>
<tr valign="top"><td></td><td><a href="https://priceschool.usc.edu/daniel-mcfadden/"
target="_blank">Prof. Daniel McFadden</a></td><td>University of
Southern California [Nobel Prize Laureate, 2000]</td></tr>
<tr valign="top"><td></td><td><a href="https://people.epfl.ch/michel.bierlaire"
target="_blank">Prof. Michel
Bierlaire</a></td><td>Ecole Polytechnique
F&eacute;d&eacute;rale de Lausanne, Switzerland</td></tr>
</table>
"""
)

resources['Online courses'] = (
"""
An online course entitled "Introduction to Discrete Choice
Models" is available on the following platforms:
<ul>
<li><a href="https://courseware.epfl.ch/courses/course-v1:EPFL+ChoiceModels+2020/about"
target="_blank">EPFL
Courseware</a></li>
<li><a href="https://www.edx.org/course/introduction-to-discrete-choice-models"
target="_blank">edX.org</a></li>
</ul>
""",
)

resources['MIT Summer Course'] = (
"""
<a href="http://professional.mit.edu/programs/short-programs/discrete-choice-analysis"
	     target="_blank">
Click here for information about the course
</a>
""",
"""
MIT proposes a 5-day short course entitled <em>"Discrete Choice
Analysis: Predicting demand and market shares"</em>. It is
organized every year in June.
""",
"""
Lecturer: <a href="https://cee.mit.edu/people_individual/moshe-e-ben-akiva/"
target="_blank">Prof. Moshe
Ben-Akiva</a>, Massachusetts Institute of Technology,
Cambridge, Ma (USA)
"""
)

resources['Other software packages'] = (
"""
<dl>
<dt><a href="https://rdrr.io/cran/mixl" target="_blank">mixl</a></dt>
<dd>
Simulated Maximum Likelihood Estimation of Mixed Logit
Models for Large Datasets, by Joseph Malloy
</dd>
<dt><a href="https://larch.newman.me" target="_blank">LARCH</a></dt>
<dd>LARCH: A Freeware Package for Estimating Discrete Choice Models, by
Jeffrey Newman.</dd>
<dt><a href="http://www.apollochoicemodelling.com" target="_blank">Apollo</a></dt>
<dd>
Apollo: a flexible, powerful and customisable freeware
package for choice model estimation and application, by Stephane Hess
and David Palma.
</dd>
<dt><a href="https://github.com/timothyb0912/pylogit"
	     target="_blank">Pylogit</a></dt>
<dd>
PyLogit is a Python package developed by Timothy Brathwaite for
performing maximum likelihood estimation of conditional logit models
and similar discrete choice models.
</dd>
</dl>      
""",
)

archives['PandasBiogeme Version 3'] = (
"""
The releases of PandasBiogeme are available on the  <a href="https://pypi.org/project/biogeme/" target="_blank">Python
Package Index</a> repository.
""",
"""
Previous webpages:
<ul>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.8/"
target="blank">Webpage for Pandasbiogeme 3.2.8</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.6/"
target="blank">Webpage for Pandasbiogeme 3.2.6</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.5/"
target="blank">Webpage for Pandasbiogeme 3.2.5</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.2/"
target="blank">Webpage for Pandasbiogeme 3.2.2</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.1/"
target="blank">Webpage for Pandasbiogeme 3.2.1</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.1.0/"
target="blank">Webpage for Pandasbiogeme 3.1.0</a></li>
</ul>
"""
)


archives['PythonBiogeme Version 2.5'] = (
"""
<ul>
<li><a href="https://transp-or.epfl.ch/biogeme-2.5/home.html"
target="_blank"> Webpage</a>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v25/biogeme-2.5.tar.gz">biogeme-2.5.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v25/examples-v25-bison.zip">examples-v25-bison.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v25/examples-v25-python.zip">examples-v25-python.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v25/PythonBiogeme-2.5-WindowsInstaller.exe">PythonBiogeme-2.5-WindowsInstaller.exe</a></li>

</ul>
""",
)

archives['PythonBiogeme Version 2.4'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v24/biogeme-2.4.tar.gz">biogeme-2.4.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v24/examples-v24.zip">examples-v24.zip</a></li>
</ul>
""",
)

archives['PythonBiogeme Version 2.3'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v23/biogeme-2.3.tar.gz">biogeme-2.3.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v23/examples-v23.zip">examples-v23.zip</a></li>
</ul>
""",
)

archives['PythonBiogeme Version 2.2'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v22/biogeme-2.2.tar.gz">biogeme-2.2.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v22/biogeme-v22.zip">biogeme-v22.zip</a></li>
</ul>
""",
)

archives['PythonBiogeme Version 2.1'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v21/biogeme-2.1.mpkg.zip">biogeme-2.1.mpkg.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v21/biogeme-2.1beta.tar.gz">biogeme-2.1beta.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v21/biogeme-2.1gamma.tar.gz">biogeme-2.1gamma.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v21/examples-2.1beta.zip">examples-2.1beta.zip</a></li>
</ul>
""",
)

archives['PythonBiogeme Version 2.0'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v20/biogeme-2.0.tar.gz">biogeme-2.0.tar.gz</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v20/biogeme-v20-macosx.zip">biogeme-v20-macosx.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v20/biogeme-v20.zip">biogeme-v20.zip</a></li>
</ul>
""",
)

archives['PythonBiogeme Version 1.8'] = (
"""
<ul>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v18/biogeme-v18-macosx.zip">biogeme-v18-macosx.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v18/biogeme-v18-src.zip">biogeme-v18-src.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v18/biogeme-v18.zip">biogeme-v18.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v18/examples-v18.zip">examples-v18.zip</a></li>
<li><a href="http://transp-or.epfl.ch/pythonbiogeme/archives/v18/tutorialv18.pdf">tutorialv18.pdf</a></li>
</ul>
""",
)
