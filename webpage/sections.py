"""
Dictionary. Key: title of the card. Value: list of paragraphs.
"""

special = {}
about = {}
install = {}
documentation = {}
archives = {}
resources = {}


special['New release'] = 'The new version of Biogeme is substantially faster. Try it.'
# special['Users meeting'] = 'Biogeme users\' meeting, September 5, 2023, Zurich, Switzerland, from 10:00 to 13:00. <a href="https://transp-or-academia.epfl.ch/biogeme">Click here to register.</a>'
# special['EPFL Short Course'] = (
#    'Discrete Choice Analysis: Predicting Individual Behavior and Market Demand. January 26-30, 2025 <a href="https://transp-or-academia.epfl.ch/dca">Click here to register.</a>'
# )
# special['Other special'] = 'Other special as well'

# special['ChatGPT'] = (
#    'Try the Biogeme Assistant <a href="https://chatgpt.com/g/g-mArtaAszx-biogeme-assistant" '
#    'target="_blank">[Click here]</a> (credits: Yousef Maknoon)'
# )

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
""",
)

about['Links'] = (
    """
<ul>
<li><a href="https://pypi.org/project/biogeme/">PyPi Distribution</a></li>
<li> <a href="https://www.pepy.tech/projects/Biogeme">Download statistics</a></li>
<li> <a href="https://groups.google.com/d/forum/biogeme">Users' group </a></li>
</ul>
""",
)

about['What\'s new in Biogeme 3.3.0?'] = (
    """
<p>
  In this major release, arithmetic expressions and their derivatives are no longer evaluated using <code>cythonbiogeme</code>.
  Instead, Biogeme now relies on 
  <a href="https://docs.jax.dev" target="_blank" rel="noopener noreferrer">JAX</a>, 
  a high-performance numerical computing library for Python.
</p>

<p>
  This transition required substantial changes to the underlying codebase. A large portion had to be re-implemented from scratch. 
  Although the release has undergone extensive testing, some issues may still persist. 
  If you encounter any problems, please report them on the 
  <a href="https://groups.google.com/d/forum/biogeme" target="_blank" rel="noopener noreferrer">user group</a>.
</p>

<p>
  To the extent possible, the user interface, namely the syntax for model specification, remains consistent with previous versions. 
  As a result, we anticipate only minor adjustments will be needed in existing code. 
  However, some updates may be necessary.
</p>

<p>
  The impact on computation time is substantial. Preliminary tests on the Swissmetro dataset show the following improvements:
</p>
<p>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Model</th>
      <th>Computation</th>
      <th>Speedup (3.3.0 vs 3.2.14)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logit</td>
      <td>Function only</td>
      <td>60.0×</td>
    </tr>
    <tr>
      <td>Logit</td>
      <td>Function + Gradient</td>
      <td>164.8×</td>
    </tr>
    <tr>
      <td>Logit</td>
      <td>Function + Gradient + Hessian</td>
      <td>54.7×</td>
    </tr>
    <tr>
      <td>CNL</td>
      <td>Function only</td>
      <td>120.0×</td>
    </tr>
    <tr>
      <td>CNL</td>
      <td>Function + Gradient</td>
      <td>42.7×</td>
    </tr>
    <tr>
      <td>CNL</td>
      <td>Function + Gradient + Hessian</td>
      <td>2.7×</td>
    </tr>
    <tr>
      <td>Mixtures_100</td>
      <td>Function only</td>
      <td>29.0×</td>
    </tr>
    <tr>
      <td>Mixtures_100</td>
      <td>Function + Gradient</td>
      <td>89.2×</td>
    </tr>
    <tr>
      <td>Mixtures_100</td>
      <td>Function + Gradient + Hessian</td>
      <td>13.2×</td>
    </tr>
  </tbody>
</table>
</p>
""",
)

about['Conditions of use'] = (
    """
<p>BIOGEME is distributed free of charge. We ask each user
<ul>
<li>to register
to <a href="https://groups.google.com/d/forum/biogeme"
target="_blank">Biogeme users group</a>, and </li>
<li>to mention explicitly the use of the package when
publishing results, using the following reference:
<p><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier23.pdf"
target="_blank">Bierlaire, M. (2023). A
short introduction to
Biogeme. Technical report TRANSP-OR
230620. Transport and Mobility Laboratory,
ENAC, EPFL.</a></p></li></ul>
""",
)

about['Disclaimer'] = (
    """
<p><strong>Disclaimer</strong> This software is provided free of charge and "AS
IS" WITHOUT ANY WARRANTY of any kind. The implied
warranties of merchantability, fitness for a
particular purpose and non-infringement are expressly
disclaimed. In no event will the
author (Michel Bierlaire) or his employer (EPFL) be
liable to any party for any direct, indirect, special
or other consequential damages for any use of the
code including, without limitation, any lost
profits, business interruption, loss of programs or
other data on your information handling system or
otherwise, even if we are expressly advised of the
possibility of such damages.</p>
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
        Pedro Camargo,
	    Andrew Daly,
        Nicolas Dubois,
	    Anna Fernandez Antolin,
	    Mamy Fetiarison,
	    Mogens Fosgerau,
	    Emma Frejinger,
	    Carmine Gioia,
	    Marie-H&eacute;l&egrave;ne Godbout,
        Jason Hawkins,
	    Stephane Hess,
	    Tim Hillel,
	    Richard Hurni,
	    Eva Kazagli,
	    Jasper Knockaert,
	    Xinjun Lai,
	    Gael Lederrey,
	    Virginie Lurkin,
	    Yousef Maknoon,
	    Nicholas Molyneaux,
	    Nicola Ortelli,
	    Carolina Osorio,
	    Meritxell Pacheco Paneque,
	    Thomas Robin,
	    Pascal Scheiben,
	    Matteo Sorci,
	    Ewout ter Hoeven,
	    Michael Th&eacute;mans,
	    Joan Walker,
        Mengyi Wang.
""",
    """
I would like to give special thanks to Moshe Ben-Akiva
and Daniel McFadden for their friendship, and for the immense
influence that they had and still have on my work.
""",
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
""",
)

install['Installing PandasBiogeme on MaxOSX'] = (
    """
<p class='text-center'>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Z1hkeWP0k9A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>
""",
)

install['Installing Biogeme on Windows'] = (
    """
<p class=text-center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/-P6zXrcodGs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
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
)

install['CythonBiogeme on Github'] = (
    """
A significant part of Biogeme is coded in C++ for the sake of computational efficiency. Since version 3.2.11, this part
of the code has been isolated in a separate package called <samp>cythonbiogeme</samp>. Binaries for Mac OSX and Windowns 
are available for versions of Python ranging from 3.10 to 3.12. If, for some reasons, the binary distribution for your 
system is not available, pip will attempt to compile the package from sources.
In that case, it requires a proper
environment to compile C++ code. In general, it is readily available on Linux, and
MacOSX (if <a href="https://developer.apple.com/xcode/"
target="_blank">Xcode</a> has been installed). It may be more
complicated on Windows.
""",
    """
The source code of CythonBiogeme is available on
<a href="https://github.com/michelbierlaire/cythonbiogeme" target="_blank">GitHub</a>.
There are several tutorials available on the internet
such
as <a href="https://guides.github.com/activities/hello-world/"
target="_blank">this one</a> or
<a href="https://youtu.be/HkdAHXoRtos"
target="_blank">this one</a>.
""",
    """
The command to install CythonBiogeme from source is
<pre>pip install .</pre>
""",
    """
that must be executed in the root directory, containing the
file setup.py.
""",
    """
Note that it requires a proper
environment to compile C++ code. In general, it is readily available on Linux, and
MacOSX (if <a href="https://developer.apple.com/xcode/"
target="_blank">Xcode</a> has been
installed). On Windows, it is possible to compile cythonbiogeme with Microsoft Visual C++. 
See the <a href="https://wiki.python.org/moin/WindowsCompilers">Python documentation</a>.
""",
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
<pre>pip install .</pre>
""",
    """
that must be executed in the root directory containing the
pyproject.toml file.
""",
    """
Note that it does not require to compile C++ code (thanks to CythonBiogeme) and should be working in any environment 
where Python and CythonBiogeme are properly installed.
""",
)

install['Check the installation'] = (
    """
To verify if biogeme is correctly installed, you can print
the version of Biogeme. To do so,  execute the
following commands in Python:
<ul>
<li>Import the package: <pre>from biogeme.version import get_text</pre></li>
<li>Print the version information: <pre>print(get_text())</pre>
</ul>
The result should look like the following:
<pre>
Python 3.12.4 (v3.12.4:8e8a4baf65, Jun  6 2024, 17:33:18) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from biogeme.version import get_text
>>> print(get_text())
biogeme 3.3.0 [2025-08-01]
Home page: http://biogeme.epfl.ch
Submit questions to https://groups.google.com/d/forum/biogeme
Michel Bierlaire, Transport and Mobility Laboratory, Ecole Polytechnique Fédérale de Lausanne (EPFL)
</pre>
""",
)

documentation['Code documentation'] = (
    """
<a href="sphinx/index.html" target="_blank">Click here for the online documentation</a>. It has been
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
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/Bier23.pdf"
target="_blank">A short introduction to Biogeme</a>.</li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf" target="_blank">Calculating indicators with PandasBiogeme.</a></li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier19.pdf"
target="_blank">Monte-Carlo integration with PandasBiogeme.</a></li>
<li><a href="http://transp-or.epfl.ch/documents/technicalReports/Bier18b.pdf"
target="_blank">Estimating choice models with latent
variables with PandasBiogeme.</a></li>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf" target="_blank">Assisted specification with Biogeme 3.2.12.</a></li>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/BierPasc23.pdf" target="_blank">Estimating MEV models with samples of alternatives.</a></li>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/Bier24.pdf" target="_blank">Arithmetic expressions in Biogeme.</a></li>
<li><a href="https://transp-or.epfl.ch/documents/technicalReports/BierWang24.pdf" target="_blank">The MDCEV model with Biogeme: estimation and forecasting.</a></li>
</ul>
""",
)


documentation['Preparing data for Biogeme'] = (
    """
<p class='text-center'>
<iframe width="560" height="315" src="https://www.youtube.com/embed/lhbpra2dILA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>

""",
)


documentation['Estimating my first choice model with Biogeme'] = (
    """
<p class='text-center'>
<iframe width="560" height="315" src="https://www.youtube.com/embed/jIAIsqh_g0E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>
""",
)


resources['Videos'] = (
    """
<a href="https://youtu.be/Z1hkeWP0k9A" target="_blank">
Installing Biogeme 3.2.11 on Mac OSX
</a>
""",
    """
<a href="https://youtu.be/-P6zXrcodGs" target="_blank">
Installing Biogeme 3.2.11 on Windows
</a>
""",
    """
<a href="https://youtu.be/lhbpra2dILA" target="_blank">
Preparing data for Biogeme
</a>
""",
    """
<a href="https://youtu.be/jIAIsqh_g0E" target="_blank">
Estimating my first choice model with Biogeme
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
""",
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
""",
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
""",
)

resources['University of Sydney'] = (
    """
<a href="https://www.sydney.edu.au/business/our-research/institute-of-transport-and-logistics-studies/courses/discrete-choice-analysis.html">
Click here for information about the course
</a>
""",
    """
The University of Sydney Business School offers a course taught by Prof. David Hensher, Prof. Michiel Bliemer, Prof. John Rose and Dr. Andrew Collins.
""",
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
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.12/"
target="blank">Webpage for Pandasbiogeme 3.2.12</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.11/"
target="blank">Webpage for Pandasbiogeme 3.2.11</a></li>
<li><a href="https://transp-or.epfl.ch/biogeme-3.2.10/"
target="blank">Webpage for Pandasbiogeme 3.2.10</a></li>
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
""",
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
