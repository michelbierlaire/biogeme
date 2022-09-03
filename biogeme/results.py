"""
Implementation of class contaning and processing the estimation results.

:author: Michel Bierlaire
:date: Tue Mar 26 16:50:01 2019

.. todo:: rawResults should be a dict and not a class.
"""

# Too constraining
# pylint: disable=invalid-name,
# pylint: disable=too-many-instance-attributes, too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements, too-few-public-methods
# pylint: disable=too-many-lines
#

import pickle
import datetime
from collections import namedtuple
import urllib.request as urlr
import pandas as pd
import numpy as np
from scipy import linalg
from scipy import stats
import biogeme.version as bv
import biogeme.filenames as bf
import biogeme.exceptions as excep
import biogeme.messaging as msg
from biogeme import tools

logger = msg.bioMessage()

GeneralStatistic = namedtuple('GeneralStatistic', 'value format')


def calcPValue(t):
    """Calculates the p value of a parameter from its t-statistic.

    The formula is

    .. math:: 2(1-\\Phi(|t|)

    where :math:`\\Phi(\\cdot)` is the CDF of a normal distribution.

    :param t: t-statistics
    :type t: float

    :return: p-value
    :rtype: float
    """
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    return p


class beta:
    """Class gathering the information related to the parameters
    of the model
    """

    def __init__(self, name, value, bounds):
        """
        Constructor

        :param name: name of the parameter.
        :type name: string
        :param value: value of the parameter.
        :type value: float
        :param bounds: tuple (l,b) with lower and upper bounds
        :type bounds: float,float
        """

        self.name = name  #: Name of the parameter

        self.value = value  #: Current value

        self.lb = bounds[0]  #: Lower bound

        self.ub = bounds[1]  #: Upper bound

        self.stdErr = None  #: Standard error

        self.tTest = None  #: t-test

        self.pValue = None  #: p-value

        self.robust_stdErr = None  #: Robust standard error

        self.robust_tTest = None  #: Robust t-test

        self.robust_pValue = None  #: Robust p-value

        self.bootstrap_stdErr = None  #: Std error calculated from bootstrap

        self.bootstrap_tTest = None  #: t-test calculated from bootstrap

        self.bootstrap_pValue = None  #: p-value calculated from bootstrap

    def isBoundActive(self, threshold=1.0e-6):
        """Check if one of the two bound is 'numerically' active. Being
        numerically active means that the distance between the value of
        the parameter and one of its bounds is below the threshold.

        :param threshold: distance below which the bound is considered to be
            active. Default: :math:`10^{-6}`
        :type threshold: float

        :return: True is one of the two bounds is numericall y active.
        :rtype: bool

        :raise biogemeError: if ``threshold`` is negative.

        """
        if threshold < 0:
            raise excep.biogemeError(
                f'Threshold ({threshold}) must be non negative'
            )

        if self.lb is not None and np.abs(self.value - self.lb) <= threshold:
            return True
        if self.ub is not None and np.abs(self.value - self.ub) <= threshold:
            return True
        return False

    def setStdErr(self, se):
        """Records the standard error, and calculates and records
        the corresponding t-statistic and p-value

        :param se: standard error.
        :type se: float

        """
        self.stdErr = se
        if se == 0:
            self.tTest = np.finfo(float).max
        else:
            self.tTest = np.nan_to_num(self.value / se)
        self.pValue = calcPValue(self.tTest)

    def setRobustStdErr(self, se):
        """Records the robust standard error, and calculates and records
        the corresponding t-statistic and p-value


        :param se: robust standard error
        :type se: float

        """
        self.robust_stdErr = se
        if se == 0:
            self.robust_tTest = np.finfo(float).max
        else:
            self.robust_tTest = np.nan_to_num(self.value / se)
        self.robust_pValue = calcPValue(self.robust_tTest)

    def setBootstrapStdErr(self, se):
        """Records the robust standard error calculated by bootstrap, and
        calculates and records the corresponding t-statistic and p-value

        :param se: standard error calculated by bootstrap.
        :type se: float
        """
        self.bootstrap_stdErr = se
        if se == 0:
            self.bootstrap_tTest = np.finfo(float).max
        else:
            self.bootstrap_tTest = np.nan_to_num(self.value / se)
        self.bootstrap_pValue = calcPValue(self.robust_tTest)

    def __str__(self):
        s = f'{self.name:15}: {self.value:.3g}'
        if self.stdErr is not None:
            s += f'[{self.stdErr:.3g} {self.tTest:.3g} {self.pValue:.3g}]'
        if self.robust_stdErr is not None:
            s += (
                f'[{self.robust_stdErr:.3g} {self.robust_tTest:.3g} '
                f'{self.robust_pValue:.3g}]'
            )
        if self.bootstrap_stdErr is not None:
            s += (
                f'[{self.bootstrap_stdErr:.3g} {self.bootstrap_tTest:.3g} '
                f'{self.bootstrap_pValue:.3g}]'
            )
        return s


class rawResults:
    """Class containing the raw results from the estimation"""

    def __init__(self, theModel, betaValues, fgHb, bootstrap=None):
        """
        Constructor

        :param theModel: object with the model
        :type theModel: biogeme.BIOGEME
        :param betaValues: list containing the estimated values of the
            parameters
        :type betaValues: list(float)
        :param fgHb: tuple f,g,H,bhhh containing

                 - f: the value of the function,
                 - g: the gradient,
                 - H: the second derivative matrix,
                 - bhhh: the BHHH matrix.
        :type fgHb: float,numpy.array, numpy.array, numpy.array

        :param bootstrap: output of the bootstrapping. numpy array, of
            size B x K,  where

            - B is the number of bootstrap iterations
            - K is the number of parameters to estimate

            Default: None.
        :type bootstrap: numpy.array
        """

        self.modelName = theModel.modelName  #: Name of the model

        self.userNotes = theModel.userNotes  #: User notes

        self.nparam = len(betaValues)  #: Number of parameters

        self.betaValues = betaValues  #: Values of the parameters

        self.betaNames = (
            theModel.id_manager.free_betas.names
        )  #: Names of the parameters

        self.initLogLike = theModel.initLogLike
        """Value of the likelihood function with the initial value of the
        parameters
        """

        self.nullLogLike = theModel.nullLogLike
        """Value of the likelihood function with equal probability model
        """

        self.betas = []  #: List of objects of type results.beta

        for b, n in zip(betaValues, self.betaNames):
            bounds = theModel.getBoundsOnBeta(n)
            self.betas.append(beta(n, b, bounds))

        self.logLike = fgHb[0]  #: Value of the loglikelihood function

        self.g = fgHb[1]  #: Value of the gradient of the loglik. function

        self.H = fgHb[2]  #: Value of the hessian of the loglik. function

        self.bhhh = fgHb[3]
        """Value of the BHHH matrix of the loglikelihood function"""

        self.dataname = theModel.database.name  #: Name of the database

        self.sampleSize = theModel.database.getSampleSize()
        """Sample size (number of individuals if panel data)"""

        self.numberOfObservations = theModel.database.getNumberOfObservations()
        """Number of observations"""

        self.monteCarlo = theModel.monteCarlo
        """True if the model involved Monte Carlo integration"""

        self.numberOfDraws = theModel.numberOfDraws
        """Number of draws for Monte Carlo integration"""

        self.typesOfDraws = theModel.database.typesOfDraws
        """Types of draws for Monte Carlo integration"""

        self.excludedData = theModel.database.excludedData
        """Number of excluded data"""

        self.drawsProcessingTime = theModel.drawsProcessingTime
        """Time needed to process the draws"""

        self.gradientNorm = linalg.norm(self.g) if self.g is not None else None
        """Norm of the gradient"""

        self.optimizationMessages = theModel.optimizationMessages
        """Diagnostics given by the optimization algorithm"""

        self.numberOfThreads = theModel.numberOfThreads
        """Number of threads used for parallel computing"""

        self.htmlFileName = None  #: Name of the HTML output file

        self.F12FileName = None  #: Name of the F12 output file

        self.latexFileName = None  #: Name of the LaTeX output file

        self.pickleFileName = None  #: Name of the pickle outpt file

        self.bootstrap = bootstrap
        """output of the bootstrapping. numpy array, of size B x K,
        where

        - B is the number of bootstrap iterations
        - K is the number of parameters to estimate
        """

        if bootstrap is not None:
            self.bootstrap_time = theModel.bootstrap_time
            """ Time needed to perform the bootstrap"""

        self.secondOrderTable = None  #: Second order statistics


class bioResults:
    """Class managing the estimation results"""

    def __init__(self, theRawResults=None, pickleFile=None):
        """Constructor

        :param theRawResults: object with the results of the estimation.
            Default: None.
        :type theRawResults: biogeme.results.rawResults
        :param pickleFile: name of the file containing the raw results in
            pickle format. It can be a URL. Default: None.
        :type pickleFile: string

        :raise biogeme.exceptions.biogemeError: if no data is provided.
        """

        if theRawResults is not None:
            self.data = theRawResults
            """Object of type :class:`biogeme.results.rawResults` contaning the
            raw estimation results.
            """
        elif pickleFile is not None:
            try:
                with urlr.urlopen(pickleFile) as p:
                    self.data = pickle.load(p)
            except Exception:
                pass
            try:
                with open(pickleFile, 'rb') as f:
                    self.data = pickle.load(f)
            except FileNotFoundError as e:
                error_msg = f'File {pickleFile} not found'
                raise excep.biogemeError(error_msg) from e

        else:
            raise excep.biogemeError('No data provided.')

        self._calculateStats()

    def writePickle(self):
        """Dump the data in a file in pickle format.

        :return: name of the file.
        :rtype: string
        """
        self.data.pickleFileName = bf.getNewFileName(
            self.data.modelName, 'pickle'
        )
        with open(self.data.pickleFileName, 'wb') as f:
            pickle.dump(self.data, f)

        logger.general(f'Results saved in file {self.data.pickleFileName}')
        return self.data.pickleFileName

    def _calculateTest(self, i, j, matrix):
        """Calculates a t-test comparing two coefficients

        Args:
           i: index of first coefficient \f$\\beta_i\f$.
           j: index of second coefficient \f$\\beta_i\f$.
           matrix: estimate of the variance-covariance matrix \f$m\f$.

        :return: t test
            ..math::  \f[\\frac{\\beta_i-\\beta_j}
                  {\\sqrt{m_{ii}+m_{jj} - 2 m_{ij} }}\f]
        :rtype: float

        """
        vi = self.data.betaValues[i]
        vj = self.data.betaValues[j]
        varI = matrix[i, i]
        varJ = matrix[j, j]
        covar = matrix[i, j]
        r = varI + varJ - 2.0 * covar
        if r <= 0:
            test = np.finfo(float).max
        else:
            test = (vi - vj) / np.sqrt(r)
        return test

    def _calculateStats(self):
        """Calculates the following statistics:

        - likelihood ratio test between the initial and the estimated
          models: :math:`-2(L_0-L^*)`
        - Rho square: :math:`1 - \\frac{L^*}{L^0}`
        - Rho bar square: :math:`1 - \\frac{L^* - K}{L^0}`
        - AIC: :math:`2(K - L^*)`
        - BIC: :math:`-2 L^* + K  \\log(N)`

        Estimates for the variance-covariance matrix (Rao-Cramer,
        robust, and bootstrap) are also calculated, as well as t-tests and
        p value for the comparison of pairs of coefficients.

        """
        self.data.likelihoodRatioTestNull = (
            -2.0 * (self.data.nullLogLike - self.data.logLike)
            if self.data.nullLogLike is not None
            else None
        )
        self.data.likelihoodRatioTest = (
            -2.0 * (self.data.initLogLike - self.data.logLike)
            if self.data.initLogLike is not None
            else None
        )
        try:
            self.data.rhoSquare = (
                np.nan_to_num(1.0 - self.data.logLike / self.data.initLogLike)
                if self.data.initLogLike is not None
                else None
            )
        except ZeroDivisionError:
            self.data.rhoSquare = None
        try:
            self.data.rhoSquareNull = (
                np.nan_to_num(1.0 - self.data.logLike / self.data.nullLogLike)
                if self.data.nullLogLike is not None
                else None
            )
        except ZeroDivisionError:
            self.data.rhoSquareNull = None
        try:
            self.data.rhoBarSquare = (
                np.nan_to_num(
                    1.0
                    - (self.data.logLike - self.data.nparam)
                    / self.data.initLogLike
                )
                if self.data.initLogLike is not None
                else None
            )
        except ZeroDivisionError:
            self.data.rhoBarSquare = None
        try:
            self.data.rhoBarSquareNull = (
                np.nan_to_num(
                    1.0
                    - (self.data.logLike - self.data.nparam)
                    / self.data.nullLogLike
                )
                if self.data.nullLogLike is not None
                else None
            )
        except ZeroDivisionError:
            self.data.rhoBarSquareNull = None

        self.data.akaike = 2.0 * self.data.nparam - 2.0 * self.data.logLike
        self.data.bayesian = (
            -2.0 * self.data.logLike
            + self.data.nparam * np.log(self.data.sampleSize)
        )
        # We calculate the eigenstructure to report in case of singularity
        if self.data.H is not None:
            self.data.eigenValues, self.data.eigenVectors = linalg.eigh(
                -np.nan_to_num(self.data.H)
            )
            _, self.data.singularValues, _ = linalg.svd(
                -np.nan_to_num(self.data.H)
            )
            # We use the pseudo inverse in case the matrix is singular
            self.data.varCovar = -linalg.pinv(np.nan_to_num(self.data.H))
            for i in range(self.data.nparam):
                if self.data.varCovar[i, i] < 0:
                    self.data.betas[i].setStdErr(np.finfo(float).max)
                else:
                    self.data.betas[i].setStdErr(
                        np.sqrt(self.data.varCovar[i, i])
                    )

            d = np.diag(self.data.varCovar)
            if (d > 0).all():
                diag = np.diag(np.sqrt(d))
                diagInv = linalg.inv(diag)
                self.data.correlation = diagInv.dot(
                    self.data.varCovar.dot(diagInv)
                )
            else:
                self.data.correlation = np.full_like(
                    self.data.varCovar, np.finfo(float).max
                )

            # Robust estimator
            self.data.robust_varCovar = self.data.varCovar.dot(
                self.data.bhhh.dot(self.data.varCovar)
            )
            for i in range(self.data.nparam):
                if self.data.robust_varCovar[i, i] < 0:
                    self.data.betas[i].setRobustStdErr(np.finfo(float).max)
                else:
                    self.data.betas[i].setRobustStdErr(
                        np.sqrt(self.data.robust_varCovar[i, i])
                    )
            rd = np.diag(self.data.robust_varCovar)
            if (rd > 0).all():
                diag = np.diag(np.sqrt(rd))
                diagInv = linalg.inv(diag)
                self.data.robust_correlation = diagInv.dot(
                    self.data.robust_varCovar.dot(diagInv)
                )
            else:
                self.data.robust_correlation = np.full_like(
                    self.data.robust_varCovar, np.finfo(float).max
                )

            # Bootstrap
            if self.data.bootstrap is not None:
                self.data.bootstrap_varCovar = np.cov(
                    self.data.bootstrap, rowvar=False
                )
                for i in range(self.data.nparam):
                    if self.data.bootstrap_varCovar[i, i] < 0:
                        self.data.betas[i].setBootstrapStdErr(
                            np.finfo(float).max
                        )
                    else:
                        self.data.betas[i].setBootstrapStdErr(
                            np.sqrt(self.data.bootstrap_varCovar[i, i])
                        )
                rd = np.diag(self.data.bootstrap_varCovar)
                if (rd > 0).all():
                    diag = np.diag(np.sqrt(rd))
                    diagInv = linalg.inv(diag)
                    self.data.bootstrap_correlation = diagInv.dot(
                        self.data.bootstrap_varCovar.dot(diagInv)
                    )
                else:
                    self.data.bootstrap_correlation = np.full_like(
                        self.data.bootstrap_varCovar, np.finfo(float).max
                    )

            self.data.secondOrderTable = {}
            for i in range(self.data.nparam):
                for j in range(i):
                    t = self._calculateTest(i, j, self.data.varCovar)
                    p = calcPValue(t)
                    trob = self._calculateTest(i, j, self.data.robust_varCovar)
                    prob = calcPValue(trob)
                    if self.data.bootstrap is not None:
                        tboot = self._calculateTest(
                            i, j, self.data.bootstrap_varCovar
                        )
                        pboot = calcPValue(tboot)
                    name = (self.data.betaNames[i], self.data.betaNames[j])
                    if self.data.bootstrap is not None:
                        self.data.secondOrderTable[name] = [
                            self.data.varCovar[i, j],
                            self.data.correlation[i, j],
                            t,
                            p,
                            self.data.robust_varCovar[i, j],
                            self.data.robust_correlation[i, j],
                            trob,
                            prob,
                            self.data.bootstrap_varCovar[i, j],
                            self.data.bootstrap_correlation[i, j],
                            tboot,
                            pboot,
                        ]
                    else:
                        self.data.secondOrderTable[name] = [
                            self.data.varCovar[i, j],
                            self.data.correlation[i, j],
                            t,
                            p,
                            self.data.robust_varCovar[i, j],
                            self.data.robust_correlation[i, j],
                            trob,
                            prob,
                        ]

            eigIndex = np.argmin(self.data.eigenValues)
            self.data.smallestEigenValue = self.data.eigenValues[eigIndex]
            self.data.smallestEigenVector = self.data.eigenVectors[:, eigIndex]
            self.data.smallestSingularValue = min(self.data.singularValues)
            eigIndex = np.argmax(self.data.eigenValues)
            self.data.largestEigenValue = self.data.eigenValues[eigIndex]
            self.data.largestEigenVector = self.data.eigenVectors[:, eigIndex]
            self.data.largestSingularValue = max(self.data.singularValues)
            self.data.conditionNumber = (
                self.data.largestEigenValue / self.data.smallestEigenValue
            )

    def shortSummary(self):
        """Provides a short summary of the estimation results"""
        r = ''
        r += f'Results for model {self.data.modelName}\n'
        r += f'Nbr of parameters:\t\t{self.data.nparam}\n'
        r += f'Sample size:\t\t\t{self.data.sampleSize}\n'
        if self.data.sampleSize != self.data.numberOfObservations:
            r += f'Observations:\t\t\t{self.data.numberOfObservations}\n'
        r += f'Excluded data:\t\t\t{self.data.excludedData}\n'
        if self.data.nullLogLike is not None:
            r += f'Null log likelihood:\t\t{self.data.nullLogLike:.7g}\n'
        r += f'Final log likelihood:\t\t{self.data.logLike:.7g}\n'
        if self.data.nullLogLike is not None:
            r += (
                f'Likelihood ratio test (null):\t\t'
                f'{self.data.likelihoodRatioTestNull:.7g}\n'
            )
            r += f'Rho square (null):\t\t\t{self.data.rhoSquareNull:.3g}\n'
            r += (
                f'Rho bar square (null):\t\t\t'
                f'{self.data.rhoBarSquareNull:.3g}\n'
            )
        r += f'Akaike Information Criterion:\t{self.data.akaike:.7g}\n'
        r += f'Bayesian Information Criterion:\t{self.data.bayesian:.7g}\n'
        return r

    def __str__(self):
        r = '\n'
        r += f'Results for model {self.data.modelName}\n'
        if self.data.htmlFileName is not None:
            r += f'Output file (HTML):\t\t\t{self.data.htmlFileName}\n'
        if self.data.latexFileName is not None:
            r += f'Output file (LaTeX):\t\t\t{self.data.latexFileName}\n'
        r += f'Nbr of parameters:\t\t{self.data.nparam}\n'
        r += f'Sample size:\t\t\t{self.data.sampleSize}\n'
        if self.data.sampleSize != self.data.numberOfObservations:
            r += f'Observations:\t\t\t{self.data.numberOfObservations}\n'
        r += f'Excluded data:\t\t\t{self.data.excludedData}\n'
        if self.data.nullLogLike is not None:
            r += f'Null log likelihood:\t\t{self.data.nullLogLike:.7g}\n'
        if self.data.initLogLike is not None:
            r += f'Init log likelihood:\t\t{self.data.initLogLike:.7g}\n'
        r += f'Final log likelihood:\t\t{self.data.logLike:.7g}\n'
        if self.data.nullLogLike is not None:
            r += (
                f'Likelihood ratio test (null):\t\t'
                f'{self.data.likelihoodRatioTestNull:.7g}\n'
            )
            r += f'Rho square (null):\t\t\t{self.data.rhoSquareNull:.3g}\n'
            r += (
                f'Rho bar square (null):\t\t\t'
                f'{self.data.rhoBarSquareNull:.3g}\n'
            )
        if self.data.initLogLike is not None:
            r += (
                f'Likelihood ratio test (init):\t\t'
                f'{self.data.likelihoodRatioTest:.7g}\n'
            )
            r += f'Rho square (init):\t\t\t{self.data.rhoSquare:.3g}\n'
            r += f'Rho bar square (init):\t\t\t{self.data.rhoBarSquare:.3g}\n'
        r += f'Akaike Information Criterion:\t{self.data.akaike:.7g}\n'
        r += f'Bayesian Information Criterion:\t{self.data.bayesian:.7g}\n'
        if self.data.gradientNorm is not None:
            r += f'Final gradient norm:\t\t{self.data.gradientNorm:.7g}\n'
        r += '\n'.join([f'{b}' for b in self.data.betas])
        r += '\n'
        if self.data.secondOrderTable is not None:
            for k, v in self.data.secondOrderTable.items():
                r += (
                    '{}:\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t'
                    '{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\n'
                ).format(k, *v)
        return r

    def _getLaTeXHeader(self):
        """Prepare the header for the LaTeX file, containing comments and the
        version of Biogeme.

        :return: string containing the header.
        :rtype: str
        """
        h = ''
        h += '%% This file is designed to be included into a LaTeX document\n'
        h += (
            '%% See http://www.latex-project.org for '
            'information about LaTeX\n'
        )
        h += (
            f'%% {self.data.modelName} - Report from '
            f'biogeme {bv.getVersion()} '
            f'[{bv.versionDate}]\n'
        )

        h += bv.getLaTeX()
        return h

    def getLaTeX(self, onlyRobust=True):
        """Get the results coded in LaTeX

        :param onlyRobust: if True, only the robust statistics are included
        :type onlyRobust: bool

        :return: LaTeX code
        :rtype: string
        """
        now = datetime.datetime.now()
        h = self._getLaTeXHeader()
        if self.data.latexFileName is not None:
            h += '\n%% File ' + self.data.latexFileName + '\n'
        h += f'\n%% This file has automatically been generated on {now}</p>\n'
        if self.data.dataname is not None:
            h += f'\n%%Database name: {self.data.dataname}\n'

        if self.data.userNotes is not None:
            # User notes
            h += f'\n%%{self.data.userNotes}\n'

        h += '\n%% General statistics\n'
        h += '\\section{General statistics}\n'
        d = self.getGeneralStatistics()
        h += '\\begin{tabular}{ll}\n'
        for k, (v, p) in d.items():
            if isinstance(v, bytes):
                v = str(v)
            if isinstance(v, str):
                v = v.replace('_', '\\_')
            h += f'{k} & {v:{p}} \\\\\n'
        for k, v in self.data.optimizationMessages.items():
            if k in ('Relative projected gradient', 'Relative change'):
                h += f'{k} & \\verb${v:.7g}$ \\\\\n'
            else:
                h += f'{k} & \\verb${v}$ \\\\\n'
        h += '\\end{tabular}\n'

        h += '\n%%Parameter estimates\n'
        h += '\\section{Parameter estimates}\n'
        table = self.getEstimatedParameters(onlyRobust)

        def formatting(x):
            """Defines the formatting for the to_latex function of pandas"""
            res = f'{x:.3g}'
            if '.' in res:
                return res

            return f'{res}.0'

        ## Need to check for old versions of Pandas.
        try:
            h += table.style.format(formatting).to_latex()
        except AttributeError:
            h += table.to_latex(float_format=formatting)

        h += '\n%%Correlation\n'
        h += '\\section{Correlation}\n'
        table = self.getCorrelationResults()
        ## Need to check for old versions of Pandas.
        try:
            h += table.style.format(formatting).to_latex()
        except AttributeError:
            h += table.to_latex(float_format=formatting)

        return h

    def getGeneralStatistics(self):
        """Format the results in a dict

        :return: dict with the results. The keys describe each
                 content. Each element is a GeneralStatistic tuple,
                 with the value and its preferred formatting.

        Example::

                     'Init log likelihood': (-115.30029248549191, '.7g')

        :rtype: dict(string:float,string)

        """
        d = {}
        d['Number of estimated parameters'] = GeneralStatistic(
            value=self.data.nparam, format=''
        )
        nf = self.numberOfFreeParameters()
        if nf != self.data.nparam:
            d['Number of free parameters'] = GeneralStatistic(
                value=nf, format=''
            )
        d['Sample size'] = GeneralStatistic(
            value=self.data.sampleSize, format=''
        )
        if self.data.sampleSize != self.data.numberOfObservations:
            d['Observations'] = GeneralStatistic(
                value=self.data.numberOfObservations, format=''
            )
        d['Excluded observations'] = GeneralStatistic(
            value=self.data.excludedData, format=''
        )
        if self.data.nullLogLike is not None:
            d['Null log likelihood'] = GeneralStatistic(
                value=self.data.nullLogLike, format='.7g'
            )
        d['Init log likelihood'] = GeneralStatistic(
            value=self.data.initLogLike, format='.7g'
        )
        d['Final log likelihood'] = GeneralStatistic(
            value=self.data.logLike, format='.7g'
        )
        if self.data.nullLogLike is not None:
            d['Likelihood ratio test for the null model'] = GeneralStatistic(
                value=self.data.likelihoodRatioTestNull,
                format='.7g',
            )
            d['Rho-square for the null model'] = GeneralStatistic(
                value=self.data.rhoSquareNull, format='.3g'
            )
            d['Rho-square-bar for the null model'] = GeneralStatistic(
                value=self.data.rhoBarSquareNull,
                format='.3g',
            )
        d['Likelihood ratio test for the init. model'] = GeneralStatistic(
            value=self.data.likelihoodRatioTest,
            format='.7g',
        )
        d['Rho-square for the init. model'] = GeneralStatistic(
            value=self.data.rhoSquare, format='.3g'
        )
        d['Rho-square-bar for the init. model'] = GeneralStatistic(
            value=self.data.rhoBarSquare, format='.3g'
        )
        d['Akaike Information Criterion'] = GeneralStatistic(
            value=self.data.akaike, format='.7g'
        )
        d['Bayesian Information Criterion'] = GeneralStatistic(
            value=self.data.bayesian, format='.7g'
        )
        d['Final gradient norm'] = GeneralStatistic(
            value=self.data.gradientNorm, format='.4E'
        )
        if self.data.monteCarlo:
            d['Number of draws'] = GeneralStatistic(
                value=self.data.numberOfDraws, format=''
            )
            d['Draws generation time'] = GeneralStatistic(
                value=self.data.drawsProcessingTime, format=''
            )
            d['Types of draws'] = GeneralStatistic(
                value=[f'{i}: {k}' for i, k in self.data.typesOfDraws.items()],
                format='',
            )
        if self.data.bootstrap is not None:
            d['Bootstrapping time'] = GeneralStatistic(
                value=self.data.bootstrap_time, format=''
            )
        d['Nbr of threads'] = GeneralStatistic(
            value=self.data.numberOfThreads, format=''
        )
        return d

    def printGeneralStatistics(self):
        """Print the general statistics of the estimation.

        :return: general statistics

            Example::

                Number of estimated parameters:	2
                Sample size:	5
                Excluded observations:	0
                Init log likelihood:	-67.08858
                Final log likelihood:	-67.06549
                Likelihood ratio test for the init. model:	0.04618175
                Rho-square for the init. model:	0.000344
                Rho-square-bar for the init. model:	-0.0295
                Akaike Information Criterion:	138.131
                Bayesian Information Criterion:	137.3499
                Final gradient norm:	3.9005E-07
                Bootstrapping time:	0:00:00.042713
                Nbr of threads:	16


        :rtype: str
        """
        d = self.getGeneralStatistics()
        output = ''
        for k, (v, p) in d.items():
            output += f'{k}:\t{v:{p}}\n'
        return output

    def numberOfFreeParameters(self):
        """This is the number of estimated parameters, minus those that are at
        their bounds
        """
        return sum([not b.isBoundActive() for b in self.data.betas])

    def getEstimatedParameters(self, onlyRobust=True):
        """Gather the estimated parameters and the corresponding statistics in
        a Pandas dataframe.

        :param onlyRobust: if True, only the robust statistics are included
        :type onlyRobust: bool

        :return: Pandas dataframe with the results
        :rtype: pandas.DataFrame

        """
        # There should be a more 'Pythonic' way to do this.
        anyActiveBound = False
        for b in self.data.betas:
            if b.isBoundActive():
                anyActiveBound = True
        if anyActiveBound:
            if onlyRobust:
                columns = [
                    'Value',
                    'Active bound',
                    'Rob. Std err',
                    'Rob. t-test',
                    'Rob. p-value',
                ]
            else:
                columns = [
                    'Value',
                    'Active bound',
                    'Std err',
                    't-test',
                    'p-value',
                    'Rob. Std err',
                    'Rob. t-test',
                    'Rob. p-value',
                ]
        else:
            if onlyRobust:
                columns = [
                    'Value',
                    'Rob. Std err',
                    'Rob. t-test',
                    'Rob. p-value',
                ]
            else:
                columns = [
                    'Value',
                    'Std err',
                    't-test',
                    'p-value',
                    'Rob. Std err',
                    'Rob. t-test',
                    'Rob. p-value',
                ]
        if self.data.bootstrap is not None and not onlyRobust:
            columns += [
                f'Bootstrap[{len(self.data.bootstrap)}] Std err',
                'Bootstrap t-test',
                'Bootstrap p-value',
            ]
        table = pd.DataFrame(columns=columns)
        for b in self.data.betas:
            if anyActiveBound:
                if onlyRobust:
                    arow = {
                        'Value': b.value,
                        'Active bound': {True: 1.0, False: 0.0}[
                            b.isBoundActive()
                        ],
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue,
                    }
                else:
                    arow = {
                        'Value': b.value,
                        'Active bound': {True: 1.0, False: 0.0}[
                            b.isBoundActive()
                        ],
                        'Std err': b.stdErr,
                        't-test': b.tTest,
                        'p-value': b.pValue,
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue,
                    }
            else:
                if onlyRobust:
                    arow = {
                        'Value': b.value,
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue,
                    }
                else:
                    arow = {
                        'Value': b.value,
                        'Std err': b.stdErr,
                        't-test': b.tTest,
                        'p-value': b.pValue,
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue,
                    }
            if self.data.bootstrap is not None and not onlyRobust:
                arow[
                    f'Bootstrap[{len(self.data.bootstrap)}] Std err'
                ] = b.bootstrap_stdErr
                arow['Bootstrap t-test'] = b.bootstrap_tTest
                arow['Bootstrap p-value'] = b.bootstrap_pValue

            table.loc[b.name] = pd.Series(arow)
        return table

    def getCorrelationResults(self, subset=None):
        """Get the statistics about pairs of coefficients as a Pandas dataframe

        :param subset: produce the results only for a subset of
            parameters. If None, all the parameters are involved. Default: None
        :type subset: list(str)

        :return: Pandas data frame with the correlation results
        :rtype: pandas.DataFrame

        """
        if subset is not None:
            unknown = []
            for p in subset:
                if p not in self.data.betaNames:
                    unknown.append(p)
            if unknown:
                logger.warning(f'Unknown parameters are ignored: {unknown}')
        columns = [
            'Covariance',
            'Correlation',
            't-test',
            'p-value',
            'Rob. cov.',
            'Rob. corr.',
            'Rob. t-test',
            'Rob. p-value',
        ]
        if self.data.bootstrap is not None:
            columns += [
                'Boot. cov.',
                'Boot. corr.',
                'Boot. t-test',
                'Boot. p-value',
            ]
        table = pd.DataFrame(columns=columns)
        for k, v in self.data.secondOrderTable.items():
            if subset is None:
                include = True
            else:
                include = k[0] in subset and k[1] in subset
            if include:
                arow = {
                    'Covariance': v[0],
                    'Correlation': v[1],
                    't-test': v[2],
                    'p-value': v[3],
                    'Rob. cov.': v[4],
                    'Rob. corr.': v[5],
                    'Rob. t-test': v[6],
                    'Rob. p-value': v[7],
                }
                if self.data.bootstrap is not None:
                    arow['Boot. cov.'] = v[8]
                    arow['Boot. corr.'] = v[9]
                    arow['Boot. t-test'] = v[10]
                    arow['Boot. p-value'] = v[11]
                table.loc[f'{k[0]}-{k[1]}'] = pd.Series(arow)
        return table

    def getHtml(self):
        """Get the results coded in HTML

        :return: HTML code
        :rtype: string
        """
        now = datetime.datetime.now()
        h = self._getHtmlHeader()
        h += bv.getHtml()
        h += f'<p>This file has automatically been generated on {now}</p>\n'
        h += '<table>\n'
        h += (
            f'<tr class=biostyle><td align=right>'
            f'<strong>Report file</strong>:	</td>'
            f'<td>{self.data.htmlFileName}</td></tr>\n'
        )
        h += (
            f'<tr class=biostyle><td align=right>'
            f'<strong>Database name</strong>:	</td>'
            f'<td>{self.data.dataname}</td></tr>\n'
        )
        h += '</table>\n'

        if self.data.userNotes is not None:
            # User notes
            h += (
                f'<blockquote style="border: 2px solid #666; '
                f'padding: 10px; background-color:'
                f' #ccc;">{self.data.userNotes}</blockquote>'
            )

        # Include here the part on statistics

        h += '<h1>Estimation report</h1>\n'

        h += '<table border="0">\n'
        d = self.getGeneralStatistics()
        # k is the description of the quantity
        # v is the value
        # p is the precision to format it
        for k, (v, p) in d.items():
            if v is not None:
                h += (
                    f'<tr class=biostyle><td align=right >'
                    f'<strong>{k}</strong>: </td> '
                    f'<td>{v:{p}}</td></tr>\n'
                )
        for k, v in self.data.optimizationMessages.items():
            if k == 'Relative projected gradient':
                h += (
                    f'<tr class=biostyle><td align=right >'
                    f'<strong>{k}</strong>: </td> '
                    f'<td>{v:.7g}</td></tr>\n'
                )
            else:
                h += (
                    f'<tr class=biostyle><td align=right >'
                    f'<strong>{k}</strong>: </td> '
                    f'<td>{v}</td></tr>\n'
                )

        h += '</table>\n'

        table = self.getEstimatedParameters()

        h += '<h1>Estimated parameters</h1>\n'
        h += '<table border="1">\n'
        h += '<tr class=biostyle><th>Name</th>'
        for c in table.columns:
            h += f'<th>{c}</th>'
        h += '</tr>\n'
        for name, values in table.iterrows():
            h += f'<tr class=biostyle><td>{name}</td>'
            for k, v in values.items():
                #                print(f'values[{k}] = {v}')
                h += f'<td>{v:.3g}</td>'
            h += '</tr>\n'
        h += '</table>\n'

        table = self.getCorrelationResults()
        h += '<h2>Correlation of coefficients</h2>\n'
        h += '<table border="1">\n'
        h += '<tr class=biostyle><th>Coefficient1</th><th>Coefficient2</th>'
        for c in table.columns:
            h += f'<th>{c}</th>'
        h += '</tr>\n'
        for name, values in table.iterrows():
            n = name.split('-')
            h += f'<tr class=biostyle><td>{n[0]}</td><td>{n[1]}</td>'
            for k, v in values.items():
                h += f'<td>{v:.3g}</td>'
            h += '</tr>\n'
        h += '</table>\n'

        h += (
            f'<p>Smallest eigenvalue: '
            f'{self.data.smallestEigenValue:.6g}</p>\n'
        )
        h += (
            f'<p>Largest eigenvalue: '
            f'{self.data.largestEigenValue:.6g}</p>\n'
        )
        h += f'<p>Condition number: ' f'{self.data.conditionNumber:.6g}</p>\n'
        if np.abs(self.data.smallestEigenValue) <= 1.0e-5:
            h += (
                '<p>The second derivatives matrix is close to singularity. '
                'Variables involved:'
            )
            h += '<table>'
            for i, ev in enumerate(self.data.smallestEigenVector):
                if np.abs(ev) > 1.0e-5:
                    h += (
                        f'<tr><td>{ev:.3g}</td>'
                        f'<td> *</td>'
                        f'<td> {self.data.betaNames[i]}</td></tr>\n'
                    )
            h += '</table>'
            h += '</p>\n'

        h += '</html>'
        return h

    def getBetaValues(self, myBetas=None):
        """Retrieve the values of the estimated parameters, by names.

        :param myBetas: names of the requested parameters. If None, all
                  available parameters will be reported. Default: None.
        :type myBetas: list(string)

        :return: dict containing the values, where the keys are the names.
        :rtype: dict(string:float)


        :raise biogeme.exceptions.biogemeError: if some requested parameters
            are not available.
        """
        values = {}
        if myBetas is None:
            myBetas = self.data.betaNames
        for b in myBetas:
            try:
                index = self.data.betaNames.index(b)
                values[b] = self.data.betas[index].value
            except KeyError as e:
                keys = ''
                for k in self.data.betaNames:
                    keys += f' {k}'
                err = (
                    f'The value of {b} is not available in the results. '
                    f'The following parameters are available: {keys}'
                )
                raise excep.biogemeError(err) from e
        return values

    def getVarCovar(self):
        """Obtain the Rao-Cramer variance covariance matrix as a
        Pandas data frame.

        :return: Rao-Cramer variance covariance matrix
        :rtype: pandas.DataFrame
        """
        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i, betai in enumerate(self.data.betas):
            for j, betaj in enumerate(self.data.betas):
                vc.at[betai.name, betaj.name] = self.data.varCovar[i, j]
        return vc

    def getRobustVarCovar(self):
        """Obtain the robust variance covariance matrix as a Pandas data frame.

        :return: robust variance covariance matrix
        :rtype: pandas.DataFrame
        """
        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i, betai in enumerate(self.data.betas):
            for j, betaj in enumerate(self.data.betas):
                vc.at[betai.name, betaj.name] = self.data.robust_varCovar[i, j]
        return vc

    def getBootstrapVarCovar(self):
        """Obtain the bootstrap variance covariance matrix as
        a Pandas data frame.

        :return: bootstrap variance covariance matrix, or None if not available
        :rtype: pandas.DataFrame
        """
        if self.data.bootstrap is None:
            return None

        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i, betai in enumerate(self.data.betas):
            for j, betaj in enumerate(self.data.betas):
                vc.at[betai.name, betaj.name] = self.data.bootstrap_varCovar[
                    i, j
                ]
        return vc

    def writeHtml(self):
        """Write the results in an HTML file."""
        self.data.htmlFileName = bf.getNewFileName(self.data.modelName, 'html')
        with open(self.data.htmlFileName, 'w') as f:
            f.write(self.getHtml())
        logger.general(f'Results saved in file {self.data.htmlFileName}')

    def writeLaTeX(self):
        """Write the results in a LaTeX file."""
        self.data.latexFileName = bf.getNewFileName(self.data.modelName, 'tex')
        with open(self.data.latexFileName, 'w') as f:
            f.write(self.getLaTeX())
        logger.general(f'Results saved in file {self.data.latexFileName}')

    def _getHtmlHeader(self):
        """Prepare the header for the HTML file, containing comments and the
        version of Biogeme.

        :return: string containing the header.
        :rtype: str
        """
        h = ''
        h += '<html>\n'
        h += '<head>\n'
        h += (
            '<script src="http://transp-or.epfl.ch/biogeme/sorttable.js">'
            '</script>\n'
        )
        h += (
            '<meta http-equiv="Content-Type" content="text/html; '
            'charset=utf-8" />\n'
        )
        h += (
            f'<title>{self.data.modelName} - Report from '
            f'biogeme {bv.getVersion()} '
            f'[{bv.versionDate}]</title>\n'
        )
        h += (
            '<meta name="keywords" content="biogeme, discrete choice, '
            'random utility">\n'
        )
        h += (
            f'<meta name="description" content="Report from '
            f'biogeme {bv.getVersion()} '
            f'[{bv.versionDate}]">\n'
        )
        h += '<meta name="author" content="{bv.author}">\n'
        h += '<style type=text/css>\n'
        h += '.biostyle\n'
        h += '	{font-size:10.0pt;\n'
        h += '	font-weight:400;\n'
        h += '	font-style:normal;\n'
        h += '	font-family:Courier;}\n'
        h += '.boundstyle\n'
        h += '	{font-size:10.0pt;\n'
        h += '	font-weight:400;\n'
        h += '	font-style:normal;\n'
        h += '	font-family:Courier;\n'
        h += '        color:red}\n'
        h += '</style>\n'
        h += '</head>\n'
        h += '<body bgcolor="#ffffff">\n'
        return h

    def getBetasForSensitivityAnalysis(
        self, myBetas, size=100, useBootstrap=True
    ):
        """Generate draws from the distribution of the estimates, for
        sensitivity analysis.

        :param myBetas: names of the parameters for which draws are requested.
        :type myBetas: list(string)
        :param size: number of draws. If useBootstrap is True, the value is
            ignored and a warning is issued. Default: 100.
        :type size: int
        :param useBootstrap: if True, the bootstrap estimates are
                  directly used. The advantage is that it does not reyl on the
                  assumption that the estimates follow a normal
                  distribution. Default: True.
        :type useBootstrap: bool

        :raise biogeme.exceptions.biogemeError: if useBootstrap is True and
            the bootstrap results are not available

        :return: list of dict. Each dict has a many entries as parameters.
                The list has as many entries as draws.
        :rtype: list(dict)

        """
        if useBootstrap and self.data.bootstrap is None:
            err = (
                'Bootstrap results are not available for simulation. '
                'Use useBootstrap=False.'
            )
            raise excep.biogemeError(err)

        index = [self.data.betaNames.index(b) for b in myBetas]

        if useBootstrap:
            results = [
                {myBetas[i]: value for i, value in enumerate(row)}
                for row in self.data.bootstrap[:, index]
            ]

            return results

        theMatrix = (
            self.data.bootstrap_varCovar
            if useBootstrap
            else self.data.robust_varCovar
        )
        simulatedBetas = np.random.multivariate_normal(
            self.data.betaValues, theMatrix, size
        )

        index = [self.data.betaNames.index(b) for b in myBetas]

        results = [
            {myBetas[i]: value for i, value in enumerate(row)}
            for row in simulatedBetas[:, index]
        ]
        return results

    def getF12(self, robustStdErr=True):
        """F12 is a format used by the software ALOGIT to
        report estimation results.

        :param robustStdErr: if True, the robust standard errors are reports.
                             If False, the Rao-Cramer are.
        :type robustStdErr: bool

        :return: results in F12 format
        :rtype: string
        """

        # checkline1 = (
        #    '0000000001111111111222222222233333333334444444444'
        #    '5555555555666666666677777777778'
        # )
        # checkline2 = (
        #    '1234567890123456789012345678901234567890123456789'
        #    '0123456789012345678901234567890'
        # )

        results = ''

        # results += f'{checkline1}\n'
        # results += f'{checkline2}\n'

        # Line 1, title, characters 1-79
        results += f'{self.data.modelName[:79]: >79}\n'

        # Line 2, subtitle, characters 1-27, and time-date, characters 57-77
        t = f'From biogeme {bv.getVersion()}'
        d = f'{datetime.datetime.now()}'[:19]
        results += f'{t[:27]: <56}{d: <21}\n'

        # Line 3, "END" (this is historical!)
        results += 'END\n'

        # results += f'{checkline1}\n'
        # results += f'{checkline2}\n'

        # Line 4-(K+3), coefficient values
        #  characters 1-4, "   0" (again historical)
        #  characters 6-15, coefficient label, suggest using first 10
        #      characters of label in R
        #  characters 16-17, " F" (this indicates whether or not the
        #      coefficient is constrained)
        #  characters 19-38, coefficient value   20 chars
        #  characters 39-58, standard error      20 chars

        mystats = self.getGeneralStatistics()
        table = self.getEstimatedParameters(onlyRobust=False)
        coefNames = list(table.index.values)
        for name in coefNames:
            values = table.loc[name]
            results += '   0 '
            results += f'{name[:10]: >10}'
            if 'Active bound' in values:
                if values['Active bound'] == 1:
                    results += ' T'
                else:
                    results += ' F'
            else:
                results += ' F'
            results += ' '
            results += f' {values["Value"]: >+19.12e}'
            if robustStdErr:
                results += f' {values["Rob. Std err"]: >+19.12e}'
            else:
                results += f' {values["Std err"]: >+19.12e}'
            results += '\n'

        # Line K+4, "  -1" indicates end of coefficients
        results += '  -1\n'

        # results += f'{checkline1}\n'
        # results += f'{checkline2}\n'

        # Line K+5, statistics about run
        #   characters 1-8, number of observations        8 chars
        #   characters 9-27, likelihood-with-constants   19 chars
        #   characters 28-47, null likelihood            20 chars
        #   characters 48-67, final likelihood           20 chars

        results += f'{mystats["Sample size"][0]: >8}'
        # The cte log likelihood is not available. We put 0 instead.
        results += f' {0: >18}'
        if self.data.nullLogLike is not None:
            results += f' {mystats["Null log likelihood"][0]: >+19.12e}'
        else:
            results += f' {0: >19}'
        results += f' {mystats["Final log likelihood"][0]: >+19.12e}'
        results += '\n'

        # results += f'{checkline1}\n'
        # results += f'{checkline2}\n'

        # Line K+6, more statistics
        #   characters 1-4, number of iterations (suggest use 0)        4 chars
        #   characters 5-8, error code (please use 0)                   4 chars
        #   characters 9-29, time and date (sugg. repeat from line 2)  21 chars

        if "Number of iterations" in mystats:
            results += f'{mystats["Number of iterations"][0]: >4}'
        else:
            results += f'{0: >4}'
        results += f'{0: >4}'
        results += f'{d: >21}'
        results += '\n'

        # results += f'{checkline1}\n'
        # results += f'{checkline2}\n'

        # Lines (K+7)-however many we need, correlations*100000
        #   10 per line, fields of width 7
        #   The order of these is that correlation i,j (i>j) is in position
        #   (i-1)*(i-2)/2+j, i.e.
        #   (2,1) (3,1) (3,2) (4,1) etc.

        count = 0
        for i, coefi in enumerate(coefNames):
            for j in range(0, i):
                name = (coefi, coefNames[j])
                if robustStdErr:
                    corr = int(100000 * self.data.secondOrderTable[name][5])
                else:
                    corr = int(100000 * self.data.secondOrderTable[name][1])
                results += f'{corr:7d}'
                count += 1
                if count % 10 == 0:
                    results += '\n'
        results += '\n'
        return results

    def writeF12(self, robustStdErr=True):
        """Write the results in F12 file."""
        self.data.F12FileName = bf.getNewFileName(self.data.modelName, 'F12')
        with open(self.data.F12FileName, 'w') as f:
            f.write(self.getF12(robustStdErr))
        logger.general(f'Results saved in file {self.data.F12FileName}')

    def likelihood_ratio_test(self, other_model, significance_level=0.05):
        """This function performs a likelihood ratio test between a restricted
        and an unrestricted model. The "self" model can be either the
        restricted or the unrestricted.

        :param other_model: other model to perform the test.
        :type other_model: biogeme.results.bioResults

        :param significance_level: level of significance of the
            test. Default: 0.05
        :type significance_level: float

        :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

        :rtype: LRTuple(str, float, float)

        """
        LR = self.data.logLike
        LU = other_model.data.logLike
        KR = self.data.nparam
        KU = other_model.data.nparam
        return tools.likelihood_ratio_test(
            (LU, KU), (LR, KR), significance_level
        )


def compileEstimationResults(
    dict_of_results,
    statistics=(
        'Number of estimated parameters',
        'Sample size',
        'Final log likelihood',
        'Akaike Information Criterion',
        'Bayesian Information Criterion',
    ),
    include_parameter_estimates=True,
):

    """Compile estimation results into a common table

    :param dict_of_results: dictionary where the keys are the names of
        the models, and the values are either the estimation results,
        or the name of the pickle file where to find them.
    :type dict_of_results: dict(str:bioResults) or dict(str:str)

    :param statistics: list of statistics to include in the summary
        table
    :type statistics: tuple(str)

    :param include_parameter_estimates: if True, the parameter
        estimates are included.
    :type include_parameter_estimates: bool

    :return: pandas dataframe with the requested results.
    :rtype: pandas.DataFrame

    """
    df = pd.DataFrame(columns=dict_of_results.keys())

    for col, res in dict_of_results.items():
        if not isinstance(res, bioResults):
            try:
                res = bioResults(pickleFile=res)
            except excep.biogemeError:
                warning = f'Impossible to access result file {res}'
                logger.warning(warning)
                res = None
        if res is not None:
            stats_results = res.getGeneralStatistics()
            for s in statistics:
                df.loc[s, col] = stats_results[s][0]
            if include_parameter_estimates:
                betas = res.getBetaValues()
                for name, value in betas.items():
                    df.loc[name, col] = value

    return df.fillna('')
