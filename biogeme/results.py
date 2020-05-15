"""
Implementation of class contaning and processing the estimation results.

:author: Michel Bierlaire
:date: Tue Mar 26 16:50:01 2019

... todo:: rawResults should be a dict and not a class.
"""

# Too constraining
# pylint: disable=invalid-name,
# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches
# pylint: disable=too-many-statements, too-few-public-methods

#

import pickle
import datetime
import pandas as pd
import numpy as np
from scipy import linalg
from scipy import stats
import biogeme.version as bv
import biogeme.filenames as bf
import biogeme.exceptions as excep
import biogeme.messaging as msg



def calcPValue(t):
    """ Calculates the p value of a parameter from its t-statistic.

    The formula is

    .. math:: 2(1-\\Phi(|t|)

    where :math:`\\Phi(\\cdot)` is the CDF of a normal distribution.

    :param t: t-statistics
    :type: float

    :return: p-value
    :rtype: float
    """
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    return p

class beta:
    """ Class gathering the information related to the parameters of the model
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
        ## Name of the parameter
        self.name = name
        ## Current value
        self.value = value
        ## Lower bound
        self.lb = bounds[0]
        ## Upper bound
        self.ub = bounds[1]
        ## Standard error
        self.stdErr = None
        ## t-test
        self.tTest = None
        ## p-value
        self.pValue = None
        ## Robust standard error
        self.robust_stdErr = None
        ## Robust t-test
        self.robust_tTest = None
        ## Robust p-value
        self.robust_pValue = None
        ## Standard error calculated from bootstrap
        self.bootstrap_stdErr = None
        ## t-test  calculated from bootstrap
        self.bootstrap_tTest = None
        ## p-value calculated from bootstrap
        self.bootstrap_pValue = None


    def isBoundActive(self, threshold=1.0e-6):
        """Check if one of the two bound is 'numerically' active.

        Being numerically active means that the distance between the value of the parameter
        and one of its bounds is below the threshold.

        :param threshold: distance below which the bound is considered to be active.
                Default: :math:`10^{-6}`

        :return: True is one of the two bounds is numericall y active.
        :rtype: bool

        """
        if threshold < 0:
            raise excep.biogemeError(f'Threshold ({threshold}) must be non negative')

        if self.lb is not None and np.abs(self.value - self.lb) <= threshold:
            return True
        if self.ub is not None and np.abs(self.value - self.ub) <= threshold:
            return True
        return False

    def setStdErr(self, se):
        """ Records the standard error, and calculates and records
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
        """ Records the robust standard error, and calculates and records
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
        """ Records the robust standard error calculated by bootstrap, and calculates and
            records the corresponding t-statistic and p-value

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
            s += f'[{self.robust_stdErr:.3g} {self.robust_tTest:.3g} {self.robust_pValue:.3g}]'
        if self.bootstrap_stdErr is not None:
            s += (f'[{self.bootstrap_stdErr:.3g} {self.bootstrap_tTest:.3g} '
                  f'{self.bootstrap_pValue:.3g}]')
        return s

class rawResults:
    """ Class containing the raw results from the estimation
    """
    def __init__(self, theModel, betaValues, fgHb, bootstrap=None):
        """
        Constructor

        :param theModel: object with the model
        :type theModel: biogeme.BIOGEME
        :param betaValues: list containing the estimated values of the parameters
        :type betaValues: list(float)
        :param fgHb: tuple f,g,H,bhhh containing

                 - f: the value of the function,
                 - g: the gradient,
                 - H: the second derivative matrix,
                 - bhhh: the BHHH matrix.
        :type fgHb: float,numpy.array, numpy.array, numpy.array

        :param bootstrap: output of the bootstrapping. numpy array, of size B x K,  where

                - B is the number of bootstrap iterations
                - K is the number of parameters to estimate

                          Default: None.
        :type bootstrap: numpy.array
    """

        ## Name of the model
        self.modelName = theModel.modelName
        ## User notes
        self.userNotes = theModel.userNotes
        ## Number of parameters
        self.nparam = len(betaValues)
        ## Values of the parameters
        self.betaValues = betaValues
        ## Names of the parameters
        self.betaNames = theModel.freeBetaNames
        ## Value of the likelihood function with the initial value of the parameters
        self.initLogLike = theModel.initLogLike
        ## List of objects of type results.beta
        self.betas = list()
        for b, n in zip(betaValues, self.betaNames):
            bounds = theModel.getBoundsOnBeta(n)
            self.betas.append(beta(n, b, bounds))
        ## Value of the loglikelihood function
        self.logLike = fgHb[0]
        ## Value of the gradient of the loglikelihood function
        self.g = fgHb[1]
        ## Value of the hessian of the loglikelihood function
        self.H = fgHb[2]
        ## Value of the BHHH matrix of the loglikelihood function
        self.bhhh = fgHb[3]
        ## Name of the database
        self.dataname = theModel.database.name
        ## Sample size (number of individuals if panel data)
        self.sampleSize = theModel.database.getSampleSize()
        ## NUmber of observations
        self.numberOfObservations = theModel.database.getNumberOfObservations()
        ## True if the model involved Monte Carlo integration
        self.monteCarlo = theModel.monteCarlo
        ## Number of draws for Monte Carlo integration
        self.numberOfDraws = theModel.numberOfDraws
        ## Types of draws for Monte Carlo integration
        self.typesOfDraws = theModel.database.typesOfDraws
        ## Number of excluded data
        self.excludedData = theModel.database.excludedData
        ## Time needed to process the draws
        self.drawsProcessingTime = theModel.drawsProcessingTime
        ## Norm of the gradient
        self.gradientNorm = linalg.norm(self.g) if self.g is not None else None
        ## Diagnostics given by the optimization algorithm
        self.optimizationMessages = theModel.optimizationMessages
        ## Number of threads used for parallel computing
        self.numberOfThreads = theModel.numberOfThreads
        ## Name of the HTML output file
        self.htmlFileName = None
        ## Name of the LaTeX output file
        self.latexFileName = None
        ## Name of the pickle outpt file
        self.pickleFileName = None
        ##  output of the bootstrapping. numpy array, of size B x K,
        #    where
        #        - B is the number of bootstrap iterations
        #        - K is the number of parameters to estimate
        self.bootstrap = bootstrap
        if bootstrap is not None:
            ## Time needed to perform the bootstrap
            self.bootstrap_time = theModel.bootstrap_time

        ## Second order statistics
        self.secondOrderTable = None

class bioResults:
    """ Class managing the estimation results
    """
    def __init__(self, theRawResults=None, pickleFile=None):
        """ Constructor

        :param theRawResults: object with the results of the estimation. Default: None.
        :type theRawResults: biogeme.results.rawResults
        :param pickleFile: name of the file containing the raw results in pickle format.
                  Default: None.
        :type pickleFile: string

        :raise biogeme.exceptions.biogemeError: if no data is provided.
        """

        self.logger = msg.bioMessage()

        if theRawResults is not None:
            ### Object of type results.rawResults contaning the raw estimation results.
            self.data = theRawResults
        elif pickleFile is not None:
            with open(pickleFile, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise excep.biogemeError('No data provided.')

        self._calculateStats()

    def writePickle(self):
        """ Dump the data in a file in pickle format.

        :return: name of the file.
        :rtype: string
        """
        self.data.pickleFileName = bf.getNewFileName(self.data.modelName, 'pickle')
        with open(self.data.pickleFileName, 'wb') as f:
            pickle.dump(self.data, f)

        self.logger.general(f'Results saved in file {self.data.pickleFileName}')
        return self.data.pickleFileName



    def _calculateTest(self, i, j, matrix):
        """ Calculates a t-test comparing two coefficients

        Args:
           i: index of first coefficient \f$\\beta_i\f$.
           j: index of second coefficient \f$\\beta_i\f$.
           matrix: estimate of the variance-covariance matrix \f$m\f$.

        Returns:
           t test \f[ \\frac{\\beta_i-\\beta_j}{\\sqrt{m_{ii} + m_{jj} - 2 m_{ij} }}\f]
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

            - likelihood ratio test between the initial and ethe estimated models:
                   :math:`-2(L_0-L^*)`
            - Rho square: :math:`1 - \\frac{L^*}{L^0}`
            - Rho bar square: :math:`1 - \\frac{L^* - K}{L^0}`
            - AIC: :math:`2(K - L^*)`
            - BIC: :math:`-2 L^* + K  \\log(N)`

        Estimates for the variance-covariance matrix (Rao-Cramer,
        robust, and bootstrap) are also calculated, as well as t-tests and
        p value for the comparison of pairs of coefficients.

        """
        self.data.likelihoodRatioTest = -2.0 * (self.data.initLogLike - self.data.logLike) \
            if self.data.initLogLike is not None else None
        self.data.rhoSquare = np.nan_to_num(1.0 - self.data.logLike / self.data.initLogLike) \
            if self.data.initLogLike is not None else None
        self.data.rhoBarSquare = np.nan_to_num(1.0 - (self.data.logLike-self.data.nparam) \
                                               / self.data.initLogLike) \
            if self.data.initLogLike is not None else None
                                               
        self.data.akaike = 2.0 * self.data.nparam - 2.0 * self.data.logLike
        self.data.bayesian = - 2.0 * self.data.logLike + self.data.nparam * \
            np.log(self.data.sampleSize)
        # We calculate the eigenstructure to report in case of singularity
        if self.data.H is not None:
            self.data.eigenValues, self.data.eigenVectors = \
                linalg.eigh(-np.nan_to_num(self.data.H))
            _, self.data.singularValues, _ = linalg.svd(-np.nan_to_num(self.data.H))
            # We use the pseudo inverse in case the matrix is singular
            self.data.varCovar = -linalg.pinv(np.nan_to_num(self.data.H))
            for i in range(self.data.nparam):
                if self.data.varCovar[i, i] < 0:
                    self.data.betas[i].setStdErr(np.finfo(float).max)
                else:
                    self.data.betas[i].setStdErr(np.sqrt(self.data.varCovar[i, i]))

            d = np.diag(self.data.varCovar)
            if (d > 0).all():
                diag = np.diag(np.sqrt(d))
                diagInv = linalg.inv(diag)
                self.data.correlation = diagInv.dot(self.data.varCovar.dot(diagInv))
            else:
                self.data.correlation = np.full_like(self.data.varCovar, np.finfo(float).max)


            # Robust estimator
            self.data.robust_varCovar = \
                self.data.varCovar.dot(self.data.bhhh.dot(self.data.varCovar))
            for i in range(self.data.nparam):
                if self.data.robust_varCovar[i, i] < 0:
                    self.data.betas[i].setRobustStdErr(np.finfo(float).max)
                else:
                    self.data.betas[i].setRobustStdErr(np.sqrt(self.data.robust_varCovar[i, i]))
            rd = np.diag(self.data.robust_varCovar)
            if (rd > 0).all():
                diag = np.diag(np.sqrt(rd))
                diagInv = linalg.inv(diag)
                self.data.robust_correlation = diagInv.dot(self.data.robust_varCovar.dot(diagInv))
            else:
                self.data.robust_correlation = \
                    np.full_like(self.data.robust_varCovar, np.finfo(float).max)

            # Bootstrap
            if self.data.bootstrap is not None:
                self.data.bootstrap_varCovar = np.cov(self.data.bootstrap,rowvar=False)
                for i in range(self.data.nparam):
                    if self.data.bootstrap_varCovar[i,i] < 0:
                        self.data.betas[i].setBootstrapStdErr(np.finfo(float).max)
                    else:
                        self.data.betas[i].\
                            setBootstrapStdErr(np.sqrt(self.data.bootstrap_varCovar[i,i]))
                rd = np.diag(self.data.bootstrap_varCovar)
                if (rd > 0).all():
                    diag = np.diag(np.sqrt(rd))
                    diagInv = linalg.inv(diag)
                    self.data.bootstrap_correlation = \
                        diagInv.dot(self.data.bootstrap_varCovar.dot(diagInv))
                else:
                    self.data.bootstrap_correlation = \
                        np.full_like(self.data.bootstrap_varCovar,np.finfo(float).max)

            self.data.secondOrderTable = dict()
            for i in range(self.data.nparam):
                for j in range(i):
                    t = self._calculateTest(i, j, self.data.varCovar)
                    p = calcPValue(t)
                    trob = self._calculateTest(i, j, self.data.robust_varCovar)
                    prob = calcPValue(trob)
                    if self.data.bootstrap is not None:
                        tboot = self._calculateTest(i, j, self.data.bootstrap_varCovar)
                        pboot = calcPValue(tboot)
                    name = (self.data.betaNames[i], self.data.betaNames[j])
                    if self.data.bootstrap is not None:
                        self.data.secondOrderTable[name] = [self.data.varCovar[i, j],
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
                                                            pboot]
                    else:
                        self.data.secondOrderTable[name] = [self.data.varCovar[i, j],
                                                            self.data.correlation[i, j],
                                                            t,
                                                            p,
                                                            self.data.robust_varCovar[i, j],
                                                            self.data.robust_correlation[i, j],
                                                            trob,
                                                            prob]

            eigIndex = np.argmin(self.data.eigenValues)
            self.data.smallestEigenValue = self.data.eigenValues[eigIndex]
            self.data.smallestEigenVector = self.data.eigenVectors[:, eigIndex]
            self.data.smallestSingularValue = min(self.data.singularValues)
            eigIndex = np.argmax(self.data.eigenValues)
            self.data.largestEigenValue = self.data.eigenValues[eigIndex]
            self.data.largestEigenVector = self.data.eigenVectors[:, eigIndex]
            self.data.largestSingularValue = max(self.data.singularValues)
            self.data.conditionNumber = self.data.largestEigenValue / self.data.smallestEigenValue

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
        if self.data.initLogLike is not None:
            r += f'Init log likelihood:\t\t{self.data.initLogLike:.7g}\n'
        r += f'Final log likelihood:\t\t{self.data.logLike:.7g}\n'
        if self.data.initLogLike is not None:
            r += f'Likelihood ratio test:\t\t{self.data.likelihoodRatioTest:.7g}\n'
            r += f'Rho square:\t\t\t{self.data.rhoSquare:.3g}\n'
            r += f'Rho bar square:\t\t\t{self.data.rhoBarSquare:.3g}\n'
        r += f'Akaike Information Criterion:\t{self.data.akaike:.7g}\n'
        r += f'Bayesian Information Criterion:\t{self.data.bayesian:.7g}\n'
        if self.data.gradientNorm is not None:
            r += f'Final gradient norm:\t\t{self.data.gradientNorm:.7g}\n'
        r += '\n'.join([f'{b}' for b in self.data.betas])
        r += '\n'
        if self.data.secondOrderTable is not None:
            for k, v in self.data.secondOrderTable.items():
                r += ('{}:\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t'
                      '{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\n').format(k, *v)
        return r

    def _getLaTeXHeader(self):
        """Prepare the header for the LaTeX file, containing comments and the
        version of Biogeme.

        Return:
           string containing the header.
        """
        h = ''
        h += '%% This file is designed to be included into a LaTeX document\n'
        h += '%% See http://www.latex-project.org/ for information about LaTeX\n'
        h += (f'%% {self.data.modelName} - Report from biogeme {bv.getVersion()} '
              f'[{bv.versionDate}]\n')

        h += bv.getLaTeX()
        return h

    def getLaTeX(self):
        """ Get the results coded in LaTeX

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
            ## User notes
            h += (f'\n%%{self.data.userNotes}\n')

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
            if k == 'Relative projected gradient':
                h += f'{k} & \\verb${v:.7g}$ \\\\\n'
            else:
                h += f'{k} & \\verb${v}$ \\\\\n'
        h += '\\end{tabular}\n'

        h += '\n%%Parameter estimates\n'
        h += '\\section{Parameter estimates}\n'
        table = self.getEstimatedParameters()

        def formatting(x):
            """ Defines the formatting for the to_latex function of pandas
            """
            res = f'{x:.3g}'
            if '.' in res:
                return res

            return f'{res}.0'

        h += table.to_latex(float_format=formatting)

        h += '\n%%Correlation\n'
        h += '\\section{Correlation}\n'
        table = self.getCorrelationResults()
        h += table.to_latex(float_format=formatting)
        return h

    def getGeneralStatistics(self):
        """Format the results in a dict

        :return: dict with the results. The keys describe each
                 content. Each element is a tuple, with the value and its
                 preferred formatting.

        Example::

                     'Init log likelihood': (-115.30029248549191, '.7g')

        :rtype: dict(string:float,string)
        """
        d = {}
        d['Number of estimated parameters'] = self.data.nparam, ''
        d['Sample size'] = self.data.sampleSize, ''
        if self.data.sampleSize != self.data.numberOfObservations:
            d['Observations'] = self.data.numberOfObservations, ''
        d['Excluded observations'] = self.data.excludedData, ''
        d['Init log likelihood'] = self.data.initLogLike, '.7g'
        d['Final log likelihood'] = self.data.logLike, '.7g'
        d['Likelihood ratio test for the init. model'] = self.data.likelihoodRatioTest, '.7g'
        d['Rho-square for the init. model'] = self.data.rhoSquare, '.3g'
        d['Rho-square-bar for the init. model'] = self.data.rhoBarSquare, '.3g'
        d['Akaike Information Criterion'] = self.data.akaike, '.7g'
        d['Bayesian Information Criterion'] = self.data.bayesian, '.7g'
        d['Final gradient norm'] = self.data.gradientNorm, '.4E'
        if self.data.monteCarlo:
            d['Number of draws'] = self.data.numberOfDraws, ''
            d['Draws generation time'] = self.data.drawsProcessingTime, ''
            d['Types of draws'] = [f'{i}: {k}' for i, k in self.data.typesOfDraws.items()], ''
        if self.data.bootstrap is not None:
            d['Bootstrapping time'] = self.data.bootstrap_time, ''
        d['Nbr of threads'] = self.data.numberOfThreads, ''
        return d

    def getEstimatedParameters(self):
        """Gather the estimated parameters and the corresponding statistics in
a Pandas dataframe.

        :return: Pandas dataframe with the results
        :rtype: pandas.DataFrame

        """
        ### There should be a more 'Pythonic' way to do this.
        anyActiveBound = False
        for b in self.data.betas:
            if b.isBoundActive():
                anyActiveBound = True
        if anyActiveBound:
            columns = ['Value',
                       'Active bound',
                       'Std err',
                       't-test',
                       'p-value',
                       'Rob. Std err',
                       'Rob. t-test',
                       'Rob. p-value']
        else:
            columns = ['Value',
                       'Std err',
                       't-test',
                       'p-value',
                       'Rob. Std err',
                       'Rob. t-test',
                       'Rob. p-value']
        if self.data.bootstrap is not None:
            columns += [f'Bootstrap[{len(self.data.bootstrap)}] Std err',
                        'Bootstrap t-test',
                        'Bootstrap p-value']
        table = pd.DataFrame(columns=columns)
        for b in self.data.betas:
            if anyActiveBound:
                arow = {'Value': b.value,
                        'Active bound': {True: 1.0, False: 0.0}[b.isBoundActive()],
                        'Std err': b.stdErr,
                        't-test': b.tTest,
                        'p-value': b.pValue,
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue}
            else:
                arow = {'Value': b.value,
                        'Std err': b.stdErr,
                        't-test': b.tTest,
                        'p-value': b.pValue,
                        'Rob. Std err': b.robust_stdErr,
                        'Rob. t-test': b.robust_tTest,
                        'Rob. p-value': b.robust_pValue}
            if self.data.bootstrap is not None:
                arow[f'Bootstrap[{len(self.data.bootstrap)}] Std err'] = b.bootstrap_stdErr
                arow['Bootstrap t-test'] = b.bootstrap_tTest
                arow['Bootstrap p-value'] = b.bootstrap_pValue

            table.loc[b.name] = pd.Series(arow)
        return table

    def getCorrelationResults(self):
        """ Get the statistics about pairs of coefficients as a Pandas dataframe

        :return: Pandas data frame with the correlation results
        :rtpye: pandas.DataFrame
        """
        columns = ['Covariance',
                   'Correlation',
                   't-test',
                   'p-value',
                   'Rob. cov.',
                   'Rob. corr.',
                   'Rob. t-test',
                   'Rob. p-value']
        if self.data.bootstrap is not None:
            columns += ['Boot. cov.', 'Boot. corr.', 'Boot. t-test', 'Boot. p-value']
        table = pd.DataFrame(columns=columns)
        for k, v in self.data.secondOrderTable.items():
            arow = {'Covariance': v[0],
                    'Correlation' :v[1],
                    't-test': v[2],
                    'p-value': v[3],
                    'Rob. cov.': v[4],
                    'Rob. corr.': v[5],
                    'Rob. t-test': v[6],
                    'Rob. p-value': v[7]}
            if self.data.bootstrap is not None:
                arow['Boot. cov.'] = v[8]
                arow['Boot. corr.'] = v[9]
                arow['Boot. t-test'] = v[10]
                arow['Boot. p-value'] = v[11]
            table.loc[f'{k[0]}-{k[1]}'] = pd.Series(arow)
        return table

    def getHtml(self):
        """ Get the results coded in HTML

        :return: HTML code
        :rtpye: string
        """
        now = datetime.datetime.now()
        h = self._getHtmlHeader()
        h += bv.getHtml()
        h += f'<p>This file has automatically been generated on {now}</p>\n'
        h += ('<p>If you drag this HTML file into the Calc application of '
              '<a href="http://www.openoffice.org/" target="_blank">OpenOffice</a>, '
              'or the spreadsheet of <a href="https://www.libreoffice.org/" '
              'target="_blank">LibreOffice</a>, you will be able to perform additional '
              'calculations.</p>\n')
        h += '<table>\n'
        h += (f'<tr class=biostyle><td align=right><strong>Report file</strong>:	</td>'
              f'<td>{self.data.htmlFileName}</td></tr>\n')
        h += (f'<tr class=biostyle><td align=right><strong>Database name</strong>:	</td>'
              f'<td>{self.data.dataname}</td></tr>\n')
        h += '</table>\n'

        if self.data.userNotes is not None:
            ## User notes
            h += (f'<blockquote style="border: 2px solid #666; padding: 10px; background-color:'
                  f' #ccc;">{self.data.userNotes}</blockquote>')

        ### Include here the part on statistics

        h += '<h1>Estimation report</h1>\n'

        h += '<table border="0">\n'
        d = self.getGeneralStatistics()
        # k is the description of the quantity
        # v is the value
        # p is the precision to format it
        for k, (v, p) in d.items():
            h += (f'<tr class=biostyle><td align=right ><strong>{k}</strong>: </td> '
                  f'<td>{v:{p}}</td></tr>\n')
        for k, v in self.data.optimizationMessages.items():
            if k == 'Relative projected gradient':
                h += (f'<tr class=biostyle><td align=right ><strong>{k}</strong>: </td> '
                      f'<td>{v:.7g}</td></tr>\n')
            else:
                h += (f'<tr class=biostyle><td align=right ><strong>{k}</strong>: </td> '
                      f'<td>{v}</td></tr>\n')

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

        h += '<p>Smallest eigenvalue: {:.6g}</p>\n'.format(self.data.smallestEigenValue)
        h += '<p>Largest eigenvalue: {:.6g}</p>\n'.format(self.data.largestEigenValue)
        h += '<p>Condition number: {:.6g}</p>\n'.format(self.data.conditionNumber)
        if np.abs(self.data.smallestEigenValue) <= 1.0e-5:
            h += '<p>The second derivatives is close to singularity. Variables involved:'
            h += '<table>'
            for i in range(len(self.data.smallestEigenVector)):
                if np.abs(self.data.smallestEigenVector[i]) > 1.0e-5:
                    h += (f'<tr><td>{self.data.smallestEigenVector[i]:.3g}</td>'
                          f'<td> *</td>'
                          f'<td> {self.data.betaNames[i]}</td></tr>\n')
            h += '</table>'
            h += '</p>\n'

#        h += '<p>Smallest singular value: {:.6g}</p>\n'.format(self.data.smallestSingularValue)
        h += '</html>'
        return h

    def getBetaValues(self, myBetas=None):
        """Retrieve the values of the estimated parameters, by names.

        :param myBetas: names of the requested parameters. If None, all
                  available parameters will be reported. Default: None.
        :type myBetas: list(string)

        :return: dict containing the values, where the keys are the names.
        :rtype: dict(string:float)


        :raise biogeme.exceptions.biogemeError: if some requested parameters are not available.
        """
        values = dict()
        if myBetas is None:
            myBetas = self.data.betaNames
        for b in myBetas:
            try:
                index = self.data.betaNames.index(b)
                values[b] = self.data.betas[index].value
            except:
                keys = ''
                for k in self.data.betaNames:
                    keys += f' {k}'
                err = (f'The value of {b} is not available in the results. '
                       f'The following parameters are available: {keys}')
                raise excep.biogemeError(err)
        return values

    def getVarCovar(self):
        """ Obtain the Rao-Cramer variance covariance matrix as a Pandas data frame.

        :return: Rao-Cramer variance covariance matrix
        :rtype: pandas.DataFrame
        """
        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i in range(len(self.data.betas)):
            for j in range(len(self.data.betas)):
                vc.at[self.data.betas[i].name, self.data.betas[j].name] = self.data.varCovar[i, j]
        return vc


    def getRobustVarCovar(self):
        """ Obtain the robust variance covariance matrix as a Pandas data frame.

        :return: robust variance covariance matrix
        :rtype: pandas.DataFrame
        """
        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i in range(len(self.data.betas)):
            for j in range(len(self.data.betas)):
                vc.at[self.data.betas[i].name,
                      self.data.betas[j].name] = self.data.robust_varCovar[i, j]
        return vc


    def getBootstrapVarCovar(self):
        """ Obtain the bootstrap variance covariance matrix as a Pandas data frame.

        :return: bootstrap variance covariance matrix, or None if not available
        :rtype: pandas.DataFrame
        """
        if self.data.bootstrap is None:
            return None

        names = [b.name for b in self.data.betas]
        vc = pd.DataFrame(index=names, columns=names)
        for i in range(len(self.data.betas)):
            for j in range(len(self.data.betas)):
                vc.at[self.data.betas[i].name,
                      self.data.betas[j].name] = self.data.bootstrap_varCovar[i, j]
        return vc

    def writeHtml(self):
        """ Write the results in an HTML file.

        """
        self.data.htmlFileName = bf.getNewFileName(self.data.modelName, 'html')
        f = open(self.data.htmlFileName, 'w')
        f.write(self.getHtml())
        self.logger.general(f'Results saved in file {self.data.htmlFileName}')
        f.close()

    def writeLaTeX(self):
        """ Write the results in a LaTeX file.
        """
        self.data.latexFileName = bf.getNewFileName(self.data.modelName, 'tex')
        f = open(self.data.latexFileName, 'w')
        f.write(self.getLaTeX())
        f.close()

    def _getHtmlHeader(self):
        """Prepare the header for the HTML file, containing comments and the
        version of Biogeme.

        Return:
           string containing the header.
        """
        h = ''
        h += '<html>\n'
        h += '<head>\n'
        h += '<script src="http://transp-or.epfl.ch/biogeme/sorttable.js"></script>\n'
        h += '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n'
        h += (f'<title>{self.data.modelName} - Report from biogeme {bv.getVersion()} '
              f'[{bv.versionDate}]</title>\n')
        h += '<meta name="keywords" content="biogeme, discrete choice, random utility">\n'
        h += (f'<meta name="description" content="Report from biogeme {bv.getVersion()} '
              f'[{bv.versionDate}]">\n')
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

    def getBetasForSensitivityAnalysis(self, myBetas, size=100, useBootstrap=False):
        """Generate draws from the distribution of the estimates, for
        sensitivity analysis.

        :param myBetas: names of the parameters for which draws are requested.
        :type myBetas: list(string)
        :param size: number of draws. Default: 100.
        :type size: int
        :param useBootstrap: if True, the variance-covariance matrix
                  generated by the bootstrapping is used for simulation. If
                  False, the robust variance-covariance matrix is used. Default: False.
        :type useBootstrap: bool

        :raise biogeme.exceptions.biogemeError: if useBootstrap is True and the bootstrap
                          matrix is not available.

        :return: numpy table with as many rows as draws, and as many
           columns as parameters.
        :rtype: numpy.array

        """
        if useBootstrap and self.data.bootstrap is None:
            err = (f'Bootstrap variance-covariance matrix not available for simulation. '
                   f'Use useBootstrap=False.')
            raise excep.biogemeError(err)


        theMatrix = self.data.bootstrap_varCovar if useBootstrap else self.data.robust_varCovar
        simulatedBetas = np.random.multivariate_normal(self.data.betaValues,
                                                       theMatrix,
                                                       size)

        index = [self.data.betaNames.index(b) for b in myBetas]

        results = [{myBetas[i]: value for i, value in enumerate(row)}
                   for row in simulatedBetas[:, index]]
        return results
