"""Implementation of the main Biogeme class that combines the database
   and the model specification.

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

"""

# There seems to be a bug in PyLint.
# pylint: disable=invalid-unary-operand-type

# Too constraining
# pylint: disable=invalid-name,
# pylint: disable=too-many-arguments, too-many-locals,
# pylint: disable=too-many-statements, too-many-branches,
# pylint: disable=too-many-instance-attributes, too-many-lines,
# pylint: disable=too-many-function-args

import multiprocessing as mp
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.cbiogeme as cb
import biogeme.expressions as eb
import biogeme.tools as tools
import biogeme.results as res
import biogeme.exceptions as excep
import biogeme.filenames as bf
import biogeme.messaging as msg
import biogeme.optimization as opt

logger = msg.bioMessage()

#import yep

class BIOGEME:
    """Main class that combines the database and the model specification.

    It works in two modes: estimation and simulation.

    """

    def __init__(self,
                 database,
                 formulas,
                 userNotes=None,
                 numberOfThreads=None,
                 numberOfDraws=1000,
                 seed=None,
                 skipAudit=False,
                 removeUnusedVariables=True,
                 suggestScales=True,
                 missingData=99999):
        """Constructor

        :param database: choice data.
        :type database: biogeme.database

        :param formulas: expression or dictionary of expressions that
             define the model specification.  The concept is that each
             expression is applied to each entry of the database. The
             keys of the dictionary allow to provide a name to each
             formula.  In the estimation mode, two formulas are
             needed, with the keys 'loglike' and 'weight'. If only one
             formula is provided, it is associated with the label
             'loglike'. If no formula is labeled 'weight', the weight
             of each piece of data is supposed to be 1.0. In the
             simulation mode, the labels of each formula are used as
             labels of the resulting database.
        :type formulas: biogeme.expressions.Expression, or dict(biogeme.expressions.Expression)

        :param userNotes: these notes will be included in the report file.
        :type userNotes: str

        :param numberOfThreads: multi-threading can be used for
            estimation. This parameter defines the number of threads
            to be used. If the parameter is set to None, the number of
            available threads is calculated using
            cpu_count(). Ignored in simulation mode. Defaults: None.
        :type numberOfThreads:  int

        :param numberOfDraws: number of draws used for Monte-Carlo
            integration. Default: 1000.
        :type numberOfDraws: int

        :param seed: seed used for the pseudo-random number
            generation. It is useful only when each run should
            generate the exact same result. If None, a new seed is
            used at each run. Default: None.
        :type seed: int

        :param skipAudit: if True, does not check the validity of the
            formulas. It may save significant amount of time for large
            models and large data sets. Default: False.
        :type skipAudit: bool

        :param removeUnusedVariables: if True, all variables not used
           in the expression are removed from the database. Default:
           True.
        :type removeUnusedVariables: bool

        :param suggestScales: if True, Biogeme suggests the scaling of the variables in the database. Default: True. See also :func:`biogeme.database.Database.suggestScaling`
        :type suggestScales: bool. 

        :param missingData: if one variable has this value, it is
           assumed that a data is missing and an exception will be
           triggered. Default: 99999.
        :type missingData: float

        """

        ## Logger that controls the output of messages to the screen and log file.
        self.logger = logger
        if not skipAudit:
            database.data = database.data.replace({True: 1, False: 0})
            listOfErrors, listOfWarnings = database._audit()
            if listOfWarnings:
                self.logger.warning('\n'.join(listOfWarnings))
            if listOfErrors:
                self.logger.warning('\n'.join(listOfErrors))
                raise excep.biogemeError('\n'.join(listOfErrors))

        ## Keyword used for the name of the loglikelihood formula. Default: 'loglike'
        self.loglikeName = 'loglike'
        ## Keyword used for the name of the weight formula. Default: 'weight'
        self.weightName = 'weight'
        ## Name of the model. Default: 'biogemeModelDefaultName'
        self.modelName = 'biogemeModelDefaultName'
        ## monteCarlo is True if one of the expression involves a
        # Monte-Carlo integration.
        self.monteCarlo = False
        np.random.seed(seed)
        ## If True, the values are saved on a file each time the likelihood function is calculated
        self.saveIterations = False
        if not isinstance(formulas, dict):

            ## Object of type biogeme.expressions.Expression
            ## calculating the formula for the loglikelihood
            self.loglike = formulas

            ## Object of type biogeme.expressions.Expression
            ## calculating the weight of each observation in the
            ## sample
            self.weight = None

            ## Dictionary containing Biogeme formulas of type
            ## biogeme.expressions.Expression.
            # The keys are the names of the formulas.
            self.formulas = dict({self.loglikeName: formulas})
        else:
            self.loglike = formulas.get(self.loglikeName)
            self.weight = formulas.get(self.weightName)
            self.formulas = formulas


        ## biogeme.database object
        self.database = database

        ## User notes
        self.userNotes = userNotes
        
        ## Missing data
        self.missingData = missingData

        ## keep track of the sample of data used to calculate the
        ## stochastic gradient / hessian
        self.lastSample = None

        ## Init value of the likelihood function
        self.initLogLike = None

        self.usedVariables = set()
        for k, f in self.formulas.items():
            self.usedVariables = self.usedVariables.union(f.setOfVariables())
        if self.database.isPanel():
            self.usedVariables.add(self.database.panelColumn)
        if removeUnusedVariables:
            unusedVariables = set(self.database.data.columns) - self.usedVariables
            error_msg = (f'Remove {len(unusedVariables)} '
                         'unused variables from the database '
                         f'as only {len(self.usedVariables)} are used.')
            self.logger.general(error_msg)
            self.database.data = \
                self.database.data.drop(columns=list(unusedVariables))

        if suggestScales:
            suggestedScales = self.database.suggestScaling(columns=self.usedVariables)
            if not suggestedScales.empty:
                logger.detailed('It is suggested to scale the following variables.')
                for index, row in suggestedScales.iterrows():
                    error_msg = (f'Multiply {row["Column"]} by\t{row["Scale"]} '
                                 'because the largest (abs) value is\t'
                                 f'{row["Largest"]}')
                    logger.detailed(error_msg)
                error_msg = ('To remove this feature, set the parameter '
                             'suggestScales to False when creating the '
                             'BIOGEME object.')
                logger.detailed(error_msg)

        if not skipAudit:
            self._audit()

        self.theC = cb.pyBiogeme()

        self._prepareDatabaseForFormula()
        self._prepareLiterals()
        
        ## Boolean variable, True if the HTML file with the results must be generated.
        self.generateHtml = True
        
        ## Boolean variable, True if the pickle file with the results must be generated.
        self.generatePickle = True

        ## Name of the column defining weights for batch sampling in
        ## stochastic optimization.
        self.columnForBatchSamplingWeights = None

        ## Number of threads used for parallel computing. Default: the number of CPU available.
        self.numberOfThreads = mp.cpu_count() if numberOfThreads is None else numberOfThreads
        start_time = datetime.now()
        self._generateDraws(numberOfDraws)
        if self.monteCarlo:
            self.theC.setDraws(self.database.theDraws)
        ## Time needed to generate the draws.
        self.drawsProcessingTime = datetime.now() - start_time
        if self.loglike is not None:

            ## Internal signature of the formula for the loglikelihood
            self.loglikeSignatures = self.loglike.getSignature()
            if self.weight is None:
                self.theC.setExpressions(self.loglikeSignatures,
                                         self.numberOfThreads)
            else:
                ## Internal signature of the formula for the weight
                self.weightSignatures = self.weight.getSignature()
                self.theC.setExpressions(self.loglikeSignatures,
                                         self.numberOfThreads,
                                         self.weightSignatures)

        ## Time needed to calculate the bootstrap standard errors
        self.bootstrap_time = None

        ## Results of the bootstrap calculation.
        self.bootstrap_results = None

        ## Information provided by the optimization algorithm after completion.
        self.optimizationMessages = None

        ## Name of the File where intermediate iterations are stotred
        self.file_iterations = None

        ## Default bounds, replacing None, for the CFSQP algorithm
        self.cfsqp_default_bounds = 1000

        ## Parameters to be transferred to the optimization algorithm
        self.algoParameters = None

        ## Optimization algorithm
        self.algorithm = None

        ## Store the best iteration found so far.
        self.bestIteration = None

    def _audit(self):
        """Each expression provides an audit function, that verifies its
           validity. Each formula is audited, and the list of errors
           and warnings reported.

           :raise biogemeError: if the formula has issues, an error is
                                detected and an exception is raised.

        """

        listOfErrors = []
        listOfWarnings = []
        for k, v in self.formulas.items():
            err, war = v.audit(self.database)
            listOfErrors += err
            listOfWarnings += war
        if listOfWarnings:
            self.logger.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            self.logger.warning('\n'.join(listOfErrors))
            raise excep.biogemeError('\n'.join(listOfErrors))

    def _generateDraws(self, numberOfDraws):
        """If Monte-Carlo integration is involved in one of the formulas, this
           function instructs the database to generate the draws.

        Args:
            numberOfDraws: self explanatory (int)
        """


        ## Number of draws for Monte-Carlo integration.
        self.numberOfDraws = numberOfDraws
        ## Draws
        self.monteCarlo = len(self.allDraws) > 0
        if self.monteCarlo:
            self.database.generateDraws(self.allDraws,
                                        self.drawNames,
                                        numberOfDraws)


    def _prepareDatabaseForFormula(self, sample=None):
        # Prepare the dataset.
        if sample is None:
            if self.lastSample == 1.0:
                # We continue to use the full data set. Nothing to be done.
                return
            self.lastSample = 1.0
            self.database.useFullSample()
        else:
            # Check if the sample size is valid
            if sample <= 0 or sample > 1.0:
                error_msg = (f'The value of the parameter sample must be '
                             f'strictly between 0.0 and 1.0,'
                             f' and not {sample}')
                raise ValueError(error_msg)

            if sample == 1.0:
                self.database.useFullSample()
            else:
                self.logger.detailed(f'Use {100*sample}% of the data.')
                if self.database.isPanel():
                    self.database.\
                        sampleIndividualMapWithoutReplacement(sample,
                                                              self.columnForBatchSamplingWeights)
                else:
                    self.database.sampleWithoutReplacement(sample,
                                                           self.columnForBatchSamplingWeights)
            self.lastSample = sample

        # Rebuild the map for panel data
        if self.database.isPanel():
            self.database.buildPanelMap()
            self.theC.setPanel(True)
            self.theC.setDataMap(self.database.individualMap)

        # Transfer the data to the C++ formula
        self.theC.setData(self.database.data)
        self.theC.setMissingData(self.missingData)

    def getBoundsOnBeta(self, betaName):
        """ Returns the bounds on the parameter as defined by the user.

        :param betaName: name of the parameter
        :type betaName: string
        :return: lower bound, upper bound
        :rtype: tuple
        :raises biogemeError: if the name of the parameter is not found.
        """

        if betaName not in self.freeBetaNames:
            raise excep.biogemeError(f'Unknown parameter {betaName}')
        index = self.freeBetaNames.index(betaName)
        return self.bounds[index]

    def _prepareLiterals(self):
        """ Extract from the formulas the literals (parameters,
        variables, random variables) and decide a numbering convention.
        """

        collectionOfFormulas = [f for k, f in self.formulas.items()]
        variableNames = list(self.database.data.columns.values)


        self.elementaryExpressionIndex, \
        self.allFreeBetas, \
        self.freeBetaNames, \
        self.allFixedBetas, \
        self.fixedBetaNames, \
        self.allRandomVariables, \
        self.randomVariableNames, \
        self.allDraws, \
        self.drawNames = \
            eb.defineNumberingOfElementaryExpressions(collectionOfFormulas,
                                                      variableNames)

        ### List of tuples (ell, u) containing the lower and upper bounds
        # for each free parameter
        self.bounds = list()
        for x in self.freeBetaNames:
            self.bounds.append((self.allFreeBetas[x].lb,
                                self.allFreeBetas[x].ub))
        ## List of ids of the free beta parameters (those to be estimated)
        self.betaIds = list(range(len(self.freeBetaNames)))

        ## List of initial values of the free beta parameters (those to be estimated)
        self.betaInitValues = [float(self.allFreeBetas[x].initValue) for x in self.freeBetaNames]
        ## Values of the fixed parameters (not estimated).
        self.fixedBetaValues = [float(self.allFixedBetas[x].initValue) for x in self.fixedBetaNames]


    def calculateInitLikelihood(self):
        """Calculate the value of the log likelihood function

        The default values of the parameters are used.

        :return: value of the log likelihood.
        :rtype: float.
        """

        ## Value of the loglikelihood for the default values of the parameters.
        self.initLogLike = self.calculateLikelihood(self.betaInitValues,
                                                    scaled=False)
        return self.initLogLike

    def calculateLikelihood(self, x, scaled, batch=None):
        """Calculates the value of the log likelihood function

        :param x: vector of values for the parameters.
        :type x: list(float)

        :param scaled: if True, the value is diviced by the number of
                       observations used to calculate it. In this
                       case, the values with different sample sizes
                       are comparable. Default: True
        :type scaled: bool

        :param batch: if not None, calculates the likelihood on a
                       random sample of the data. The value of the
                       parameter must be strictly between 0 and 1, and
                       represents the share of the data that will be
                       used. Default: None
        :type batch: float

        :return: the calculated value of the log likelihood
        :rtype: float.

        :raises ValueError: if the length of the list x is incorrect.

        """

        if len(x) != len(self.betaInitValues):
            error_msg = (f'Input vector must be of length '
                         f'{len(self.betaInitValues)} and '
                         f'not {len(x)}')
            raise ValueError(error_msg)

        self._prepareDatabaseForFormula(batch)
        f = self.theC.calculateLikelihood(x, self.fixedBetaValues)

        self.logger.detailed(f'Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}')

        if scaled:
            return f / float(self.database.getSampleSize())

        return f

    def calculateLikelihoodAndDerivatives(self,
                                          x,
                                          scaled,
                                          hessian=False,
                                          bhhh=False,
                                          batch=None):
        """Calculate the value of the log likelihood function and its derivatives.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :param hessian: if True, the hessian is calculated. Default: False.
        :type hessian: bool

        :param bhhh: if True, the BHHH matrix is calculated. Default: False.
        :type bhhh: bool

        :param batch: if not None, calculates the likelihood on a
                       random sample of the data. The value of the
                       parameter must be strictly between 0 and 1, and
                       represents the share of the data that will be
                       used. Default: None
        :type batch: float


        :return: f, g, h, bh where

                - f is the value of the function (float)
                - g is the gradient (numpy.array)
                - h is the hessian (numpy.array)
                - bh is the BHHH matrix (numpy.array)

        :rtype: tuple  float, numpy.array, numpy.array, numpy.array

        :raises ValueError: if the length of the list x is incorrect

        """

        if len(x) != len(self.betaInitValues):
            error_msg = (f'Input vector must be of length '
                         f'{len(self.betaInitValues)} and not {len(x)}')
            raise ValueError(error_msg)
        self._prepareDatabaseForFormula(batch)

        f, g, h, bh = self.theC.calculateLikelihoodAndDerivatives(x,
                                                                  self.fixedBetaValues,
                                                                  self.betaIds,
                                                                  hessian,
                                                                  bhhh)

        if len(self.freeBetaNames) <= 30:
            for i in range(len(self.freeBetaNames)):
                self.logger.debug(f'{self.freeBetaNames[i]}: {x[i]:10.7g}')
        hmsg = ''
        if hessian:
            hmsg = f'Hessian norm:  {np.linalg.norm(h):10.1g}'
        bhhhmsg = ''
        if bhhh:
            bhhhmsg = f'BHHH norm:  {np.linalg.norm(bh):10.1g}'
        self.logger.general(f'Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}'
                            f' Gradient norm: {np.linalg.norm(g):10.1g}'
                            f' {hmsg} {bhhhmsg}')

        if self.saveIterations:
            if self.bestIteration is None:
                self.bestIteration = f
            if f >= self.bestIteration:
                with open(self.file_iterations, 'w') as pf:
                    for i, v in enumerate(x):
                        print(f'{self.freeBetaNames[i]} = {v}', file=pf)

        if scaled:
            N = float(self.database.getSampleSize())
            if N == 0:
                raise excep.biogemeError(f'Sample size is {N}')

            return f / N, np.asarray(g) / N, np.asarray(h) / N, np.asarray(bh) / N
        return f, np.asarray(g), np.asarray(h), np.asarray(bh)

    def likelihoodFiniteDifferenceHessian(self, x):
        """Calculate the hessian of the log likelihood function using finite differences.

        May be useful when the analytical hessian has numerical issues.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :return: finite differences approximation of the hessian.
        :rtype: numpy.array

        :raises ValueError: if the length of the list x is incorrect

        """

        def theFunction(x):
            f, g, _, _ = self.calculateLikelihoodAndDerivatives(x,
                                                                scaled=False,
                                                                hessian=False,
                                                                bhhh=False)
            return f, np.asarray(g)
        return tools.findiff_H(theFunction, np.asarray(x))

    def checkDerivatives(self, verbose=False):
        """Verifies the implementation of the derivatives.

        It compares the analytical version with the finite differences approximation.

        :param verbose: if True, the comparisons are reported. Default: False.
        :type verbose: bool

        :rtype: tuple.

        :return: f, g, h, gdiff, hdiff where

            - f is the value of the function,
            - g is the analytical gradient,
            - h is the analytical hessian,
            - gdiff is the difference between the analytical and the
              finite differences gradient,
            - hdiff is the difference between the analytical and the
              finite differences hessian,

        """
        def theFunction(x):
            """ Wrapper function to use tools.checkDerivatives """
            f, g, h, _ = self.calculateLikelihoodAndDerivatives(x,
                                                                scaled=False,
                                                                hessian=True,
                                                                bhhh=False)
            return f, np.asarray(g), np.asarray(h)

        return tools.checkDerivatives(theFunction,
                                      np.asarray(self.betaInitValues),
                                      self.freeBetaNames,
                                      verbose)


    def loadSavedIteration(self, filename='__savedIterations.txt'):
        """Reads the values of the parameters from a text file where each line
has the form name_of_beta = value_of_beta, and use these values in all
formulas.

        :param filename: name of the text file to read. Default: '__savedIterations.txt'
        :type filename: str.

        """
        betas = {}
        try:
            with open(filename) as fp:
                for line in fp:
                    l = line.split('=')
                    betas[l[0].strip()] = float(l[1])
            self.changeInitValues(betas)
            logger.detailed(f'Parameter values restored from {filename}')
        except IOError:
            logger.warning(f'Cannot read file {filename}. Statement is ignored.')

    def changeInitValues(self, betas):
        """Modifies the initial values of the pameters in all formula

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        if self.loglike is not None:
            self.loglike.changeInitValues(betas)
        if self.weight is not None:
            self.weight.changeInitValues(betas)
        for _, f in self.formulas.items():
            f.changeInitValues(betas)
        for i in range(len(self.freeBetaNames)):
            value = betas.get(self.freeBetaNames[i])
            if value is not None:
                self.betaInitValues[i] = value

    def estimate(self,
                 bootstrap=0,
                 algorithm=opt.simpleBoundsNewtonAlgorithmForBiogeme,
                 algoParameters=None,
                 cfsqp_default_bounds=1000.0,
                 saveIterations=False,
                 file_iterations='__savedIterations.txt'):

        """Estimate the parameters of the model.

        :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
        :type bootstrap: int

        :param algorithm: optimization algorithm to use for the
               maximum likelihood estimation. If None, cfsqp is
               . Default: Biogeme's Newton's algorithm with simple bounds.
        :type algorithm: function

        :param algoParameters: parameters to transfer to the optimization algorithm
        :type algoParameters: dict

        :param cfsqp_default_bounds: if the user does not provide bounds
              on the parameters, CFSQP assumes that the bounds are
              [-cfsqp_default_bounds, cfsqp_default_bounds]
        :type cfsqp_default_bounds: float

        :param saveIterations: if True, the values of the parameters
                               corresponding to the largest value of
                               the likelihood function are saved in a
                               pickle file at each iteration of the
                               algorithm. Default: False.
        :type saveIterations: bool

        :param file_iterations: name of the file where to save the
                               values of the parameters. Default:
                               '__savedIterations.txt'
        :type file_iterations: str

        :return: object containing the estimation results.
        :rtype: biogeme.bioResults

        Example::

            # Create an instance of biogeme
            biogeme  = bio.BIOGEME(database, logprob)

            # Gives a name to the model
            biogeme.modelName = 'mymodel'

            # Estimate the parameters
            results = biogeme.estimate()

        :raises biogemeError: if no expression has been provided for the likelihood

        """

        if self.loglike is None:
            raise excep.biogemeError('No log likelihood function has been specificed')
        if len(self.freeBetaNames) == 0:
            raise excep.biogemeError(f'There is no parameter to estimate'
                                     f' in the formula: {self.loglike}.')


        self.algorithm = algorithm
        self.algoParameters = algoParameters

        self.cfsqp_default_bounds = cfsqp_default_bounds

        self.calculateInitLikelihood()
        self.saveIterations = saveIterations
        self.file_iterations = f'{file_iterations}'
        self.bestIteration = None

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(self.betaInitValues)
        xstar, optimizationMessages = output
        ## Running time of the optimization algorithm
        optimizationMessages['Optimization time'] = datetime.now() - start_time
        ## Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages

        fgHb = self.calculateLikelihoodAndDerivatives(xstar,
                                                      scaled=False,
                                                      hessian=True,
                                                      bhhh=True)
        if not np.isfinite(fgHb[2]).all():
            warning_msg = ('Numerical problems in calculating '
                           'the analytical hessian. Finite differences'
                           ' is tried instead.')
            self.logger.warning(warning_msg)
            finDiffHessian = self.likelihoodFiniteDifferenceHessian(xstar)
            if not np.isfinite(fgHb[2]).all():
                self.logger.warning('Numerical problems with finite difference hessian as well.')
            else:
                fgHb = fgHb[0], fgHb[1], finDiffHessian, fgHb[3]
        ## numpy array, of size B x K,
        # where
        #        - B is the number of bootstrap iterations
        #        - K is the number pf parameters to estimate
        self.bootstrap_results = None
        if bootstrap > 0:
            start_time = datetime.now()

            self.logger.general(f'Re-estimate the model {bootstrap} times for bootstrapping')
            self.bootstrap_results = np.empty(shape=[bootstrap, len(xstar)])
            self.logger.temporarySilence()
            for b in range(bootstrap):
                if self.database.isPanel():
                    sample = self.database.sampleIndividualMapWithReplacement()
                    self.theC.setDataMap(sample)
                else:
                    sample = self.database.sampleWithReplacement()
                    self.theC.setData(sample)
                x_br, _ = self.optimize(xstar)
                self.bootstrap_results[b] = x_br

            ## Time needed to generate the bootstrap results
            self.bootstrap_time = datetime.now() - start_time
            self.logger.resume()
        rawResults = res.rawResults(self,
                                    xstar,
                                    fgHb,
                                    bootstrap=self.bootstrap_results)
        r = res.bioResults(rawResults)
        if self.generateHtml:
            r.writeHtml()
        if self.generatePickle:
            r.writePickle()
        return r


    def quickEstimate(self,
                      algorithm=opt.simpleBoundsNewtonAlgorithmForBiogeme,
                      algoParameters=None):

        """Estimate the parameters of the model. Same as estimate, where any extra 
           calculation is skipped (init loglikelihood, t-statistics, etc.)

        :param algorithm: optimization algorithm to use for the
               maximum likelihood estimation. If None, cfsqp is
               . Default: Biogeme's Newton's algorithm with simple bounds.
        :type algorithm: function

        :param algoParameters: parameters to transfer to the optimization algorithm
        :type algoParameters: dict

        :return: object containing the estimation results.
        :rtype: biogeme.results.bioResults

        Example::

            # Create an instance of biogeme
            biogeme  = bio.BIOGEME(database, logprob)

            # Gives a name to the model
            biogeme.modelName = 'mymodel'

            # Estimate the parameters
            results = biogeme.quickEstimate()

        :raises biogemeError: if no expression has been provided for the likelihood

        """

        if self.loglike is None:
            raise excep.biogemeError('No log likelihood function has been specificed')
        if len(self.freeBetaNames) == 0:
            raise excep.biogemeError(f'There is no parameter to estimate'
                                     f' in the formula: {self.loglike}.')


        self.algorithm = algorithm
        self.algoParameters = algoParameters

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(self.betaInitValues)
        xstar, optimizationMessages = output
        ## Running time of the optimization algorithm
        optimizationMessages['Optimization time'] = datetime.now() - start_time
        ## Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages

        f = self.calculateLikelihood(xstar,
                                     scaled=False)

        fgHb = f, None, None, None
        rawResults = res.rawResults(self,
                                    xstar,
                                    fgHb,
                                    bootstrap=self.bootstrap_results)
        r = res.bioResults(rawResults)
        return r

    def validate(self, estimationResults, slices=5):
        """Perform out-of-sample validation.

        The function performs the following tasks:

          - it shuffles the data set,
          - it splits the data set into slices of (approximatively) the same size,
          - each slice defines a validation set (the slice itself)
            and an estimation set (the rest of the data),
          - the model is re-estimated on the estimation set,
          - the estimated model is applied on the validation set,
          - the value of the log likelihood for each observation is reported.

        :param estimationResults: results of the model estimation based on the full data.
        :type estimationResults: biogeme.results.bioResults

        :param slices: number of slices.
        :type slices: int

        :return: a list containing as many items as slices. Each item
                 is the result of the simulation on the validation set.
        :rtype: list(pandas.DataFrame)

        """
        if self.database.isPanel():
            raise excep.biogemeError('Validation for panel data is not yet implemented')
        # Split the database
        validationData = self.database.split(slices)

        keepDatabase = self.database

        allSimulationResults = []
        for v in validationData:
            # v[0] is the estimation data set
            self.database = db.Database('Estimation data', v[0])
            self.loglike.changeInitValues(estimationResults.getBetaValues())
            results = self.estimate()
            simulate = {'Loglikelihood': self.loglike}
            simBiogeme = BIOGEME(db.Database('Validation data', v[1]),
                                 simulate)
            simResult = simBiogeme.simulate(results.getBetaValues())
            allSimulationResults.append(simResult)
        self.database = keepDatabase
        if self.generatePickle:
            fname = f'{self.modelName}_validation'
            pickleFileName = bf.getNewFileName(fname, 'pickle')
            with open(pickleFileName, 'wb') as f:
                pickle.dump(allSimulationResults, f)
            self.logger.general(f'Simulation results saved in file {pickleFileName}')

        return allSimulationResults

    def optimize(self, startingValues=None):
        """ Calls the optimization algorithm.

        The function self.algorithm is called. If None, CFSQP is invoked.

        :param startingValues: starting point for the algorithm
        :type: list(float)

        :return: x, messages

           - x is the solution generated by the algorithm,
           - messages is a dictionary describing several information about the lagorithm

        :rtype: numpay.array, dict(str:object)

        """

        theFunction = negLikelihood(like=self.calculateLikelihood,
                                    like_deriv=self.calculateLikelihoodAndDerivatives,
                                    scaled=True)

        if startingValues is None:
            startingValues = self.betaInitValues

        if self.algorithm is None:
            parameters = {'mode': 100,
                          'iprint': self.logger.screenLevel,
                          'miter': 1000,
                          'eps': 6.05545e-06}
            return self.cfsqp(startingValues,
                              self.bounds,
                              parameters)

        results = self.algorithm(theFunction,
                                 startingValues,
                                 self.bounds,
                                 self.algoParameters)
        return results

    def simulate(self, theBetaValues=None):
        """Applies the formulas to each row of the database.

        :param theBetaValues: values of the parameters to be used in
                the calculations. If None, the default values are
                used. Default: None.
        :type theBetaValues: dict(str, float)

        :return: a pandas data frame with the simulated value. Each
              row corresponds to a row in the database, and each
              column to a formula.

        :rtype: Pandas data frame

        Example::

              # Read the estimation results from a file
              results = res.bioResults(pickleFile = 'myModel.pickle')
              # Simulate the formulas using the nominal values
              simulatedValues = biogeme.simulate(betaValues)

        :raises biogemeError: if the number of parameters is incorrect

        """

        if self.database.isPanel():
            error_msg = ('Simulation for panel data is not yet'
                         ' implemented. Remove the "panel" '
                         'statement to simulate each observation.')
            raise excep.biogemeError(error_msg)

        if theBetaValues is None:
            betaValues = self.betaInitValues
        else:
            if not isinstance(theBetaValues, dict):
                err = (f'Deprecated. A dictionary must be provided. '
                       f'It can be obtained from results.getBetaValues()')
                raise excep.biogemeError(err)
            else:
                betaValues = list()
                for i in range(len(self.freeBetaNames)):
                    x = self.freeBetaNames[i]
                    if x in theBetaValues:
                        betaValues.append(theBetaValues[x])
                    else:
                        betaValues.append(self.betaInitValues[i])

        output = pd.DataFrame(index=self.database.data.index)
        for k, v in self.formulas.items():
            signature = v.getSignature()
            result = self.theC.simulateFormula(signature,
                                               betaValues,
                                               self.fixedBetaValues,
                                               self.database.data)
            output[k] = result
        return output


    def confidenceIntervals(self, betaValues, intervalSize=0.9):
        """Calculate confidence intervals on the simulated quantities


        :param betaValues: array of parameters values to be used in
               the calculations. Typically, it is a sample drawn from
               a distribution.
        :type betaValues: list(dict(str: float))

        :param intervalSize: size of the reported confidence interval,
                    in percentage. If it is denoted by s, the interval
                    is calculated for the quantiles (1-s)/2 and
                    (1+s)/2. The default (0.9) corresponds to
                    quantiles for the confidence interval [0.05, 0.95].
        :type intervalSize: float

        :return: two pandas data frames 'left' and 'right' with the
            same dimensions. Each row corresponds to a row in the
            database, and each column to a formula. 'left' contains the
            left value of the confidence interval, and 'right' the right
            value

            Example::

                # Read the estimation results from a file
                results = res.bioResults(pickleFile = 'myModel.pickle')
                # Retrieve the names of the betas parameters that have been estimated
                betas = biogeme.freeBetaNames

                # Draw 100 realization of the distribution of the estimators
                b = results.getBetasForSensitivityAnalysis(betas, size = 100)

                # Simulate the formulas using the nominal values
                simulatedValues = biogeme.simulate(betaValues)

                # Calculate the confidence intervals for each formula
                left, right = biogeme.confidenceIntervals(b, 0.9)

        :rtype: tuple of two Pandas dataframes.

        """
        listOfResults = []
        for b in betaValues:
            r = self.simulate(b)
            listOfResults += [r]
        allResults = pd.concat(listOfResults)
        r = (1.0 - intervalSize) / 2.0
        left = allResults.groupby(level=0).quantile(r)
        right = allResults.groupby(level=0).quantile(1.0 - r)
        return left, right

    def createLogFile(self, verbosity=3):
        """ Creates a log file with the messages produced by Biogeme.

        The name of the file is the name of the model with an extension .log

        :param verbosity: types of messages to be captured

            - 0: no output
            - 1: warnings
            - 2: only general information
            - 3: more verbose
            - 4: debug messages

            Default: 3.

        :type level: int

        """
        self.logger.createLog(fileLevel=verbosity,
                              fileName=self.modelName)

    def __str__(self):
        r = f'{self.modelName}: database [{self.database.name}]'
        r += str(self.formulas)
        print(r)
        return r

    def cfsqp(self, betas, bounds, parameters):
        """
        Invokes the CFSQP algorithm for estimation

        :param betas: initial values of the parameters to be estimated.
        :type betas: list of float
        :param bounds: lower and upper bounds on each parameter.
        :type bounds: list of tuple
        :param parameters: user defined parameters for CFSQP

           - mode = CBA: specifies job options as described below

                 * A = 0: ordinary minimax problems
                 * A = 1: ordinary minimax problems with each individual
                          function replaced by its absolute value, ie, an L_infty problem
                 * B = 0: monotone decrease of objective function after each iteration
                 * B = 1: monotone decrease of objective function after at most four iterations
                 * C = 1: default operation.
                 * C = 2: requires that constraints always be evaluated
                          before objectives during the line search.

           -  iprint: print level indicator with the following options

               * iprint = 0: no normal output, only error
                             information (this option is
                             imposed during phase 1)
               * iprint = 1: a final printout at a local solution
               * iprint = 2: a brief printout at the end of each iteration
               * iprint = 3: detailed infomation is printed out
                             at the end of each iteration
                             (for debugging purposes)

           -  miter: maximum number of iterations allowed by
                     the user to solve the problem


        :type parameters: dict


        :return: x, messages

        - x is the solution generated by the algorithm,
        - messages is a dictionary describing information about the lagorithm

        :rtype: numpay.array, dict(str:object)


        """
        lb, ub = zip(*bounds)
        lb = [-self.cfsqp_default_bounds if v is None else v for v in lb]
        ub = [self.cfsqp_default_bounds if v is None else v for v in ub]
        self.theC.setBounds(lb, ub)
        mode = parameters.get('mode', 100)
        iprint = parameters.get('iprint', 2)
        if iprint > 3:
            iprint = 3
        miter = parameters.get('miter', 1000)
        eps = parameters.get('eps', 6.05545e-06)

        x, iters, nf, diag = self.theC.cfsqp(betas,
                                             self.fixedBetaValues,
                                             self.betaIds,
                                             mode,
                                             iprint,
                                             miter,
                                             eps)
        messages = {'Algorithm': 'CFSQP',
                    'Number of iterations': iters,
                    'Number fo function evaluations': nf,
                    'Cause of termination': diag}
        return x, messages

class negLikelihood(opt.functionToMinimize):
    """Provides the value of the function to be minimized, as well as its
        derivatives. To be used by the opimization package.

    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, like, like_deriv, scaled):
        """ Constructor
        """
        self.recalculate = True
        self.x = None
        self.batch = None
        self.fv = None
        self.gv = None
        self.hv = None
        self.bhhhv = None
        self.like = like
        self.like_deriv = like_deriv
        self.scaled = scaled


    def setVariables(self, x):
        self.recalculate = True
        self.x = x
        self.fv = None
        self.gv = None
        self.hv = None
        self.bhhhv = None

    def f(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv = self.like(self.x, self.scaled, self.batch)
            self.gv = None
            self.hv = None
            self.bhhhv = None

        return -self.fv

    def f_g(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None or self.gv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv, self.gv, *_ = self.like_deriv(self.x,
                                                   self.scaled,
                                                   hessian=False,
                                                   bhhh=False,
                                                   batch=batch)
            self.hv = None
            self.bhhhv = None

        return -self.fv, -self.gv

    def f_g_h(self, batch=None):
        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.fv is None or self.gv is None or self.hv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv, self.gv, self.hv, _ = self.like_deriv(self.x,
                                                           self.scaled,
                                                           hessian=True,
                                                           bhhh=False,
                                                           batch=batch)
            self.bhhhv = None

        return -self.fv, -self.gv, -self.hv

    def f_g_bhhh(self, batch=None):
        if batch is not None or self.batch is not None:
            self.batch = batch
            self.recalculate = True

        if self.x is None:
            raise excep.biogemeError('The variables must be set first.')

        if self.fv is None or self.gv is None or self.bhhhv is None:
            self.recalculate = True

        if self.recalculate:
            self.fv, self.gv, _, self.bhhhv = self.like_deriv(self.x,
                                                              self.scaled,
                                                              hessian=False,
                                                              bhhh=True,
                                                              batch=batch)
            self.hv = None

        return (-self.fv,
                -self.gv,
                -self.bhhhv)
