"""Implementation of the main Biogeme class that combines the database
and the model specification.

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

"""

# Too constraining
# pylint: disable=invalid-name,
# pylint: disable=too-many-arguments, too-many-locals,
# pylint: disable=too-many-statements, too-many-branches,
# pylint: disable=too-many-instance-attributes, too-many-lines,
# pylint: disable=too-many-function-args, invalid-unary-operand-type


import glob
import multiprocessing as mp
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import tqdm

import biogeme.database as db
import biogeme.cythonbiogeme as cb
import biogeme.expressions as eb
import biogeme.results as res
import biogeme.exceptions as excep
import biogeme.filenames as bf
import biogeme.messaging as msg
import biogeme.optimization as opt
from biogeme import tools
from biogeme.algorithms import functionToMinimize
from biogeme.idmanager import IdManager

# import yep

logger = msg.bioMessage()
"""Logger that controls the output of
        messages to the screen and log file.
        Type: class :class:`biogeme.messaging.bioMessage`."""


class BIOGEME:
    """Main class that combines the database and the model specification.

    It works in two modes: estimation and simulation.

    """

    def __init__(
        self,
        database,
        formulas,
        userNotes=None,
        numberOfThreads=None,
        numberOfDraws=1000,
        seed=None,
        skipAudit=False,
        suggestScales=True,
        missingData=99999,
    ):
        """Constructor

        :param database: choice data.
        :type database: :class:`biogeme.database.Database`

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
        :type formulas: :class:`biogeme.expressions.Expression`, or
                        dict(:class:`biogeme.expressions.Expression`)

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

        :param suggestScales: if True, Biogeme suggests the scaling of
            the variables in the database. Default: True.
            See also :func:`biogeme.database.Database.suggestScaling`

        :type suggestScales: bool.

        :param missingData: if one variable has this value, it is
           assumed that a data is missing and an exception will be
           triggered. Default: 99999.
        :type missingData: float

        :raise biogemeError: an audit of the formulas is performed.
           If a formula has issues, an error is detected and an
           exception is raised.

        """

        if not skipAudit:
            database.data = database.data.replace({True: 1, False: 0})
            listOfErrors, listOfWarnings = database._audit()
            if listOfWarnings:
                logger.warning('\n'.join(listOfWarnings))
            if listOfErrors:
                logger.warning('\n'.join(listOfErrors))
                raise excep.biogemeError('\n'.join(listOfErrors))

        self.loglikeName = 'loglike'
        """ Keyword used for the name of the loglikelihood formula.
        Default: 'loglike'"""

        self.weightName = 'weight'
        """Keyword used for the name of the weight formula. Default: 'weight'
        """

        self.modelName = 'biogemeModelDefaultName'
        """Name of the model. Default: 'biogemeModelDefaultName'
        """
        self.monteCarlo = False
        """ ``monteCarlo`` is True if one of the expressions involves a
        Monte-Carlo integration.
        """

        if seed is not None:
            np.random.seed(seed)

        self.saveIterations = True
        """If True, the current iterate is saved after each iteration, in a
        file named ``__[modelName].iter``, where ``[modelName]`` is the
        name given to the model. If such a file exists, the starting
        values for the estimation are replaced by the values saved in
        the file.
        """

        self.missingData = missingData  #: code for missing data

        self.database = database  #: :class:`biogeme.database.Database` object

        self.numberOfThreads = (
            mp.cpu_count() if numberOfThreads is None else numberOfThreads
        )
        """ Number of threads used for parallel computing. Default: the number
        of available CPU.
        """

        self.numberOfDraws = numberOfDraws
        """ Number of draws for Monte-Carlo integration."""

        if not isinstance(formulas, dict):
            if not isinstance(formulas, eb.Expression):
                raise excep.biogemeError(
                    f'Expression {formulas} is not of type '
                    f'biogeme.expressions.Expression. '
                    f'It is of type {type(formulas)}'
                )

            self.loglike = formulas
            """ Object of type :class:`biogeme.expressions.Expression`
            calculating the formula for the loglikelihood
            """

            if self.database.isPanel():
                check_variables = self.loglike.check_panel_trajectory()

                if check_variables:
                    err_msg = (
                        f'Error in the loglikelihood function. '
                        f'Some variables are not inside PanelLikelihoodTrajectory: '
                        f'{check_variables} .'
                        f'If the database is organized as panel data, '
                        f'all variables must be used inside a '
                        f'PanelLikelihoodTrajectory. '
                        f'If it is not consistent with your model, generate a flat '
                        f'version of the data using the function '
                        f'`generateFlatPanelDataframe`.'
                    )
                    raise excep.biogemeError(err_msg)

            self.weight = None
            """ Object of type :class:`biogeme.expressions.Expression`
            calculating the weight of each observation in the
            sample.
            """

            self.formulas = dict({self.loglikeName: formulas})
            """ Dictionary containing Biogeme formulas of type
            :class:`biogeme.expressions.Expression`.
            The keys are the names of the formulas.
            """
        else:
            self.formulas = formulas
            # Verify the validity of the formulas
            for k, f in formulas.items():
                if not isinstance(f, eb.Expression):
                    raise excep.biogemeError(
                        f'Expression for "{k}" is not of type '
                        f'biogeme.expressions.Expression. '
                        f'It is of type {type(f)}'
                    )
            self.loglike = formulas.get(self.loglikeName)
            self.weight = formulas.get(self.weightName)

        for f in self.formulas.values():
            f.missingData = self.missingData

        self.userNotes = userNotes  #: User notes

        self.lastSample = None
        """ keeps track of the sample of data used to calculate the
        stochastic gradient / hessian
        """

        self.initLogLike = None  #: Init value of the likelihood function

        self.nullLogLike = None  #: Log likelihood of the null model

        if suggestScales:
            suggestedScales = self.database.suggestScaling()
            if not suggestedScales.empty:
                logger.detailed(
                    'It is suggested to scale the following variables.'
                )
                for _, row in suggestedScales.iterrows():
                    error_msg = (
                        f'Multiply {row["Column"]} by\t{row["Scale"]} '
                        'because the largest (abs) value is\t'
                        f'{row["Largest"]}'
                    )
                    logger.detailed(error_msg)
                error_msg = (
                    'To remove this feature, set the parameter '
                    'suggestScales to False when creating the '
                    'BIOGEME object.'
                )
                logger.detailed(error_msg)

        self._prepareDatabaseForFormula()

        if not skipAudit:
            self._audit()

        self.reset_id_manager()

        self.theC = cb.pyBiogeme(self.id_manager.number_of_free_betas)
        if self.database.isPanel():
            self.theC.setPanel(True)
            self.theC.setDataMap(self.database.individualMap)
        # Transfer the data to the C++ formula
        self.theC.setData(self.database.data)
        self.theC.setMissingData(self.missingData)

        self.generateHtml = True
        """ Boolean variable, True if the HTML file with the results must
        be generated.
        """

        self.generatePickle = True
        """ Boolean variable, True if the pickle file with the results must
        be generated.
        """

        self.columnForBatchSamplingWeights = None
        """ Name of the column defining weights for batch sampling in
        stochastic optimization.
        """

        start_time = datetime.now()
        self._generateDraws(numberOfDraws)
        if self.monteCarlo:
            self.theC.setDraws(self.database.theDraws)
        self.drawsProcessingTime = datetime.now() - start_time
        """ Time needed to generate the draws. """

        self.reset_id_manager()
        if self.loglike is not None:
            self.loglikeSignatures = self.loglike.getSignature()
            """ Internal signature of the formula for the loglikelihood."""
            if self.weight is None:
                self.theC.setExpressions(
                    self.loglikeSignatures, self.numberOfThreads
                )
            else:
                self.weightSignatures = self.weight.getSignature()
                """ Internal signature of the formula for the weight."""
                self.theC.setExpressions(
                    self.loglikeSignatures,
                    self.numberOfThreads,
                    self.weightSignatures,
                )

        self.bootstrap_time = None
        """ Time needed to calculate the bootstrap standard errors"""

        self.bootstrap_results = None  #: Results of the bootstrap calculation.

        self.optimizationMessages = None
        """ Information provided by the optimization algorithm
        after completion.
        """

        self.algoParameters = None
        """ Parameters to be transferred to the optimization algorithm
        """

        self.algorithm = None  #: Optimization algorithm

        self.bestIteration = None  #: Store the best iteration found so far.

    def reset_id_manager(self):
        """Reset all the ids of the elementary expression in the formulas"""
        # First, we reset the IDs
        for f in self.formulas.values():
            f.setIdManager(id_manager=None)
        # Second, we calculate a new set of IDs.
        self.id_manager = IdManager(
            self.formulas.values(),
            self.database,
            self.numberOfDraws,
            force_new_ids=True,
        )
        for f in self.formulas.values():
            f.setIdManager(id_manager=self.id_manager)

    def _saveIterationsFileName(self):
        """
        :return: The name of the file where the iterations are saved.
        :rtype: str
        """
        return f'__{self.modelName}.iter'

    def _audit(self):
        """Each expression provides an audit function, that verifies its
        validity. Each formula is audited, and the list of errors
        and warnings reported.

        :raise biogemeError: if the formula has issues, an error is
                             detected and an exception is raised.

        """

        listOfErrors = []
        listOfWarnings = []
        for v in self.formulas.values():
            check_draws = v.check_draws()
            if check_draws:
                err_msg = (
                    f'The following draws are defined outside the '
                    f'MonteCarlo operator: {check_draws}'
                )
                listOfErrors.append(err_msg)
            check_rv = v.check_rv()
            if check_rv:
                err_msg = (
                    f'The following random variables are defined '
                    f'outside the Integrate operator: {check_draws}'
                )
                listOfErrors.append(err_msg)
            err, war = v.audit(self.database)
            listOfErrors += err
            listOfWarnings += war
        if self.weight is not None:
            total = self.weight.getValue_c(
                database=self.database, aggregation=True, prepareIds=True
            )
            s_size = self.database.getSampleSize()
            ratio = s_size / total
            if np.abs(ratio - 1) >= 0.01:
                theWarning = (
                    f'The sum of the weights ({total}) is different from '
                    f'the sample size ({self.database.getSampleSize()}). '
                    f'Multiply the weights by {ratio} to reconcile the two.'
                )
                listOfWarnings.append(theWarning)
        if listOfWarnings:
            logger.warning('\n'.join(listOfWarnings))
        if listOfErrors:
            logger.warning('\n'.join(listOfErrors))
            raise excep.biogemeError('\n'.join(listOfErrors))

    def _generateDraws(self, numberOfDraws):
        """If Monte-Carlo integration is involved in one of the formulas, this
           function instructs the database to generate the draws.

        Args:
            numberOfDraws: self explanatory (int)
        """

        # Draws
        self.monteCarlo = self.id_manager.requires_draws
        if self.monteCarlo:
            self.database.generateDraws(
                self.id_manager.draws.expressions,
                self.id_manager.draws.names,
                numberOfDraws,
            )

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
                error_msg = (
                    f'The value of the parameter sample must be '
                    f'strictly between 0.0 and 1.0,'
                    f' and not {sample}'
                )
                raise ValueError(error_msg)

            if sample == 1.0:
                self.database.useFullSample()
            else:
                logger.detailed(f'Use {100*sample}% of the data.')
                if self.database.isPanel():
                    self.database.sampleIndividualMapWithoutReplacement(
                        sample, self.columnForBatchSamplingWeights
                    )
                else:
                    self.database.sampleWithoutReplacement(
                        sample, self.columnForBatchSamplingWeights
                    )
            self.lastSample = sample

        # Rebuild the map for panel data
        if self.database.isPanel():
            self.database.buildPanelMap()

    def freeBetaNames(self):
        """Returns the names of the parameters that must be estimated

        :return: list of names of the parameters
        :rtype: list(str)
        """
        return self.id_manager.free_betas.names

    def getBoundsOnBeta(self, betaName):
        """Returns the bounds on the parameter as defined by the user.

        :param betaName: name of the parameter
        :type betaName: string
        :return: lower bound, upper bound
        :rtype: tuple
        :raises biogemeError: if the name of the parameter is not found.
        """
        index = self.id_manager.free_betas.indices.get(betaName)
        if index is None:
            raise excep.biogemeError(f'Unknown parameter {betaName}')
        return self.id_manager.bounds[index]

    def calculateNullLoglikelihood(self, avail):
        """Calculate the log likelihood of the null model that predicts equal
        probability for each alternative

        :param avail: list of expressions to evaluate the availability
                      conditions for each alternative. If None, all
                      alternatives are always available.
        :type avail: list of :class:`biogeme.expressions.Expression`

        :return: value of the log likelihood
        :rtype: float

        """
        expression = -eb.log(eb.bioMultSum(avail))

        self.nullLogLike = expression.getValue_c(
            database=self.database,
            aggregation=True,
            prepareIds=True,
        )
        return self.nullLogLike

    def calculateInitLikelihood(self):
        """Calculate the value of the log likelihood function

        The default values of the parameters are used.

        :return: value of the log likelihood.
        :rtype: float.
        """
        # Value of the loglikelihood for the default values of the parameters.
        self.initLogLike = self.calculateLikelihood(
            self.id_manager.free_betas_values, scaled=False
        )
        return self.initLogLike

    def calculateLikelihood(self, x, scaled, batch=None):
        """Calculates the value of the log likelihood function

        :param x: vector of values for the parameters.
        :type x: list(float)

        :param scaled: if True, the value is divided by the number of
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
        if len(x) != len(self.id_manager.free_betas_values):
            error_msg = (
                f'Input vector must be of length '
                f'{len(self.id_manager.free_betas_values)} and '
                f'not {len(x)}'
            )
            raise ValueError(error_msg)

        self._prepareDatabaseForFormula(batch)
        f = self.theC.calculateLikelihood(
            x, self.id_manager.fixed_betas_values
        )

        logger.detailed(
            f'Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}'
        )

        if scaled:
            return f / float(self.database.getSampleSize())

        return f

    def calculateLikelihoodAndDerivatives(
        self, x, scaled, hessian=False, bhhh=False, batch=None
    ):
        """Calculate the value of the log likelihood function
        and its derivatives.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :param scaled: if True, the results are devided by the number of
            observations.
        :type scaled: bool

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

        :raises biogemeError: if the norm of the gradient is not finite, an
            error is raised.
        """
        n = len(x)
        if n != self.id_manager.number_of_free_betas:
            error_msg = (
                f'Input vector must be of length '
                f'{self.id_manager.number_of_free_betas} and not {len(x)}'
            )
            raise ValueError(error_msg)
        self._prepareDatabaseForFormula(batch)

        g = np.empty(n)
        h = np.empty([n, n])
        bh = np.empty([n, n])
        f, g, h, bh = self.theC.calculateLikelihoodAndDerivatives(
            x,
            self.id_manager.fixed_betas_values,
            self.id_manager.free_betas.indices.values(),
            g,
            h,
            bh,
            hessian,
            bhhh,
        )

        hmsg = ''
        if hessian:
            hmsg = f'Hessian norm:  {np.linalg.norm(h):10.1g}'
        bhhhmsg = ''
        if bhhh:
            bhhhmsg = f'BHHH norm:  {np.linalg.norm(bh):10.1g}'
        gradnorm = np.linalg.norm(g)
        logger.general(
            f'Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}'
            f' Gradient norm: {gradnorm:10.1g}'
            f' {hmsg} {bhhhmsg}'
        )

        if not np.isfinite(gradnorm):
            error_msg = f'The norm of the gradient is {gradnorm}: g={g}'
            raise excep.biogemeError(error_msg)

        if self.saveIterations:
            if self.bestIteration is None:
                self.bestIteration = f
            if f >= self.bestIteration:
                with open(self._saveIterationsFileName(), 'w') as pf:
                    for i, v in enumerate(x):
                        print(
                            f'{self.id_manager.free_betas.names[i]} = {v}',
                            file=pf,
                        )

        if scaled:
            N = float(self.database.getSampleSize())
            if N == 0:
                raise excep.biogemeError(f'Sample size is {N}')

            return (
                f / N,
                np.asarray(g) / N,
                np.asarray(h) / N,
                np.asarray(bh) / N,
            )
        return f, np.asarray(g), np.asarray(h), np.asarray(bh)

    def likelihoodFiniteDifferenceHessian(self, x):
        """Calculate the hessian of the log likelihood function using finite
        differences.

        May be useful when the analytical hessian has numerical issues.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :return: finite differences approximation of the hessian.
        :rtype: numpy.array

        :raises ValueError: if the length of the list x is incorrect

        """

        def theFunction(x):
            f, g, _, _ = self.calculateLikelihoodAndDerivatives(
                x, scaled=False, hessian=False, bhhh=False
            )
            return f, np.asarray(g)

        return tools.findiff_H(theFunction, np.asarray(x))

    def checkDerivatives(self, verbose=False):
        """Verifies the implementation of the derivatives.

        It compares the analytical version with the finite differences
        approximation.

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
            """Wrapper function to use tools.checkDerivatives"""
            f, g, h, _ = self.calculateLikelihoodAndDerivatives(
                x, scaled=False, hessian=True, bhhh=False
            )
            return f, np.asarray(g), np.asarray(h)

        return tools.checkDerivatives(
            theFunction,
            np.asarray(self.id_manager.free_betas_values),
            self.id_manager.free_betas.names,
            verbose,
        )

    def _loadSavedIteration(self):
        """Reads the values of the parameters from a text file where each line
        has the form name_of_beta = value_of_beta, and use these values in all
        formulas.

        """
        filename = self._saveIterationsFileName()
        betas = {}
        try:
            with open(filename) as fp:
                for line in fp:
                    ell = line.split('=')
                    betas[ell[0].strip()] = float(ell[1])
            self.changeInitValues(betas)
            logger.detailed(f'Parameter values restored from {filename}')
        except IOError:
            logger.warning(
                f'Cannot read file {filename}. Statement is ignored.'
            )

    def setRandomInitValues(self, defaultBound=100.0):
        """Modifies the initial values of the parameters in all formulas,
        using randomly generated values. The value is drawn from a
        uniform distribution on the interval defined by the
        bounds.

        :param defaultBound: If the upper bound is missing, it is
            replaced by this value. If the lower bound is missing, it is
            replaced by the opposite of this value. Default: 100.
        :type defaultBound: float
        """
        randomBetas = {
            name: np.random.uniform(
                low=-defaultBound if beta.lb is None else beta.lb,
                high=defaultBound if beta.ub is None else beta.ub,
            )
            for name, beta in self.id_manager.free_betas.expressions.items()
        }
        self.changeInitValues(randomBetas)

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
        for i, name in enumerate(self.id_manager.free_betas.names):
            value = betas.get(name)
            if value is not None:
                self.id_manager.free_betas_values[i] = value

    def estimate(
        self,
        recycle=False,
        bootstrap=0,
        algorithm=opt.simpleBoundsNewtonAlgorithmForBiogeme,
        algoParameters=None,
    ):

        """Estimate the parameters of the model.

        :param recycle: if True, the results are read from the pickle
            file, if it exists. If False, the estimation is performed.
        :type recycle: bool

        :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
        :type bootstrap: int

        :param algorithm: optimization algorithm to use for the
               maximum likelihood estimation. Default: Biogeme's
               Newton's algorithm with simple bounds.
        :type algorithm: function

        :param algoParameters: parameters to transfer to the optimization
            algorithm
        :type algoParameters: dict

        :return: object containing the estimation results.
        :rtype: biogeme.bioResults

        Example::

            # Create an instance of biogeme
            biogeme  = bio.BIOGEME(database, logprob)

            # Gives a name to the model
            biogeme.modelName = 'mymodel'

            # Estimate the parameters
            results = biogeme.estimate()

        :raises biogemeError: if no expression has been provided for the
            likelihood

        """
        if self.modelName == 'biogemeModelDefaultName':
            logger.warning(
                'You have not defined a name for the model. '
                'The output files are named from the model name. '
                'The default is [biogemeModelDefaultName]'
            )

        if recycle:
            pickle_files = self.files_of_type('pickle')
            pickle_files.sort()
            if pickle_files:
                pickle_to_read = pickle_files[-1]
                if len(pickle_files) > 1:
                    warning_msg = (
                        f'Several pickle files are available for '
                        f'this model: {pickle_files}. '
                        f'The file {pickle_to_read} '
                        f'is used to load the results.'
                    )
                    logger.warning(warning_msg)
                results = res.bioResults(pickleFile=pickle_to_read)
                logger.warning(
                    f'Estimation results read from {pickle_to_read}. '
                    f'There is no guarantee that they correspond '
                    f'to the specified model.'
                )
                return results
            warning_msg = (
                'Recycling was requested, but no pickle file was found'
            )
            logger.warning(warning_msg)
        if self.loglike is None:
            raise excep.biogemeError(
                'No log likelihood function has been specified'
            )
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.biogemeError(
                f'There is no parameter to estimate'
                f' in the formula: {self.loglike}.'
            )

        if self.saveIterations:
            logger.general(
                f'*** Initial values of the parameters are '
                f'obtained from the file {self._saveIterationsFileName()}'
            )
            self._loadSavedIteration()
        self.algorithm = algorithm
        self.algoParameters = algoParameters

        self.calculateInitLikelihood()
        self.bestIteration = None

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(self.id_manager.free_betas_values)
        xstar, optimizationMessages = output
        # Running time of the optimization algorithm
        optimizationMessages['Optimization time'] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages

        fgHb = self.calculateLikelihoodAndDerivatives(
            xstar, scaled=False, hessian=True, bhhh=True
        )
        if not np.isfinite(fgHb[2]).all():
            warning_msg = (
                'Numerical problems in calculating '
                'the analytical hessian. Finite differences'
                ' is tried instead.'
            )
            logger.warning(warning_msg)
            finDiffHessian = self.likelihoodFiniteDifferenceHessian(xstar)
            if not np.isfinite(fgHb[2]).all():
                logger.warning(
                    'Numerical problems with finite '
                    'difference hessian as well.'
                )
            else:
                fgHb = fgHb[0], fgHb[1], finDiffHessian, fgHb[3]

        # numpy array, of size B x K,
        # where
        #        - B is the number of bootstrap iterations
        #        - K is the number pf parameters to estimate
        self.bootstrap_results = None
        if bootstrap > 0:

            start_time = datetime.now()

            logger.general(
                f'Re-estimate the model {bootstrap} times for bootstrapping'
            )
            self.bootstrap_results = np.empty(shape=[bootstrap, len(xstar)])
            hideProgress = logger.screenLevel == 0
            logger.temporarySilence()
            for b in tqdm.tqdm(range(bootstrap), disable=hideProgress):
                if self.database.isPanel():
                    sample = self.database.sampleIndividualMapWithReplacement()
                    self.theC.setDataMap(sample)
                else:
                    sample = self.database.sampleWithReplacement()
                    self.theC.setData(sample)
                x_br, _ = self.optimize(xstar)
                self.bootstrap_results[b] = x_br

            # Time needed to generate the bootstrap results
            self.bootstrap_time = datetime.now() - start_time
            logger.resume()
        rawResults = res.rawResults(
            self, xstar, fgHb, bootstrap=self.bootstrap_results
        )
        r = res.bioResults(rawResults)
        if self.generateHtml:
            r.writeHtml()
        if self.generatePickle:
            r.writePickle()
        return r

    def quickEstimate(
        self,
        algorithm=opt.simpleBoundsNewtonAlgorithmForBiogeme,
        algoParameters=None,
    ):

        """| Estimate the parameters of the model. Same as estimate, where any
             extra calculation is skipped (init loglikelihood,
             t-statistics, etc.)

        :param algorithm: optimization algorithm to use for the
               maximum likelihood estimation.Default: Biogeme's
               Newton's algorithm with simple bounds.
        :type algorithm: function

        :param algoParameters: parameters to transfer to the optimization
            algorithm
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

        :raises biogemeError: if no expression has been provided for the
            likelihood

        """
        if self.loglike is None:
            raise excep.biogemeError(
                'No log likelihood function has been specificed'
            )
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.biogemeError(
                f'There is no parameter to estimate'
                f' in the formula: {self.loglike}.'
            )

        self.algorithm = algorithm
        self.algoParameters = algoParameters

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(self.id_manager.free_betas_values)
        xstar, optimizationMessages = output
        # Running time of the optimization algorithm
        optimizationMessages['Optimization time'] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages

        f = self.calculateLikelihood(xstar, scaled=False)

        fgHb = f, None, None, None
        rawResults = res.rawResults(
            self, xstar, fgHb, bootstrap=self.bootstrap_results
        )
        r = res.bioResults(rawResults)
        return r

    def validate(self, estimationResults, validationData):
        """Perform out-of-sample validation.

        The function performs the following tasks:

          - each slice defines a validation set (the slice itself)
            and an estimation set (the rest of the data),
          - the model is re-estimated on the estimation set,
          - the estimated model is applied on the validation set,
          - the value of the log likelihood for each observation is reported.

        :param estimationResults: results of the model estimation based on the
            full data.
        :type estimationResults: biogeme.results.bioResults

        :param validationData: list of estimation and validation data sets
        :type validationData: list(tuple(pandas.DataFrame, pandas.DataFrame))

        :return: a list containing as many items as slices. Each item
                 is the result of the simulation on the validation set.
        :rtype: list(pandas.DataFrame)

        :raises biogemeError: An error is raised if the database is structured
            as panel data.
        """
        if self.database.isPanel():
            raise excep.biogemeError(
                'Validation for panel data is not yet implemented'
            )

        keepDatabase = self.database

        allSimulationResults = []
        for v in validationData:
            # v[0] is the estimation data set
            database = db.Database('Estimation data', v.estimation)
            self.loglike.changeInitValues(estimationResults.getBetaValues())
            estBiogeme = BIOGEME(database, self.loglike)
            results = estBiogeme.estimate()
            simulate = {'Loglikelihood': self.loglike}
            simBiogeme = BIOGEME(
                db.Database('Validation data', v.validation), simulate
            )
            simResult = simBiogeme.simulate(results.getBetaValues())
            allSimulationResults.append(simResult)

        self.database = keepDatabase
        if self.generatePickle:
            fname = f'{self.modelName}_validation'
            pickleFileName = bf.getNewFileName(fname, 'pickle')
            with open(pickleFileName, 'wb') as f:
                pickle.dump(allSimulationResults, f)
            logger.general(
                f'Simulation results saved in file {pickleFileName}'
            )

        return allSimulationResults

    def optimize(self, startingValues=None):
        """Calls the optimization algorithm. The function self.algorithm
        is called.

        :param startingValues: starting point for the algorithm
        :type startingValues: list(float)

        :return: x, messages

           - x is the solution generated by the algorithm,
           - messages is a dictionary describing several information about the
             algorithm

        :rtype: numpay.array, dict(str:object)


        :raises biogemeError: an error is raised if no algorithm is specified.
        """
        theFunction = negLikelihood(
            like=self.calculateLikelihood,
            like_deriv=self.calculateLikelihoodAndDerivatives,
            scaled=True,
        )

        if startingValues is None:
            startingValues = self.id_manager.free_betas_values

        if self.algorithm is None:
            err = (
                'An algorithm must be specified. The CFSQP algorithm '
                'is not available anymore.'
            )
            raise excep.biogemeError(err)

        results = self.algorithm(
            theFunction,
            startingValues,
            self.id_manager.bounds,
            self.algoParameters,
        )

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

        if theBetaValues is None:
            betaValues = self.id_manager.free_betas_values
        else:
            if not isinstance(theBetaValues, dict):
                err = (
                    'Deprecated. A dictionary must be provided. '
                    'It can be obtained from results.getBetaValues()'
                )
                raise excep.biogemeError(err)
            for x in theBetaValues.keys():
                if x not in self.id_manager.free_betas.names:
                    logger.warning(f'Parameter {x} not present in the model')
            betaValues = []
            for i, x in enumerate(self.id_manager.free_betas.names):
                if x in theBetaValues:
                    betaValues.append(theBetaValues[x])
                else:
                    logger.warning(
                        f'Simulation: initial value of {x} not provided.'
                    )
                    betaValues.append(self.id_manager.free_betas_values[i])

        if self.database.isPanel():
            for f in self.formulas.values():
                count = f.countPanelTrajectoryExpressions()
                if count != 1:
                    theError = (
                        f'For panel data, the expression must '
                        f'contain exactly one PanelLikelihoodTrajectory '
                        f'operator. It contains {count}: {f}'
                    )
                    raise excep.biogemeError(theError)

        output = pd.DataFrame(index=self.database.data.index)
        formulas_signature = [v.getSignature() for v in self.formulas.values()]

        if self.database.isPanel():
            self.database.buildPanelMap()
            self.theC.setDataMap(self.database.individualMap)

        for v in self.formulas.values():
            listOfErrors, listOfWarnings = v.audit(database=self.database)
            if listOfWarnings:
                logger.warning('\n'.join(listOfWarnings))
            if listOfErrors:
                logger.warning('\n'.join(listOfErrors))
                raise excep.biogemeError('\n'.join(listOfErrors))

        result = self.theC.simulateSeveralFormulas(
            formulas_signature,
            betaValues,
            self.id_manager.fixed_betas_values,
            self.database.data,
            self.numberOfThreads,
        )
        for key, r in zip(self.formulas.keys(), result):
            output[key] = r
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
                # Retrieve the names of the betas parameters that have been
                # estimated
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
        """Creates a log file with the messages produced by Biogeme.

        The name of the file is the name of the model with an extension .log

        :param verbosity: types of messages to be captured

            - 0: no output
            - 1: warnings
            - 2: only general information
            - 3: more verbose
            - 4: debug messages

            Default: 3.

        :type verbosity: int

        """
        logger.createLog(fileLevel=verbosity, fileName=self.modelName)

    def __str__(self):
        r = f'{self.modelName}: database [{self.database.name}]'
        r += str(self.formulas)
        print(r)
        return r

    def files_of_type(self, extension, all_files=False):
        """Identify the list of files with a given extension in the
        local directory

        :param extension: extension of the requested files (without
            the dot): 'pickle', or 'html'
        :type extension: str
        :param all_files: if all_files is False, only files containing
            the name of the model are identified. If all_files is
            True, all files with the requested extension are
            identified.
        :type all_files: bool

        :return: list of files with the requested extension.
        :rtype: list(str)
        """
        if all_files:
            pattern = f'*.{extension}'
            return glob.glob(pattern)
        pattern1 = f'{self.modelName}.{extension}'
        pattern2 = f'{self.modelName}~*.{extension}'
        files = glob.glob(pattern1) + glob.glob(pattern2)
        return files


class negLikelihood(functionToMinimize):
    """Provides the value of the function to be minimized, as well as its
    derivatives. To be used by the opimization package.

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, like, like_deriv, scaled):
        """Constructor"""
        self.recalculate = True
        """True if the log likelihood must be recalculated
        """

        self.x = None  #: Vector of unknown parameters values

        self.batch = None
        """Value betwen 0 and 1 defining the size of the batch, that is the
        percentage of the data that should be used to approximate the
        log likelihood.
        """

        self.fv = None  #: value of the function

        self.gv = None  #: vector with the gradient

        self.hv = None  #: second derivatives matrix

        self.bhhhv = None  #: BHHH matrix

        self.like = like  #: function calculating the log likelihood

        self.like_deriv = like_deriv
        """function calculating the log likelihood and its derivatives.
        """

        self.scaled = scaled
        """if True, the value of the log likelihood is divided by the number
        of observations used to calculate it. In this case, the values
        with different sample sizes are comparable.
        """

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
            self.fv, self.gv, *_ = self.like_deriv(
                self.x, self.scaled, hessian=False, bhhh=False, batch=batch
            )
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

            self.fv, self.gv, self.hv, _ = self.like_deriv(
                self.x, self.scaled, hessian=True, bhhh=False, batch=batch
            )
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
            self.fv, self.gv, _, self.bhhhv = self.like_deriv(
                self.x, self.scaled, hessian=False, bhhh=True, batch=batch
            )
            self.hv = None

        return (-self.fv, -self.gv, -self.bhhhv)
