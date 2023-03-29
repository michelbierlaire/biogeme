"""
Implementation of the main Biogeme class that combines the database
and the model specification.

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019
"""

import logging
import glob
from collections import namedtuple
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
import biogeme.optimization as opt
from biogeme import tools
from biogeme.idmanager import IdManager
from biogeme.negative_likelihood import NegativeLikelihood
from biogeme import toml
from biogeme.multiple_expressions import (
    string_id_to_configuration,
    configuration_to_string_id,
)

# import yep

DEFAULT_MODEL_NAME = 'biogemeModelDefaultName'
logger = logging.getLogger(__name__)

OldNewParamTuple = namedtuple('OldNewParamTuple', 'old new section')


class BIOGEME:
    """Main class that combines the database and the model
        specification.

    It works in two modes: estimation and simulation.

    """

    def __init__(
        self,
        database,
        formulas,
        userNotes=None,
        parameter_file=None,
        skip_audit=False,
        **kwargs,
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

        :param parameter_file: name of the .toml file where the parameters are read
        :type parameter_file: str

        :raise biogemeError: an audit of the formulas is performed.
           If a formula has issues, an error is detected and an
           exception is raised.

        """
        self.toml = toml.Toml(parameter_file)
        self.skip_audit = skip_audit

        # Code for the transition period to inform the user of the
        # change of parameter management scheme
        old_params = (
            OldNewParamTuple(
                old='numberOfThreads',
                new='number_of_threads',
                section='MultiThreading',
            ),
            OldNewParamTuple(
                old='numberOfDraws',
                new='number_of_draws',
                section='MonteCarlo',
            ),
            OldNewParamTuple(old='seed', new='seed', section='MonteCarlo'),
            OldNewParamTuple(
                old='missingData', new='missing_data', section='Specification'
            ),
        )

        for the_param in old_params:
            value = kwargs.get(the_param.old)
            if value is not None:
                BIOGEME.argument_warning(the_param)
                self.toml.parameters.set_value(
                    the_param.new,
                    value,
                    the_param.section,
                )
        obsolete = ('suggestScales',)
        for the_param in obsolete:
            value = kwargs.get(the_param)
            if value is not None:
                warning_msg = f'Parameter {the_param} is obsolete and ignored.'
                logger.warning(warning_msg)

        self._algorithm = opt.algorithms.get(self.algorithm_name)
        self.algoParameters = None

        if not self.skip_audit:
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

        self.modelName = DEFAULT_MODEL_NAME
        """Name of the model. Default: 'biogemeModelDefaultName'
        """
        self.monteCarlo = False
        """ ``monteCarlo`` is True if one of the expressions involves a
        Monte-Carlo integration.
        """

        if self.seed_param != 0:
            np.random.seed(self.seed_param)

        self.database = database  #: :class:`biogeme.database.Database` object

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

        self._prepareDatabaseForFormula()

        if not self.skip_audit:
            self._audit()

        self.reset_id_manager()

        self.theC = cb.pyBiogeme(self.id_manager.number_of_free_betas)
        if self.database.isPanel():
            self.theC.setPanel(True)
            self.theC.setDataMap(self.database.individualMap)
        # Transfer the data to the C++ formula
        self.theC.setData(self.database.data)
        self.theC.setMissingData(self.missingData)

        start_time = datetime.now()
        self._generateDraws(self.numberOfDraws)
        if self.monteCarlo:
            self.theC.setDraws(self.database.theDraws)
        self.drawsProcessingTime = datetime.now() - start_time
        """ Time needed to generate the draws. """

        self.reset_id_manager()
        if self.loglike is not None:
            self.loglikeSignatures = self.loglike.getSignature()
            """Internal signature of the formula for the
            loglikelihood."""
            if self.weight is None:
                self.theC.setExpressions(self.loglikeSignatures, self.numberOfThreads)
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

        self.bestIteration = None  #: Store the best iteration found so far.

    # @staticmethod
    def argument_warning(old_new_tuple):
        """Displays a deprecation warning when parameters are provided
        as arguments."""
        warning_msg = (
            f'The use of argument {old_new_tuple.old} in the constructor of the '
            f'BIOGEME object is deprecated and will be removed in future '
            f'versions of Biogeme. Instead, define parameter {old_new_tuple.new} '
            f'in section {old_new_tuple.section}'
            f'of the .toml parameter file. The default file name is '
            f'{toml.DEFAULT_FILE_NAME}'
        )
        logger.warning(warning_msg)

    @property
    def maximum_number_catalog_expressions(self):
        """Maximum number of multiple expressions when Catalog's are used."""
        return self.toml.parameters.get_value(
            name='maximum_number_catalog_expressions', section='Estimation'
        )

    @maximum_number_catalog_expressions.setter
    def maximum_number_catalog_expressions(self, value):
        self.toml.parameters.set_value(
            name='maximum_number_catalog_expressions',
            value=value,
            section='Estimation',
        )

    @property
    def algorithm_name(self):
        """Name of the optimization algorithm"""
        return self.toml.parameters.get_value(
            name='optimization_algorithm', section='Estimation'
        )

    @algorithm_name.setter
    def algorithm_name(self, value):
        self.toml.parameters.set_value(
            name='optimization_algorithm',
            value=value,
            section='Estimation',
        )
        self._algorithm = opt.algorithms.get(self.algorithm_name)

    @property
    def identification_threshold(self):
        """Threshold for the eigenvalue to trigger an identification warning"""
        return self.toml.parameters.get_value(
            name='identification_threshold', section='Output'
        )

    @identification_threshold.setter
    def identification_threshold(self, value):
        self.toml.parameters.set_value(
            name='identification_threshold',
            value=value,
            section='Output',
        )

    @property
    def seed_param(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value('seed')

    @seed_param.setter
    def seed_param(self, value):
        self.toml.parameters.set_value(name='seed', value=value, section='MonteCarlo')

    @property
    def saveIterations(self):
        """If True, the current iterate is saved after each iteration, in a
        file named ``__[modelName].iter``, where ``[modelName]`` is the
        name given to the model. If such a file exists, the starting
        values for the estimation are replaced by the values saved in
        the file.
        """
        return self.toml.parameters.get_value(
            name='save_iterations', section='Estimation'
        )

    @saveIterations.setter
    def saveIterations(self, value):
        self.toml.parameters.set_value(
            name='save_iterations', value=value, section='Estimation'
        )

    @property
    def save_iterations(self):
        """Same as saveIterations, with another syntax"""
        return self.toml.parameters.get_value(
            name='save_iterations', section='Estimation'
        )

    @save_iterations.setter
    def save_iterations(self, value):
        self.toml.parameters.set_value(
            name='save_iterations', value=value, section='Estimation'
        )

    @property
    def missingData(self):
        """Code for missing data"""
        return self.toml.parameters.get_value(
            name='missing_data', section='Specification'
        )

    @missingData.setter
    def missingData(self, value):
        self.toml.parameters.set_value(
            name='missing_data', value=value, section='Specification'
        )

    @property
    def missing_data(self):
        """Code for missing data"""
        return self.toml.parameters.get_value(
            name='missing_data', section='Specification'
        )

    @missing_data.setter
    def missing_data(self, value):
        self.toml.parameters.set_value(
            name='missing_data', value=value, section='Specification'
        )

    @property
    def numberOfThreads(self):
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        """
        nbr_threads = self.toml.parameters.get_value('number_of_threads')
        return mp.cpu_count() if nbr_threads == 0 else nbr_threads

    @numberOfThreads.setter
    def numberOfThreads(self, value):
        self.toml.parameters.set_value(
            name='number_of_threads', value=value, section='MultiThreading'
        )

    @property
    def number_of_threads(self):
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        """
        nbr_threads = self.toml.parameters.get_value('number_of_threads')
        return mp.cpu_count() if nbr_threads == 0 else nbr_threads

    @number_of_threads.setter
    def number_of_threads(self, value):
        self.toml.parameters.set_value(
            name='number_of_threads', value=value, section='MultiThreading'
        )

    @property
    def numberOfDraws(self):
        """Number of draws for Monte-Carlo integration."""
        return self.toml.parameters.get_value('number_of_draws')

    @numberOfDraws.setter
    def numberOfDraws(self, value):
        self.toml.parameters.set_value(
            name='number_of_draws', value=value, section='MonteCarlo'
        )

    @property
    def number_of_draws(self):
        """Number of draws for Monte-Carlo integration."""
        return self.toml.parameters.get_value('number_of_draws')

    @number_of_draws.setter
    def number_of_draws(self, value):
        self.toml.parameters.set_value(
            name='number_of_draws', value=value, section='MonteCarlo'
        )

    @property
    def only_robust_stats(self):
        """True if only the robust statistics need to be reported. If
        False, the statistics from the Rao-Cramer bound are also reported.

        """
        return self.toml.parameters.get_value('only_robust_stats')

    @only_robust_stats.setter
    def only_robust_stats(self, value):
        self.toml.parameters.set_value(
            name='only_robust_stats', value=value, section='Output'
        )

    @property
    def generateHtml(self):
        logger.warning('Obsolete syntax. Use generate_html instead of generateHtml')
        """Boolean variable, True if the HTML file with the results must
        be generated.
        """
        return self.toml.parameters.get_value('generate_html')

    @generateHtml.setter
    def generateHtml(self, value):
        logger.warning('Obsolete syntax. Use generate_html instead of generateHtml')
        self.toml.parameters.set_value(
            name='generate_html', value=value, section='Output'
        )

    @property
    def generate_html(self):
        """Boolean variable, True if the HTML file with the results must
        be generated.
        """
        return self.toml.parameters.get_value('generate_html')

    @generate_html.setter
    def generate_html(self, value):
        self.toml.parameters.set_value(
            name='generate_html', value=value, section='Output'
        )

    @property
    def generatePickle(self):
        """Boolean variable, True if the PICKLE file with the results must
        be generated.
        """
        logger.warning('Obsolete syntax. Use generate_pickle instead of generatePickle')
        return self.toml.parameters.get_value('generate_pickle')

    @generatePickle.setter
    def generatePickle(self, value):
        logger.warning('Obsolete syntax. Use generate_pickle instead of generatePickle')
        self.toml.parameters.set_value(
            name='generate_pickle', value=value, section='Output'
        )

    @property
    def generate_pickle(self):
        """Boolean variable, True if the PICKLE file with the results must
        be generated.
        """
        return self.toml.parameters.get_value('generate_pickle')

    @generate_pickle.setter
    def generate_pickle(self, value):
        self.toml.parameters.set_value(
            name='generate_pickle', value=value, section='Output'
        )

    @property
    def tolerance(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(name='tolerance', section='SimpleBounds')

    @tolerance.setter
    def tolerance(self, value):
        self.toml.parameters.set_value(
            name='tolerance', value=value, section='SimpleBounds'
        )

    @property
    def second_derivatives(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(
            name='second_derivatives', section='SimpleBounds'
        )

    @second_derivatives.setter
    def second_derivatives(self, value):
        self.toml.parameters.set_value(
            name='second_derivatives', value=value, section='SimpleBounds'
        )

    @property
    def infeasible_cg(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(
            name='infeasible_cg', section='SimpleBounds'
        )

    @infeasible_cg.setter
    def infeasible_cg(self, value):
        self.toml.parameters.set_value(
            name='infeasible_cg', value=value, section='SimpleBounds'
        )

    @property
    def initial_radius(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(
            name='initial_radius', section='SimpleBounds'
        )

    @initial_radius.setter
    def initial_radius(self, value):
        self.toml.parameters.set_value(
            name='initial_radius', value=value, section='SimpleBounds'
        )

    @property
    def steptol(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(name='steptol', section='SimpleBounds')

    @steptol.setter
    def steptol(self, value):
        self.toml.parameters.set_value(
            name='steptol', value=value, section='SimpleBounds'
        )

    @property
    def enlarging_factor(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(
            name='enlarging_factor', section='SimpleBounds'
        )

    @enlarging_factor.setter
    def enlarging_factor(self, value):
        self.toml.parameters.set_value(
            name='enlarging_factor', value=value, section='SimpleBounds'
        )

    @property
    def maxiter(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(
            name='max_iterations', section='SimpleBounds'
        )

    @maxiter.setter
    def maxiter(self, value):
        self.toml.parameters.set_value(
            name='max_iterations', value=value, section='SimpleBounds'
        )

    @property
    def dogleg(self):
        """getter for the parameter"""
        return self.toml.parameters.get_value(name='dogleg', section='TrustRegion')

    @dogleg.setter
    def dogleg(self, value):
        self.toml.parameters.set_value(
            name='dogleg', value=value, section='TrustRegion'
        )

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

    def _set_algorithm_parameters(self):
        """Retrieve the function with the optimization algorithm from its name"""
        if self.algorithm_name == 'simple_bounds':
            self.algoParameters = {
                'proportionAnalyticalHessian': self.second_derivatives,
                'infeasibleConjugateGradient': self.infeasible_cg,
                'radius': self.initial_radius,
                'enlargingFactor': self.enlarging_factor,
                'steptol': self.steptol,
                'tolerance': self.tolerance,
                'maxiter': self.maxiter,
            }
        elif self.algorithm_name in ['TR-newton', 'TR-BFGS']:
            self.algoParameters = {
                'dogleg': self.dogleg,
                'radius': self.initial_radius,
                'tolerance': self.tolerance,
                'maxiter': self.maxiter,
            }
        else:
            self.algoParameters = None

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

    def _prepareDatabaseForFormula(self):
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

        :raises biogemeError: if calculatation with batch is requested
        """

        if batch is not None:
            raise excep.biogemeError('Calculation with batch not yet implemented')

        if len(x) != len(self.id_manager.free_betas_values):
            error_msg = (
                f'Input vector must be of length '
                f'{len(self.id_manager.free_betas_values)} and '
                f'not {len(x)}'
            )
            raise ValueError(error_msg)

        self._prepareDatabaseForFormula()
        f = self.theC.calculateLikelihood(x, self.id_manager.fixed_betas_values)

        logger.debug(f'Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}')

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
        :raises biogemeError: if calculatation with batch is requested
        """
        if batch is not None:
            raise excep.biogemeError('Calculation with batch not yet implemented')

        n = len(x)
        if n != self.id_manager.number_of_free_betas:
            error_msg = (
                f'Input vector must be of length '
                f'{self.id_manager.number_of_free_betas} and not {len(x)}'
            )
            raise ValueError(error_msg)
        self._prepareDatabaseForFormula()

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
        logger.debug(
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
                with open(
                    self._saveIterationsFileName(),
                    'w',
                    encoding='utf-8',
                ) as pf:
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

    def checkDerivatives(self, beta, verbose=False):
        """Verifies the implementation of the derivatives.

        It compares the analytical version with the finite differences
        approximation.

        :param x: vector of values for the parameters.
        :type x: list(float)

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
            np.asarray(beta),
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
            with open(filename, encoding='utf-8') as fp:
                for line in fp:
                    ell = line.split('=')
                    betas[ell[0].strip()] = float(ell[1])
            self.changeInitValues(betas)
            logger.info(f'Parameter values restored from {filename}')
        except IOError:
            logger.info(f'Cannot read file {filename}. Statement is ignored.')

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

    def estimate_catalog(
        self,
        selected_configurations=None,
        quick_estimate=False,
        recycle=False,
        bootstrap=0,
    ):
        """Estimate all or selected versions of a model with Catalog's,
        corresponding to multiple specifications.

        :param selected_configurations: set of configurations. If
        None, all configurations are considered.
        :type selected_configurations: set(biogeme.pareto.SetElement)

        :param quick_estimate: if True, the final statistics are not calculated.
        :type quick_estimate: bool

        :param recycle: if True, the results are read from the pickle
            file, if it exists. If False, the estimation is performed.
        :type recycle: bool

        :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
        :type bootstrap: int

        :return: object containing the estimation results associated
            with the name of each specification, as well as a
            description of each configuration
        :rtype: dict(str: bioResults)

        """
        logger.debug('ESTIMATE CATALOG')
        if self.loglike is None:
            raise excep.biogemeError('No log likelihood function has been specified')

        if selected_configurations is None:
            number_of_specifications = self.loglike.number_of_multiple_expressions()
#            logger.debug(f'{number_of_specifications=}')
#            logger.debug(f'{self.maximum_number_catalog_expressions=}')

            if number_of_specifications > self.maximum_number_catalog_expressions:
                error_msg = (
                    f'There are about {number_of_specifications} different '
                    f'specifications for the log likelihood function. This is '
                    f'above the maximum number: '
                    f'{self.maximum_number_catalog_expressions}. Either simplify '
                    f'the specification, or change the value of the parameter '
                    f'maximum_number_catalog_expressions.'
                )
                raise excep.valueOutOfRange(error_msg)

            the_iterator = iter(self.loglike)
        else:
            the_iterator = eb.SelectedExpressionsIterator(
                self.loglike, selected_configurations
            )
        configurations = {}
        for config, expression in the_iterator:
            config_id = configuration_to_string_id(config)
            b = BIOGEME(self.database, expression)
            b.modelName = config_id
            if quick_estimate:
                results = b.quickEstimate(recycle=recycle, bootstrap=bootstrap)
            else:
                results = b.estimate(recycle=recycle, bootstrap=bootstrap)

            configurations[config_id] = results
        return configurations

    def estimate(
        self,
        recycle=False,
        bootstrap=0,
        **kwargs,
    ):
        """Estimate the parameters of the model(s).

        :param recycle: if True, the results are read from the pickle
            file, if it exists. If False, the estimation is performed.
        :type recycle: bool

        :param bootstrap: number of bootstrap resampling used to
               calculate the variance-covariance matrix using
               bootstrapping. If the number is 0, bootstrapping is not
               applied. Default: 0.
        :type bootstrap: int

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

        if kwargs.get('algorithm') is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                'of the TOML parameter file'
            )
            raise excep.biogemeError(error_msg)

        if kwargs.get('algoParameters') is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                'of the TOML parameter file'
            )
            raise excep.biogemeError(error_msg)

        if self.modelName == DEFAULT_MODEL_NAME:
            logger.warning(
                f'You have not defined a name for the model. '
                f'The output files are named from the model name. '
                f'The default is [{DEFAULT_MODEL_NAME}]'
            )

        self._set_algorithm_parameters()

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
            warning_msg = 'Recycling was requested, but no pickle file was found'
            logger.warning(warning_msg)
        if self.loglike is None:
            raise excep.biogemeError('No log likelihood function has been specified')
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.biogemeError(
                f'There is no parameter to estimate' f' in the formula: {self.loglike}.'
            )

        if self.saveIterations:
            logger.info(
                f'*** Initial values of the parameters are '
                f'obtained from the file {self._saveIterationsFileName()}'
            )
            self._loadSavedIteration()

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
                    'Numerical problems with finite ' 'difference hessian as well.'
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

            logger.info(f'Re-estimate the model {bootstrap} times for bootstrapping')
            self.bootstrap_results = np.empty(shape=[bootstrap, len(xstar)])
            current_logger_level = logger.level
            logger.setLevel(logging.WARNING)
            for b in tqdm.tqdm(range(bootstrap), disable=False):
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
            logger.setLevel(current_logger_level)
        rawResults = res.rawResults(self, xstar, fgHb, bootstrap=self.bootstrap_results)
        r = res.bioResults(
            rawResults, identification_threshold=self.identification_threshold
        )
        if self.generate_html:
            r.writeHtml(self.only_robust_stats)
        if self.generate_pickle:
            r.writePickle()
        return r

    def quickEstimate(self, **kwargs):
        """| Estimate the parameters of the model. Same as estimate, where any
             extra calculation is skipped (init loglikelihood,
             t-statistics, etc.)

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
        if kwargs.get('algorithm') is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                'of the TOML parameter file'
            )
            raise excep.biogemeError(error_msg)

        if kwargs.get('algoParameters') is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                'of the TOML parameter file'
            )
            raise excep.biogemeError(error_msg)

        if self.loglike is None:
            raise excep.biogemeError('No log likelihood function has been specified')
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.biogemeError(
                f'There is no parameter to estimate' f' in the formula: {self.loglike}.'
            )

        self._set_algorithm_parameters()

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
            self,
            xstar,
            fgHb,
            bootstrap=self.bootstrap_results,
        )
        r = res.bioResults(
            rawResults, identification_threshold=self.identification_threshold
        )
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
            raise excep.biogemeError('Validation for panel data is not yet implemented')

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
                db.Database('Validation data', v.validation),
                simulate,
            )
            simResult = simBiogeme.simulate(results.getBetaValues())
            allSimulationResults.append(simResult)

        self.database = keepDatabase
        if self.generatePickle:
            fname = f'{self.modelName}_validation'
            pickleFileName = bf.getNewFileName(fname, 'pickle')
            with open(pickleFileName, 'wb') as f:
                pickle.dump(allSimulationResults, f)
            logger.info(f'Simulation results saved in file {pickleFileName}')

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
        theFunction = NegativeLikelihood(
            like=self.calculateLikelihood,
            like_deriv=self.calculateLikelihoodAndDerivatives,
            scaled=True,
        )

        if startingValues is None:
            startingValues = self.id_manager.free_betas_values

        if self._algorithm is None:
            err = (
                'An algorithm must be specified. The CFSQP algorithm '
                'is not available anymore.'
            )
            raise excep.biogemeError(err)

        logger.debug(f'Run {self.algorithm_name}')
        #logger.debug(''.join(traceback.format_stack()))
        results = self._algorithm(
            theFunction,
            startingValues,
            self.id_manager.bounds,
            self.algoParameters,
        )

        return results

    def beta_values_dict_to_list(self, beta_dict):
        """Transforms a dict with the names of the betas associated
            with their values, into a list consistent with the
            numbering of the ids.

        :param beta_dict: dict with the values of  the parameters
        :type beta_dict: dict(str: float)

        :raises biogemeError: if the parameter is not a dict

        :raises biogemeError: if a parameter is missing in the dict
        """
        if not isinstance(beta_dict, dict):
            err = (
                'A dictionary must be provided. '
                'It can be obtained from results.getBetaValues()'
            )
            raise excep.biogemeError(err)
        for x in beta_dict.keys():
            if x not in self.id_manager.free_betas.names:
                logger.warning(f'Parameter {x} not present in the model.')

        beta_list = []
        for x in self.id_manager.free_betas.names:
            v = beta_dict.get(x)
            if v is None:
                err = f'Incomplete dict. The value of {x} is not provided.'
                raise excep.biogemeError(err)

            beta_list.append(v)
        return beta_list

    def simulate(self, theBetaValues):
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

        :raises biogemeError: if theBetaValues is None.
        """

        if theBetaValues is None:
            err = (
                'Contrarily to previous versions of Biogeme, '
                'the values of beta must '
                'now be explicitly mentioned. They can be obtained from '
                'results.getBetaValues()'
            )
            raise excep.biogemeError(err)

        betaValues = self.beta_values_dict_to_list(theBetaValues)

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
