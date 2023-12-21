"""
Implementation of the main Biogeme class

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

It combines the database and the model specification.
"""

import logging
import glob
import difflib
from typing import NamedTuple
import multiprocessing as mp
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import tqdm

import cythonbiogeme.cythonbiogeme as cb

from biogeme.configuration import Configuration
import biogeme.database as db
import biogeme.expressions as eb
import biogeme.results as res
import biogeme.exceptions as excep
import biogeme.filenames as bf
import biogeme.optimization as opt
from biogeme import tools
from biogeme.expressions import IdManager
from biogeme.negative_likelihood import NegativeLikelihood
from biogeme.parameters import (
    biogeme_parameters,
    DEFAULT_FILE_NAME as DEFAULT_PARAMETER_FILE_NAME,
)

DEFAULT_MODEL_NAME = "biogemeModelDefaultName"
logger = logging.getLogger(__name__)


class OldNewParamTuple(NamedTuple):
    old: str
    new: str
    section: str


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

        :raise BiogemeError: an audit of the formulas is performed.
           If a formula has issues, an error is detected and an
           exception is raised.

        """
        self.parameter_file = parameter_file
        biogeme_parameters.read_file(parameter_file)
        self.skip_audit = skip_audit

        # Code for the transition period to inform the user of the
        # change of parameter management scheme
        old_params = (
            OldNewParamTuple(
                old="numberOfThreads",
                new="number_of_threads",
                section="MultiThreading",
            ),
            OldNewParamTuple(
                old="numberOfDraws",
                new="number_of_draws",
                section="MonteCarlo",
            ),
            OldNewParamTuple(old="seed", new="seed", section="MonteCarlo"),
            OldNewParamTuple(
                old="missingData", new="missing_data", section="Specification"
            ),
        )

        for the_param in old_params:
            value = kwargs.get(the_param.old)
            if value is not None:
                BIOGEME.argument_warning(old_new_tuple=the_param)
                biogeme_parameters.set_value(
                    the_param.new,
                    value,
                    the_param.section,
                )
        obsolete = ("suggestScales",)
        for the_param in obsolete:
            value = kwargs.get(the_param)
            if value is not None:
                warning_msg = f"Parameter {the_param} is obsolete and ignored."
                logger.warning(warning_msg)

        self._algorithm = opt.algorithms.get(self.algorithm_name)
        self.algo_parameters = None
        self.function_parameters = None

        if not self.skip_audit:
            database.data = database.data.replace({True: 1, False: 0})
            list_of_errors, list_of_warnings = database._audit()
            if list_of_warnings:
                logger.warning("\n".join(list_of_warnings))
            if list_of_errors:
                logger.warning("\n".join(list_of_errors))
                raise excep.BiogemeError("\n".join(list_of_errors))

        self.loglikeName = "loglike"
        """ Keyword used for the name of the loglikelihood formula.
        Default: 'loglike'"""

        self.weightName = "weight"
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

        self.database = database

        self.short_names: tools.ModelNames = None
        if not isinstance(formulas, dict):
            if not isinstance(formulas, eb.Expression):
                raise excep.BiogemeError(
                    f"Expression {formulas} is not of type "
                    f"biogeme.expressions.Expression. "
                    f"It is of type {type(formulas)}"
                )

            self.loglike = formulas
            """ Object of type :class:`biogeme.expressions.Expression`
            calculating the formula for the loglikelihood
            """

            if self.database.isPanel():
                check_variables = self.loglike.check_panel_trajectory()

                if check_variables:
                    err_msg = (
                        f"Error in the loglikelihood function. "
                        f"Some variables are not inside PanelLikelihoodTrajectory: "
                        f"{check_variables} ."
                        f"If the database is organized as panel data, "
                        f"all variables must be used inside a "
                        f"PanelLikelihoodTrajectory. "
                        f"If it is not consistent with your model, generate a flat "
                        f"version of the data using the function "
                        f"`generateFlatPanelDataframe`."
                    )
                    raise excep.BiogemeError(err_msg)

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
                    raise excep.BiogemeError(
                        f'Expression for "{k}" is not of type '
                        f"biogeme.expressions.Expression. "
                        f"It is of type {type(f)}"
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

        self.convergence = False
        """ True if the algorithm has converged

        """

        self.bestIteration = None  #: Store the best iteration found so far.

    @classmethod
    def from_configuration(
        cls,
        config_id,
        expression,
        database,
        user_notes=None,
        parameter_file=None,
        skip_audit=False,
    ):
        """Obtain the Biogeme object corresponding to the
        configuration of a multiple expression

        :param config_id: identifier of the configuration
        :type config_id: strftime

        :param expression: multiple expression containing all the catalogs.
        :type expression: biogeme.expression.Expression

        :param database: database to be passed to the Biogeme object
        :type database: biogeme.database.Database

        :param user_notes: these notes will be included in the report file.
        :type user_notes: str

        :param parameter_file: name of the TOML file with the parameters
        :type parameter_file: str

        :param skip_audit: if True, no auditing is performed.
        :type skip_audit: bool
        """
        if expression.set_of_configurations():
            # We verify that the configuration is valid
            the_set = {
                expr.get_string_id() for expr in expression.set_of_configurations()
            }
            if not the_set:
                error_msg = "No configuration found in the expression"
                raise excep.BiogemeError(error_msg)
            if config_id not in the_set:
                close_matches = difflib.get_close_matches(config_id, the_set)
                if close_matches:
                    error_msg = (
                        f"Unknown configuration: [{config_id}]. "
                        f"Did you mean [{close_matches[0]}]?"
                    )
                else:
                    error_msg = f"Unknown configuration: {config_id}."
                raise excep.BiogemeError(error_msg)
        the_configuration = Configuration.from_string(config_id)
        expression.configure_catalogs(the_configuration)
        if user_notes is None:
            user_notes = the_configuration.get_html()
        return cls(
            database=database,
            formulas=expression,
            userNotes=user_notes,
            parameter_file=parameter_file,
            skip_audit=skip_audit,
        )

    @staticmethod
    def argument_warning(old_new_tuple):
        """Displays a deprecation warning when parameters are provided
        as arguments."""
        warning_msg = (
            f"The use of argument {old_new_tuple.old} in the constructor of the "
            f"BIOGEME object is deprecated and will be removed in future "
            f"versions of Biogeme. Instead, define parameter {old_new_tuple.new} "
            f"in section {old_new_tuple.section} "
            f"of the .toml parameter file. The default file name is "
            f"{DEFAULT_PARAMETER_FILE_NAME}"
        )
        logger.warning(warning_msg)

    @property
    def bootstrap_samples(self):
        """Number of re-estimation for bootstrap samples"""
        return biogeme_parameters.get_value(
            name="bootstrap_samples", section="Estimation"
        )

    @bootstrap_samples.setter
    def bootstrap_samples(self, value):
        biogeme_parameters.set_value(
            name="bootstrap_samples",
            value=value,
            section="Estimation",
        )

    @property
    def max_number_parameters_to_report(self):
        """Maximum number of parameters to report."""
        return biogeme_parameters.get_value(
            name="max_number_parameters_to_report", section="Estimation"
        )

    @max_number_parameters_to_report.setter
    def max_number_parameters_to_report(self, value):
        biogeme_parameters.set_value(
            name="max_number_parameters_to_report",
            value=value,
            section="Estimation",
        )
        eb.Expression.max_number_parameters_to_report = value

    @property
    def maximum_number_catalog_expressions(self):
        """Maximum number of multiple expressions when Catalog's are used."""
        return biogeme_parameters.get_value(
            name="maximum_number_catalog_expressions", section="Estimation"
        )

    @maximum_number_catalog_expressions.setter
    def maximum_number_catalog_expressions(self, value):
        biogeme_parameters.set_value(
            name="maximum_number_catalog_expressions",
            value=value,
            section="Estimation",
        )
        eb.Expression.maximum_number_of_configurations = value

    @property
    def algorithm_name(self):
        """Name of the optimization algorithm"""
        return biogeme_parameters.get_value(
            name="optimization_algorithm", section="Estimation"
        )

    @algorithm_name.setter
    def algorithm_name(self, value):
        biogeme_parameters.set_value(
            name="optimization_algorithm",
            value=value,
            section="Estimation",
        )
        self._algorithm = opt.algorithms.get(self.algorithm_name)

    @property
    def identification_threshold(self):
        """Threshold for the eigenvalue to trigger an identification warning"""
        return biogeme_parameters.get_value(
            name="identification_threshold", section="Output"
        )

    @identification_threshold.setter
    def identification_threshold(self, value):
        biogeme_parameters.set_value(
            name="identification_threshold",
            value=value,
            section="Output",
        )

    @property
    def seed_param(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value("seed")

    @seed_param.setter
    def seed_param(self, value):
        biogeme_parameters.set_value(name="seed", value=value, section="MonteCarlo")

    @property
    def saveIterations(self):
        """If True, the current iterate is saved after each iteration, in a
        file named ``__[modelName].iter``, where ``[modelName]`` is the
        name given to the model. If such a file exists, the starting
        values for the estimation are replaced by the values saved in
        the file.
        """
        return biogeme_parameters.get_value(
            name="save_iterations", section="Estimation"
        )

    @saveIterations.setter
    def saveIterations(self, value):
        biogeme_parameters.set_value(
            name="save_iterations", value=value, section="Estimation"
        )

    @property
    def save_iterations(self):
        """Same as saveIterations, with another syntax"""
        return biogeme_parameters.get_value(
            name="save_iterations", section="Estimation"
        )

    @save_iterations.setter
    def save_iterations(self, value):
        biogeme_parameters.set_value(
            name="save_iterations", value=value, section="Estimation"
        )

    @property
    def missingData(self):
        """Code for missing data"""
        return biogeme_parameters.get_value(
            name="missing_data", section="Specification"
        )

    @missingData.setter
    def missingData(self, value):
        biogeme_parameters.set_value(
            name="missing_data", value=value, section="Specification"
        )

    @property
    def missing_data(self):
        """Code for missing data"""
        return biogeme_parameters.get_value(
            name="missing_data", section="Specification"
        )

    @missing_data.setter
    def missing_data(self, value):
        biogeme_parameters.set_value(
            name="missing_data", value=value, section="Specification"
        )

    @property
    def numberOfThreads(self):
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        """
        nbr_threads = biogeme_parameters.get_value("number_of_threads")
        return mp.cpu_count() if nbr_threads == 0 else nbr_threads

    @numberOfThreads.setter
    def numberOfThreads(self, value):
        biogeme_parameters.set_value(
            name="number_of_threads", value=value, section="MultiThreading"
        )

    @property
    def number_of_threads(self):
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        """
        nbr_threads = biogeme_parameters.get_value("number_of_threads")
        return mp.cpu_count() if nbr_threads == 0 else nbr_threads

    @number_of_threads.setter
    def number_of_threads(self, value):
        biogeme_parameters.set_value(
            name="number_of_threads", value=value, section="MultiThreading"
        )

    @property
    def numberOfDraws(self):
        """Number of draws for Monte-Carlo integration."""
        return biogeme_parameters.get_value("number_of_draws")

    @numberOfDraws.setter
    def numberOfDraws(self, value):
        biogeme_parameters.set_value(
            name="number_of_draws", value=value, section="MonteCarlo"
        )

    @property
    def number_of_draws(self):
        """Number of draws for Monte-Carlo integration."""
        return biogeme_parameters.get_value("number_of_draws")

    @number_of_draws.setter
    def number_of_draws(self, value):
        biogeme_parameters.set_value(
            name="number_of_draws", value=value, section="MonteCarlo"
        )

    @property
    def only_robust_stats(self):
        """True if only the robust statistics need to be reported. If
        False, the statistics from the Rao-Cramer bound are also reported.

        """
        return biogeme_parameters.get_value("only_robust_stats")

    @only_robust_stats.setter
    def only_robust_stats(self, value):
        biogeme_parameters.set_value(
            name="only_robust_stats", value=value, section="Output"
        )

    @property
    def generateHtml(self):
        """Boolean variable, True if the HTML file with the results must
        be generated.
        """
        logger.warning("Obsolete syntax. Use generate_html instead of generateHtml")
        return biogeme_parameters.get_value("generate_html")

    @generateHtml.setter
    def generateHtml(self, value):
        logger.warning("Obsolete syntax. Use generate_html instead of generateHtml")
        biogeme_parameters.set_value(
            name="generate_html", value=value, section="Output"
        )

    @property
    def generate_html(self):
        """Boolean variable, True if the HTML file with the results must
        be generated.
        """
        return biogeme_parameters.get_value("generate_html")

    @generate_html.setter
    def generate_html(self, value):
        biogeme_parameters.set_value(
            name="generate_html", value=value, section="Output"
        )

    @property
    def generatePickle(self):
        """Boolean variable, True if the PICKLE file with the results must
        be generated.
        """
        logger.warning("Obsolete syntax. Use generate_pickle instead of generatePickle")
        return biogeme_parameters.get_value("generate_pickle")

    @generatePickle.setter
    def generatePickle(self, value):
        logger.warning("Obsolete syntax. Use generate_pickle instead of generatePickle")
        biogeme_parameters.set_value(
            name="generate_pickle", value=value, section="Output"
        )

    @property
    def generate_pickle(self):
        """Boolean variable, True if the PICKLE file with the results must
        be generated.
        """
        return biogeme_parameters.get_value("generate_pickle")

    @generate_pickle.setter
    def generate_pickle(self, value):
        biogeme_parameters.set_value(
            name="generate_pickle", value=value, section="Output"
        )

    @property
    def tolerance(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(name="tolerance", section="SimpleBounds")

    @tolerance.setter
    def tolerance(self, value):
        biogeme_parameters.set_value(
            name="tolerance", value=value, section="SimpleBounds"
        )

    @property
    def second_derivatives(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(
            name="second_derivatives", section="SimpleBounds"
        )

    @second_derivatives.setter
    def second_derivatives(self, value):
        biogeme_parameters.set_value(
            name="second_derivatives", value=value, section="SimpleBounds"
        )

    @property
    def infeasible_cg(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(
            name="infeasible_cg", section="SimpleBounds"
        )

    @infeasible_cg.setter
    def infeasible_cg(self, value):
        biogeme_parameters.set_value(
            name="infeasible_cg", value=value, section="SimpleBounds"
        )

    @property
    def initial_radius(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(
            name="initial_radius", section="SimpleBounds"
        )

    @initial_radius.setter
    def initial_radius(self, value):
        biogeme_parameters.set_value(
            name="initial_radius", value=value, section="SimpleBounds"
        )

    @property
    def steptol(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(name="steptol", section="SimpleBounds")

    @steptol.setter
    def steptol(self, value):
        biogeme_parameters.set_value(
            name="steptol", value=value, section="SimpleBounds"
        )

    @property
    def enlarging_factor(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(
            name="enlarging_factor", section="SimpleBounds"
        )

    @enlarging_factor.setter
    def enlarging_factor(self, value):
        biogeme_parameters.set_value(
            name="enlarging_factor", value=value, section="SimpleBounds"
        )

    @property
    def maxiter(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(
            name="max_iterations", section="SimpleBounds"
        )

    @maxiter.setter
    def maxiter(self, value):
        biogeme_parameters.set_value(
            name="max_iterations", value=value, section="SimpleBounds"
        )

    @property
    def dogleg(self):
        """getter for the parameter"""
        return biogeme_parameters.get_value(name="dogleg", section="TrustRegion")

    @dogleg.setter
    def dogleg(self, value):
        biogeme_parameters.set_value(name="dogleg", value=value, section="TrustRegion")

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
        )
        for f in self.formulas.values():
            f.setIdManager(id_manager=self.id_manager)

    def _set_function_parameters(self):
        """Prepare the parameters for the function"""
        self.function_parameters = {
            "tolerance": self.tolerance,
            "steptol": self.steptol,
        }

    def _set_algorithm_parameters(self):
        """Prepare the parameters for the algorithms"""
        if self.algorithm_name == "simple_bounds":
            self.algo_parameters = {
                "proportionAnalyticalHessian": self.second_derivatives,
                "infeasibleConjugateGradient": self.infeasible_cg,
                "radius": self.initial_radius,
                "enlargingFactor": self.enlarging_factor,
                "maxiter": self.maxiter,
            }
            return
        if self.algorithm_name in ["simple_bounds_newton", "simple_bounds_BFGS"]:
            self.algo_parameters = {
                "infeasibleConjugateGradient": self.infeasible_cg,
                "radius": self.initial_radius,
                "enlargingFactor": self.enlarging_factor,
                "maxiter": self.maxiter,
            }
            return
        if self.algorithm_name in ["TR-newton", "TR-BFGS"]:
            self.algo_parameters = {
                "dogleg": self.dogleg,
                "radius": self.initial_radius,
                "maxiter": self.maxiter,
            }
            return
        if self.algorithm_name in ["LS-newton", "LS-BFGS"]:
            self.algo_parameters = {
                "maxiter": self.maxiter,
            }
            return
        self.algo_parameters = None

    def _saveIterationsFileName(self):
        """
        :return: The name of the file where the iterations are saved.
        :rtype: str
        """
        return f"__{self.modelName}.iter"

    def _audit(self):
        """Each expression provides an audit function, that verifies its
        validity. Each formula is audited, and the list of errors
        and warnings reported.

        :raise BiogemeError: if the formula has issues, an error is
                             detected and an exception is raised.

        """

        list_of_errors = []
        list_of_warnings = []
        for v in self.formulas.values():
            check_draws = v.check_draws()
            if check_draws:
                err_msg = (
                    f"The following draws are defined outside the "
                    f"MonteCarlo operator: {check_draws}"
                )
                list_of_errors.append(err_msg)
            check_rv = v.check_rv()
            if check_rv:
                err_msg = (
                    f"The following random variables are defined "
                    f"outside the Integrate operator: {check_rv}"
                )
                list_of_errors.append(err_msg)
            err, war = v.audit(self.database)
            list_of_errors += err
            list_of_warnings += war
        if self.weight is not None:
            total = self.weight.getValue_c(
                database=self.database, aggregation=True, prepareIds=True
            )
            s_size = self.database.getSampleSize()
            ratio = s_size / total
            if np.abs(ratio - 1) >= 0.01:
                theWarning = (
                    f"The sum of the weights ({total}) is different from "
                    f"the sample size ({self.database.getSampleSize()}). "
                    f"Multiply the weights by {ratio} to reconcile the two."
                )
                list_of_warnings.append(theWarning)
        if list_of_warnings:
            logger.warning("\n".join(list_of_warnings))
        if list_of_errors:
            logger.warning("\n".join(list_of_errors))
            raise excep.BiogemeError("\n".join(list_of_errors))

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
        """Deprecated"""
        logger.warning(
            'The syntax "freeBetaNames" is deprecated and is replaced by '
            'the syntax "free_beta_names".'
        )
        return self.free_beta_names()

    def free_beta_names(self):
        """Returns the names of the parameters that must be estimated

        :return: list of names of the parameters
        :rtype: list(str)
        """
        return self.id_manager.free_betas.names

    def number_unknown_parameters(self):
        """Returns the number of parameters that must be estimated

        :return: number of parameters
        :rtype: int
        """
        return len(self.id_manager.free_betas.names)

    def get_beta_values(self):
        """Returns a dict with the initial values of beta. Typically
            useful for simulation.

        :return: dict with the initial values of the beta
        :rtype: dict(str: float)
        """
        all_betas = {}
        if self.loglike:
            all_betas.update(self.loglike.get_beta_values())

        if self.weight:
            all_betas.update(self.weight.get_beta_values())

        for formula in self.formulas.values():
            all_betas.update(formula.get_beta_values())

        return all_betas

    def getBoundsOnBeta(self, betaName):
        """Returns the bounds on the parameter as defined by the user.

        :param betaName: name of the parameter
        :type betaName: string
        :return: lower bound, upper bound
        :rtype: tuple
        :raises BiogemeError: if the name of the parameter is not found.
        """
        index = self.id_manager.free_betas.indices.get(betaName)
        if index is None:
            raise excep.BiogemeError(f"Unknown parameter {betaName}")
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

        :raises BiogemeError: if calculatation with batch is requested
        """

        if batch is not None:
            raise excep.BiogemeError("Calculation with batch not yet implemented")

        if len(x) != len(self.id_manager.free_betas_values):
            error_msg = (
                f"Input vector must be of length "
                f"{len(self.id_manager.free_betas_values)} and "
                f"not {len(x)}"
            )
            raise ValueError(error_msg)

        self._prepareDatabaseForFormula()
        f = self.theC.calculateLikelihood(x, self.id_manager.fixed_betas_values)

        logger.debug(f"Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}")

        if scaled:
            return f / float(self.database.getSampleSize())

        return f

    def report_array(self, array, with_names=True):
        """Reports the entries of the array up to the maximum number

        :param array: array to report
        :type array: numpy.array

        :param with_names: if True, the names of the parameters are included
        :type with_names: bool

        :return: string reporting the values
        :rtype: str
        """
        length = min(array.size, self.max_number_parameters_to_report)
        if with_names:
            names = self.free_beta_names()
            report = ", ".join(
                [
                    f"{name}={value:.2g}"
                    for name, value in zip(names[:length], array[:length])
                ]
            )
            return report
        report = ", ".join([f"{value:.2g}" for value in array[:length]])
        return report

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

        :raises BiogemeError: if the norm of the gradient is not finite, an
            error is raised.
        :raises BiogemeError: if calculatation with batch is requested
        """
        if batch is not None:
            raise excep.BiogemeError("Calculation with batch not yet implemented")

        n = len(x)
        if n != self.id_manager.number_of_free_betas:
            error_msg = (
                f"Input vector must be of length "
                f"{self.id_manager.number_of_free_betas} and not {len(x)}"
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

        hmsg = ""
        if hessian:
            hmsg = f"Hessian norm:  {np.linalg.norm(h):10.1g}"
        bhhhmsg = ""
        if bhhh:
            bhhhmsg = f"BHHH norm:  {np.linalg.norm(bh):10.1g}"
        gradnorm = np.linalg.norm(g)
        logger.debug(
            f"Log likelihood (N = {self.database.getSampleSize()}): {f:10.7g}"
            f" Gradient norm: {gradnorm:10.1g}"
            f" {hmsg} {bhhhmsg}"
        )

        if not np.isfinite(gradnorm):
            report_x = self.report_array(x)
            report_g = self.report_array(g, with_names=False)
            error_msg = (
                f"The norm of the gradient at {report_x} is {gradnorm}: g={report_g}"
            )
            logger.warning(error_msg)

        elif self.saveIterations:
            if self.bestIteration is None:
                self.bestIteration = f
            if f >= self.bestIteration:
                with open(
                    self._saveIterationsFileName(),
                    "w",
                    encoding="utf-8",
                ) as pf:
                    for i, v in enumerate(x):
                        print(
                            f"{self.id_manager.free_betas.names[i]} = {v}",
                            file=pf,
                        )

        if scaled:
            N = float(self.database.getSampleSize())
            if N == 0:
                raise excep.BiogemeError(f"Sample size is {N}")

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

        def the_function(x):
            f, g, _, _ = self.calculateLikelihoodAndDerivatives(
                x, scaled=False, hessian=False, bhhh=False
            )
            return f, np.asarray(g)

        return tools.findiff_H(the_function, np.asarray(x))

    def checkDerivatives(self, beta, verbose=False):
        """Verifies the implementation of the derivatives.

        It compares the analytical version with the finite differences
        approximation.

        :param beta: vector of values for the parameters.
        :type beta: list(float)

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

        def the_function(x):
            """Wrapper function to use tools.checkDerivatives"""
            f, g, h, _ = self.calculateLikelihoodAndDerivatives(
                x, scaled=False, hessian=True, bhhh=False
            )
            return f, np.asarray(g), np.asarray(h)

        return tools.checkDerivatives(
            the_function,
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
            with open(filename, encoding="utf-8") as fp:
                for line in fp:
                    ell = line.split("=")
                    betas[ell[0].strip()] = float(ell[1])
            self.change_init_values(betas)
            logger.info(f"Parameter values restored from {filename}")
        except IOError:
            logger.info(f"Cannot read file {filename}. Statement is ignored.")

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
        self.change_init_values(randomBetas)

    def change_init_values(self, betas):
        """Modifies the initial values of the pameters in all formula

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        if self.loglike is not None:
            self.loglike.change_init_values(betas)
        if self.weight is not None:
            self.weight.change_init_values(betas)
        for _, f in self.formulas.items():
            f.change_init_values(betas)
        for i, name in enumerate(self.id_manager.free_betas.names):
            value = betas.get(name)
            if value is not None:
                self.id_manager.free_betas_values[i] = value

    def estimate_catalog(
        self,
        selected_configurations=None,
        quick_estimate=False,
        recycle=False,
        run_bootstrap=False,
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

        :param run_bootstrap: if True, bootstrapping is applied.
        :type run_bootstrap: bool

        :return: object containing the estimation results associated
            with the name of each specification, as well as a
            description of each configuration
        :rtype: dict(str: bioResults)

        """
        if self.short_names is None:
            self.short_names = tools.ModelNames(prefix=self.modelName)

        if self.loglike is None:
            raise excep.BiogemeError("No log likelihood function has been specified")

        if selected_configurations is None:
            number_of_specifications = self.loglike.number_of_multiple_expressions()
            logger.info(f"Estimating {number_of_specifications} models.")
            #            logger.debug(f'{number_of_specifications=}')
            # logger.info(f'{self.maximum_number_catalog_expressions=}')

            if (
                number_of_specifications is None
                or number_of_specifications > self.maximum_number_catalog_expressions
            ):
                error_msg = (
                    f"There are too many [{number_of_specifications}] different "
                    f"specifications for the log likelihood function. This is "
                    f"above the maximum number: "
                    f"{self.maximum_number_catalog_expressions}. Simplify "
                    f"the specification, change the value of the parameter "
                    f"maximum_number_catalog_expressions, or consider using "
                    f'the AssistedSpecification object in the "biogeme.assisted" '
                    f"module."
                )
                raise excep.ValueOutOfRange(error_msg)

            the_iterator = iter(self.loglike)
        else:
            the_iterator = eb.SelectedExpressionsIterator(
                self.loglike, selected_configurations
            )
        configurations = {}
        for expression in the_iterator:
            config = expression.current_configuration()
            config_id = config.get_string_id()
            b = BIOGEME.from_configuration(
                config_id=config_id, expression=expression, database=self.database
            )
            b.modelName = self.short_names(config_id)
            b.generate_html = self.generate_html
            b.generate_pickle = self.generate_pickle
            if quick_estimate:
                results = b.quickEstimate(recycle=recycle)
            else:
                results = b.estimate(recycle=recycle, run_bootstrap=run_bootstrap)

            configurations[config_id] = results

        return configurations

    def estimate(
        self,
        recycle=False,
        run_bootstrap=False,
        **kwargs,
    ):
        """Estimate the parameters of the model(s).

        :param recycle: if True, the results are read from the pickle
            file, if it exists. If False, the estimation is performed.
        :type recycle: bool

        :param run_bootstrap: if True, bootstrapping is applied.
        :type run_bootstrap: bool

        :return: object containing the estimation results.
        :rtype: biogeme.bioResults

        Example::

            # Create an instance of biogeme
            biogeme  = bio.BIOGEME(database, logprob)

            # Gives a name to the model
            biogeme.modelName = 'mymodel'

            # Estimate the parameters
            results = biogeme.estimate()

        :raises BiogemeError: if no expression has been provided for the
            likelihood

        """
        if kwargs.get("bootstrap") is not None:
            error_msg = (
                'Parameter "bootstrap" is deprecated. In order to perform '
                "bootstrapping, set parameter run_bootstrap=True in the estimate "
                "function, and specify the number of bootstrap draws in the "
                "biogeme.toml file [e.g. bootstrap_samples=100]."
            )
            raise excep.BiogemeError(error_msg)
        if kwargs.get("algorithm") is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if kwargs.get("algo_parameters") is not None:
            error_msg = (
                'The parameter "algo_parameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if kwargs.get("algoParameters") is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if self.modelName == DEFAULT_MODEL_NAME:
            logger.warning(
                f"You have not defined a name for the model. "
                f"The output files are named from the model name. "
                f"The default is [{DEFAULT_MODEL_NAME}]"
            )

        self._set_function_parameters()
        self._set_algorithm_parameters()

        if recycle:
            pickle_files = self.files_of_type("pickle")
            pickle_files.sort()
            if pickle_files:
                pickle_to_read = pickle_files[-1]
                if len(pickle_files) > 1:
                    warning_msg = (
                        f"Several pickle files are available for "
                        f"this model: {pickle_files}. "
                        f"The file {pickle_to_read} "
                        f"is used to load the results."
                    )
                    logger.warning(warning_msg)
                results = res.bioResults(
                    pickleFile=pickle_to_read,
                    identification_threshold=self.identification_threshold,
                )
                logger.warning(
                    f"Estimation results read from {pickle_to_read}. "
                    f"There is no guarantee that they correspond "
                    f"to the specified model."
                )
                return results
            warning_msg = "Recycling was requested, but no pickle file was found"
            logger.warning(warning_msg)
        if self.loglike is None:
            raise excep.BiogemeError("No log likelihood function has been specified")
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.BiogemeError(
                f"There is no parameter to estimate" f" in the formula: {self.loglike}."
            )

        if self.saveIterations:
            logger.info(
                f"*** Initial values of the parameters are "
                f"obtained from the file {self._saveIterationsFileName()}"
            )
            self._loadSavedIteration()

        self.calculateInitLikelihood()
        self.bestIteration = None

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(np.array(self.id_manager.free_betas_values))
        xstar, optimizationMessages, convergence = output
        # Running time of the optimization algorithm
        optimizationMessages["Optimization time"] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages
        self.convergence = convergence

        fgHb = self.calculateLikelihoodAndDerivatives(
            xstar, scaled=False, hessian=True, bhhh=True
        )
        if not np.isfinite(fgHb[2]).all():
            warning_msg = (
                "Numerical problems in calculating "
                "the analytical hessian. Finite differences"
                " is tried instead."
            )
            logger.warning(warning_msg)
            finDiffHessian = self.likelihoodFiniteDifferenceHessian(xstar)
            if not np.isfinite(fgHb[2]).all():
                logger.warning(
                    "Numerical problems with finite difference hessian as well."
                )
            else:
                fgHb = fgHb[0], fgHb[1], finDiffHessian, fgHb[3]

        # numpy array, of size B x K,
        # where
        #        - B is the number of bootstrap iterations
        #        - K is the number pf parameters to estimate
        self.bootstrap_results = None
        if run_bootstrap:
            # Temporarily stop reporting log messages
            start_time = datetime.now()

            logger.info(
                f"Re-estimate the model {self.bootstrap_samples} "
                f"times for bootstrapping"
            )
            self.bootstrap_results = np.empty(
                shape=[self.bootstrap_samples, len(xstar)]
            )
            current_logger_level = logger.level
            logger.setLevel(logging.WARNING)
            for b in tqdm.tqdm(range(self.bootstrap_samples), disable=False):
                if self.database.isPanel():
                    sample = self.database.sampleIndividualMapWithReplacement()
                    self.theC.setDataMap(sample)
                else:
                    sample = self.database.sampleWithReplacement()
                    self.theC.setData(sample)
                x_br, _, _ = self.optimize(xstar)
                self.bootstrap_results[b] = x_br

            # Time needed to generate the bootstrap results
            self.bootstrap_time = datetime.now() - start_time
            logger.setLevel(current_logger_level)
        rawResults = res.rawResults(self, xstar, fgHb, bootstrap=self.bootstrap_results)
        r = res.bioResults(
            rawResults,
            identification_threshold=self.identification_threshold,
        )

        estimated_betas = r.getBetaValues()
        for f in self.formulas.values():
            f.set_estimated_values(estimated_betas)

        if not r.algorithm_has_converged():
            logger.warning(
                'It seems that the optimization algorithm did not converge. '
                'Therefore, the results may not correspond to the maximum '
                'likelihood estimator. Check the specification of the model, '
                'or the criteria for convergence of the algorithm.'
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

        :raises BiogemeError: if no expression has been provided for the
            likelihood

        """
        if kwargs.get("algorithm") is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if kwargs.get("algo_parameters") is not None:
            error_msg = (
                'The parameter "algo_parameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if kwargs.get("algoParameters") is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise excep.BiogemeError(error_msg)

        if self.loglike is None:
            raise excep.BiogemeError("No log likelihood function has been specified")
        if len(self.id_manager.free_betas.names) == 0:
            raise excep.BiogemeError(
                f"There is no parameter to estimate" f" in the formula: {self.loglike}."
            )

        self._set_function_parameters()
        self._set_algorithm_parameters()

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(np.array(self.id_manager.free_betas_values))
        xstar, optimizationMessages, convergence = output
        # Running time of the optimization algorithm
        optimizationMessages["Optimization time"] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimizationMessages
        self.convergence = convergence

        f = self.calculateLikelihood(xstar, scaled=False)

        fgHb = f, None, None, None
        raw_results = res.rawResults(
            self,
            xstar,
            fgHb,
            bootstrap=self.bootstrap_results,
        )
        r = res.bioResults(
            raw_results,
            identification_threshold=self.identification_threshold,
        )
        if not r.algorithm_has_converged():
            logger.warning(
                'It seems that the optimization algorithm did not converge. '
                'Therefore, the results below may not correspond to the maximum '
                'likelihood estimator. Check the specification of the model, '
                'or the criteria for convergence of the algorithm.'
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

        :raises BiogemeError: An error is raised if the database is structured
            as panel data.
        """
        if self.database.isPanel():
            raise excep.BiogemeError("Validation for panel data is not yet implemented")

        keepDatabase = self.database

        allSimulationResults = []
        count = 0
        for v in validationData:
            count += 1
            # v[0] is the estimation data set
            database = db.Database("Estimation data", v.estimation)
            self.loglike.change_init_values(estimationResults.getBetaValues())
            est_biogeme = BIOGEME(database, self.loglike)
            est_biogeme.modelName = f"{self.modelName}_val_est_{count}"
            results = est_biogeme.estimate()
            simulate = {"Loglikelihood": self.loglike}
            sim_biogeme = BIOGEME(
                db.Database("Validation data", v.validation),
                simulate,
            )
            sim_biogeme.modelName = f"{self.modelName}_val_sim_{count}"
            sim_result = sim_biogeme.simulate(results.getBetaValues())
            allSimulationResults.append(sim_result)

        self.database = keepDatabase
        if self.generate_pickle:
            fname = f'{self.modelName}_validation'
            pickleFileName = bf.get_new_file_name(fname, 'pickle')
            with open(pickleFileName, 'wb') as f:
                pickle.dump(allSimulationResults, f)
            logger.info(f"Simulation results saved in file {pickleFileName}")

        return allSimulationResults

    def optimize(self, starting_values=None):
        """Calls the optimization algorithm. The function self.algorithm
        is called.

        :param starting_values: starting point for the algorithm
        :type starting_values: list(float)

        :return: x, messages

           - x is the solution generated by the algorithm,
           - messages is a dictionary describing several information about the
             algorithm

        :rtype: numpay.array, dict(str:object)


        :raises BiogemeError: an error is raised if no algorithm is specified.
        """
        the_function = NegativeLikelihood(
            dimension=self.id_manager.number_of_free_betas,
            like=self.calculateLikelihood,
            like_deriv=self.calculateLikelihoodAndDerivatives,
            parameters=self.function_parameters,
        )

        if starting_values is None:
            starting_values = np.array(self.id_manager.free_betas_values)

        if self._algorithm is None:
            err = (
                "An algorithm must be specified. The CFSQP algorithm "
                "is not available anymore."
            )
            raise excep.BiogemeError(err)

        # logger.debug(''.join(traceback.format_stack()))
        variable_names = (
            self.free_beta_names()
            if len(starting_values) <= self.max_number_parameters_to_report
            else None
        )

        results = self._algorithm(
            fct=the_function,
            initBetas=starting_values,
            bounds=self.id_manager.bounds,
            variable_names=variable_names,
            parameters=self.algo_parameters,
        )

        return results

    def beta_values_dict_to_list(self, beta_dict=None):
        """Transforms a dict with the names of the betas associated
            with their values, into a list consistent with the
            numbering of the ids.

        :param beta_dict: dict with the values of  the parameters
        :type beta_dict: dict(str: float)

        :raises BiogemeError: if the parameter is not a dict

        :raises BiogemeError: if a parameter is missing in the dict
        """
        if beta_dict is None:
            beta_dict = {}
            for formula in self.formulas.values():
                beta_dict |= formula.get_beta_values()
        if not isinstance(beta_dict, dict):
            err = (
                "A dictionary must be provided. "
                "It can be obtained from results.getBetaValues()"
            )
            raise excep.BiogemeError(err)
        for x in beta_dict.keys():
            if x not in self.id_manager.free_betas.names:
                logger.warning(f"Parameter {x} not present in the model.")

        beta_list = []
        for x in self.id_manager.free_betas.names:
            v = beta_dict.get(x)
            if v is None:
                err = f"Incomplete dict. The value of {x} is not provided."
                raise excep.BiogemeError(err)

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

        :raises BiogemeError: if the number of parameters is incorrect

        :raises BiogemeError: if theBetaValues is None.
        """

        if theBetaValues is None:
            err = (
                "Contrarily to previous versions of Biogeme, "
                "the values of beta must "
                "now be explicitly mentioned. They can be obtained from "
                "results.getBetaValues()"
            )
            raise excep.BiogemeError(err)

        betaValues = self.beta_values_dict_to_list(theBetaValues)

        if self.database.isPanel():
            for f in self.formulas.values():
                count = f.countPanelTrajectoryExpressions()
                if count != 1:
                    theError = (
                        f"For panel data, the expression must "
                        f"contain exactly one PanelLikelihoodTrajectory "
                        f"operator. It contains {count}: {f}"
                    )
                    raise excep.BiogemeError(theError)
            output = pd.DataFrame(index=self.database.individualMap.index)
        else:
            output = pd.DataFrame(index=self.database.data.index)
        formulas_signature = [v.getSignature() for v in self.formulas.values()]

        if self.database.isPanel():
            self.database.buildPanelMap()
            self.theC.setDataMap(self.database.individualMap)

        for v in self.formulas.values():
            list_of_errors, list_of_warnings = v.audit(database=self.database)
            if list_of_warnings:
                logger.warning("\n".join(list_of_warnings))
            if list_of_errors:
                logger.warning("\n".join(list_of_errors))
                raise excep.BiogemeError("\n".join(list_of_errors))

        result = self.theC.simulateSeveralFormulas(
            formulas_signature,
            betaValues,
            self.id_manager.fixed_betas_values,
            self.database.data,
            self.numberOfThreads,
            self.database.getSampleSize(),
        )
        for key, r in zip(self.formulas.keys(), result):
            output[key] = r
        return output

    def confidenceIntervals(self, betaValues, interval_size=0.9):
        """Calculate confidence intervals on the simulated quantities


        :param betaValues: array of parameters values to be used in
               the calculations. Typically, it is a sample drawn from
               a distribution.
        :type betaValues: list(dict(str: float))

        :param interval_size: size of the reported confidence interval,
                    in percentage. If it is denoted by s, the interval
                    is calculated for the quantiles (1-s)/2 and
                    (1+s)/2. The default (0.9) corresponds to
                    quantiles for the confidence interval [0.05, 0.95].
        :type interval_size: float

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
        r = (1.0 - interval_size) / 2.0
        left = allResults.groupby(level=0).quantile(r)
        right = allResults.groupby(level=0).quantile(1.0 - r)
        return left, right

    def __str__(self):
        r = f"{self.modelName}: database [{self.database.name}]"
        r += str(self.formulas)
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
            pattern = f"*.{extension}"
            return glob.glob(pattern)
        pattern1 = f"{self.modelName}.{extension}"
        pattern2 = f"{self.modelName}~*.{extension}"
        files = glob.glob(pattern1) + glob.glob(pattern2)
        return files
