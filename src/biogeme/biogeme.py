"""
Implementation of the main Biogeme class

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

It combines the database and the model specification.
"""

from __future__ import annotations

import difflib
import glob
import logging
import multiprocessing as mp
import pickle
from datetime import datetime
from typing import NamedTuple

import cythonbiogeme.cythonbiogeme as cb
import numpy as np
import pandas as pd
from tqdm import tqdm

import biogeme.database as db
import biogeme.filenames as bf
import biogeme.optimization as opt
import biogeme.results as res
import biogeme.tools.derivatives
import biogeme.tools.unique_ids
from biogeme.configuration import Configuration
from biogeme.deprecated import (
    deprecated,
    deprecated_parameters,
)
from biogeme.dict_of_formulas import check_validity, get_expression
from biogeme.exceptions import BiogemeError, ValueOutOfRange
from biogeme.expressions import (
    IdManager,
    Expression,
    ExpressionOrNumeric,
    log,
    bioMultSum,
    SelectedExpressionsIterator,
)
from biogeme.function_output import (
    FunctionOutput,
    BiogemeFunctionOutput,
    BiogemeFunctionOutputSmartOutputProxy,
)
from biogeme.negative_likelihood import NegativeLikelihood
from biogeme.parameters import (
    Parameters,
    DEFAULT_FILE_NAME as DEFAULT_PARAMETER_FILE_NAME,
)

DEFAULT_MODEL_NAME = 'biogemeModelDefaultName'
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

    properties_initialized = False

    @deprecated_parameters(
        obsolete_params={
            'suggestScales': None,
            'numberOfThreads': 'number_of_threads',
            'numberOfDraws': 'number_of_draws',
            'missingData': 'missing_data',
            'parameter_file': 'parameters',
            'userNotes': 'user_notes',
            'generateHtml': 'generate_html',
            'saveIterations': 'save_iterations',
            'seed_param': 'seed',
        }
    )
    def __init__(
        self,
        database: db.Database,
        formulas: Expression | dict[str, Expression],
        user_notes: str | None = None,
        parameters: str | Parameters | None = None,
        skip_audit: bool = False,
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
             needed, with the keys 'log_like' and 'weight'. If only one
             formula is provided, it is associated with the label
             'log_like'. If no formula is labeled 'weight', the weight
             of each piece of data is supposed to be 1.0. In the
             simulation mode, the labels of each formula are used as
             labels of the resulting database.
        :type formulas: :class:`biogeme.expressions.Expression`, or
                        dict(:class:`biogeme.expressions.Expression`)

        :param user_notes: these notes will be included in the report file.
        :type user_notes: str

        :param parameters: name of the .toml file where the parameters are read, or Parameters object containing
        the parameters. If None, the default values of the parameters are used.


        :raise BiogemeError: an audit of the formulas is performed.
           If a formula has issues, an error is detected and an
           exception is raised.

        """
        if isinstance(parameters, Parameters):
            logger.info('Biogeme parameters provided by the user.')
            self.biogeme_parameters: Parameters = parameters
        else:
            self.biogeme_parameters: Parameters = Parameters()
            if isinstance(parameters, str):
                self.biogeme_parameters.read_file(parameters)
            else:
                if parameters is None:
                    self.biogeme_parameters.read_file(DEFAULT_PARAMETER_FILE_NAME)
                else:
                    error_msg = (
                        f'Argument "parameters" is of wrong type: {type(parameters)}'
                    )
                    raise AttributeError(error_msg)

        self.parameter_file: str = self.biogeme_parameters.file_name

        # We allow the values of the parameters to be set with arguments
        for name, value in list(kwargs.items()):
            if name in self.biogeme_parameters.parameter_names:
                self.biogeme_parameters.set_value(name=name, value=value)

        # Check for name clashes and create properties dynamically
        properties_to_define_automatically = set(
            self.biogeme_parameters.parameter_names
        ) - {'number_of_threads'}

        if not BIOGEME.properties_initialized:
            self.initialize_properties(properties_to_define_automatically)
            BIOGEME.properties_initialized = True

        self.skip_audit = skip_audit

        self.function_parameters = None

        if not self.skip_audit:
            database.data = database.data.replace({True: 1, False: 0})
            list_of_errors, list_of_warnings = database._audit()
            if list_of_warnings:
                logger.warning("\n".join(list_of_warnings))
            if list_of_errors:
                logger.warning("\n".join(list_of_errors))
                raise BiogemeError("\n".join(list_of_errors))

        self.log_like_name: str = 'log_like'
        """ Keywords used for the name of the loglikelihood formula.
        Default: 'log_like'"""

        self.log_like_valid_names: list[str] = ['log_like', 'loglike']

        self.weight_name = 'weight'
        """Keyword used for the name of the weight formula. Default: 'weight'
        """

        self.weight_valid_names = ['weight', 'weights']

        self.modelName = DEFAULT_MODEL_NAME
        """Name of the model. Default: 'biogemeModelDefaultName'
        """
        self.monte_carlo = False
        """ ``monte_carlo`` is True if one of the expressions involves a
        Monte-Carlo integration.
        """

        self.seed = self.biogeme_parameters.get_value(name='seed')
        if self.seed != 0:
            np.random.seed(self.seed)

        self.database = database

        self.short_names: biogeme.tools.unique_ids.ModelNames | None = None
        if not isinstance(formulas, dict):
            if not isinstance(formulas, Expression):
                raise BiogemeError(
                    f"Expression {formulas} is not of type "
                    f"biogeme.expressions.Expression. "
                    f"It is of type {type(formulas)}"
                )

            self.log_like: Expression = formulas
            """ Object of type :class:`biogeme.expressions.Expression`
            calculating the formula for the loglikelihood
            """

            if self.database.is_panel():
                check_variables = self.log_like.check_panel_trajectory()

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
                    raise BiogemeError(err_msg)

            self.weight = None
            """ Object of type :class:`biogeme.expressions.Expression`
            calculating the weight of each observation in the
            sample.
            """

            self.formulas = dict({self.log_like_name: formulas})
            """ Dictionary containing Biogeme formulas of type
            :class:`biogeme.expressions.Expression`.
            The keys are the names of the formulas.
            """
        else:
            self.formulas = formulas
            # Verify the validity of the formulas
            check_validity(self.formulas)
            self.log_like: Expression = get_expression(
                dict_of_formulas=formulas, valid_keywords=self.log_like_valid_names
            )
            self.weight = get_expression(
                dict_of_formulas=formulas, valid_keywords=self.weight_valid_names
            )

        self.missing_data = self.biogeme_parameters.get_value(name='missing_data')
        for f in self.formulas.values():
            f.missing_data = self.missing_data

        self.user_notes = user_notes  #: User notes

        self.lastSample = None
        """ keeps track of the sample of data used to calculate the
        stochastic gradient / hessian
        """

        self.initLogLike = None  #: Init value of the likelihood function

        self.nullLogLike = None  #: Log likelihood of the null model

        self._prepare_database_for_formula()

        if not self.skip_audit:
            self._audit()

        self.reset_id_manager()

        self.theC = cb.pyBiogeme(self.id_manager.number_of_free_betas)
        if self.database.is_panel():
            self.theC.setPanel(True)
            self.theC.setDataMap(self.database.individualMap)
        # Transfer the data to the C++ formula
        self.theC.setData(self.database.data)
        self.theC.setMissingData(self.missing_data)

        start_time = datetime.now()
        self._generate_draws(self.number_of_draws)
        if self.monte_carlo:
            self.theC.setDraws(self.database.theDraws)
        self.drawsProcessingTime = datetime.now() - start_time
        """ Time needed to generate the draws. """

        self.reset_id_manager()
        if self.log_like is not None:
            self.loglikeSignatures: list[bytes] = self.log_like.get_signature()
            """Internal signature of the formula for the
            loglikelihood."""
            if self.weight is None:
                self.theC.setExpressions(self.loglikeSignatures, self.number_of_threads)
            else:
                self.weightSignatures: list[bytes] = self.weight.get_signature()
                """ Internal signature of the formula for the weight."""
                self.theC.setExpressions(
                    self.loglikeSignatures,
                    self.number_of_threads,
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
        config_id: str,
        expression: ExpressionOrNumeric,
        database: db.Database,
        user_notes: str | None = None,
        parameters: str | Parameters | None = None,
        skip_audit: bool = False,
    ) -> BIOGEME:
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

        :param parameters: object with the parameters

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
                raise BiogemeError(error_msg)
            if config_id not in the_set:
                close_matches = difflib.get_close_matches(config_id, the_set)
                if close_matches:
                    error_msg = (
                        f"Unknown configuration: [{config_id}]. "
                        f"Did you mean [{close_matches[0]}]?"
                    )
                else:
                    error_msg = f"Unknown configuration: {config_id}."
                raise BiogemeError(error_msg)
        the_configuration = Configuration.from_string(config_id)
        expression.configure_catalogs(the_configuration)
        if user_notes is None:
            user_notes = the_configuration.get_html()
        return cls(
            database=database,
            formulas=expression,
            user_notes=user_notes,
            parameters=parameters,
            skip_audit=skip_audit,
        )

    def is_model_complex(self) -> bool:
        """Check if the model is potentially complex to estimate"""
        if self.log_like.requires_draws():
            return True
        if self.log_like.embed_expression('Integrate'):
            return True

        return False

    @staticmethod
    def argument_warning(old_new_tuple: OldNewParamTuple):
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
    def loglike(self) -> Expression:
        """For backward compatibility"""
        return self.log_like

    @classmethod
    def initialize_properties(cls, properties):
        for param in properties:
            if hasattr(cls, param):
                raise AttributeError(f"The name '{param}' already exists in the class.")
            setattr(
                cls, param, property(cls.make_getter(param), cls.make_setter(param))
            )

    @staticmethod
    def make_getter(param_name):
        def getter(self):
            return self.biogeme_parameters.get_value(name=param_name)

        return getter

    @staticmethod
    def make_setter(param_name):
        def setter(self, value):
            self.biogeme_parameters.set_value(name=param_name, value=value)
            setattr(self, f'_{param_name}', value)

        return setter

    @property
    def numberOfThreads(self) -> int:
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        Maintained for backward compatibility.
        """
        logger.warning(
            "Obsolete syntax. Use number_of_threads instead of numberOfThreads"
        )
        return self.number_of_threads

    @numberOfThreads.setter
    def numberOfThreads(self, value: int) -> None:
        logger.warning(
            "Obsolete syntax. Use number_of_threads instead of numberOfThreads"
        )
        self.number_of_threads = value

    @property
    def number_of_threads(self) -> int:
        """Number of threads used for parallel computing. Default: the number
        of available CPU.
        """
        nbr_threads = self.biogeme_parameters.get_value("number_of_threads")
        return mp.cpu_count() if nbr_threads == 0 else nbr_threads

    @number_of_threads.setter
    def number_of_threads(self, value: int) -> None:
        self.biogeme_parameters.set_value(
            name="number_of_threads", value=value, section="MultiThreading"
        )

    @property
    def numberOfDraws(self) -> int:
        """Number of draws for Monte-Carlo integration."""
        logger.warning("Obsolete syntax. Use number_of_draws instead of numberOfDraws")

        return self.number_of_draws

    @numberOfDraws.setter
    def numberOfDraws(self, value: int) -> None:
        logger.warning("Obsolete syntax. Use number_of_draws instead of numberOfDraws")
        self.number_of_draws = value

    @property
    def generatePickle(self) -> bool:
        """Boolean variable, True if the PICKLE file with the results must
        be generated.
        """
        logger.warning("Obsolete syntax. Use generate_pickle instead of generatePickle")
        return self.biogeme_parameters.get_value("generate_pickle")

    @generatePickle.setter
    def generatePickle(self, value: bool) -> None:
        logger.warning("Obsolete syntax. Use generate_pickle instead of generatePickle")
        self.biogeme_parameters.set_value(
            name="generate_pickle", value=value, section="Output"
        )

    def reset_id_manager(self) -> None:
        """Reset all the ids of the elementary expression in the formulas"""
        # First, we reset the IDs
        for f in self.formulas.values():
            f.set_id_manager(id_manager=None)
        # Second, we calculate a new set of IDs.
        self.id_manager = IdManager(
            self.formulas.values(),
            self.database,
            self.number_of_draws,
        )
        for f in self.formulas.values():
            f.set_id_manager(id_manager=self.id_manager)

    def _set_function_parameters(self) -> None:
        """Prepare the parameters for the function"""
        self.function_parameters = {
            "tolerance": self.tolerance,
            "steptol": self.steptol,
        }

    def _set_algorithm_parameters(self) -> None:
        """Prepare the parameters for the algorithms"""
        if self.optimization_algorithm == 'automatic':
            if self.is_model_complex():
                info_msg = (
                    f'As the model is rather complex, we cancel the calculation of second derivatives. If you want '
                    f'to control the parameters, change the name of the algorithm in the TOML file from '
                    f'"automatic" to "simple_bounds"'
                )
                logger.info(info_msg)
                self.algo_parameters = {
                    "proportionAnalyticalHessian": 0,
                    "infeasibleConjugateGradient": self.infeasible_cg,
                    "radius": self.initial_radius,
                    "enlargingFactor": self.enlarging_factor,
                    "maxiter": self.max_iterations,
                }
            else:
                info_msg = (
                    f'As the model is not too complex, we activate the calculation of second derivatives. If you want '
                    f'to change it, change the name of the algorithm in the TOML file from '
                    f'"automatic" to "simple_bounds"'
                )
                logger.info(info_msg)
                self.algo_parameters = {
                    "proportionAnalyticalHessian": 1,
                    "infeasibleConjugateGradient": self.infeasible_cg,
                    "radius": self.initial_radius,
                    "enlargingFactor": self.enlarging_factor,
                    "maxiter": self.max_iterations,
                }
            return

        if self.optimization_algorithm == "simple_bounds":
            self.algo_parameters = {
                "proportionAnalyticalHessian": self.second_derivatives,
                "infeasibleConjugateGradient": self.infeasible_cg,
                "radius": self.initial_radius,
                "enlargingFactor": self.enlarging_factor,
                "maxiter": self.max_iterations,
            }
            return
        if self.optimization_algorithm in [
            "simple_bounds_newton",
            "simple_bounds_BFGS",
        ]:
            self.algo_parameters = {
                "infeasibleConjugateGradient": self.infeasible_cg,
                "radius": self.initial_radius,
                "enlargingFactor": self.enlarging_factor,
                "maxiter": self.max_iterations,
            }
            return
        if self.optimization_algorithm in ["TR-newton", "TR-BFGS"]:
            self.algo_parameters = {
                "dogleg": self.dogleg,
                "radius": self.initial_radius,
                "maxiter": self.max_iterations,
            }
            return
        if self.optimization_algorithm in ["LS-newton", "LS-BFGS"]:
            self.algo_parameters = {
                "maxiter": self.max_iterations,
            }
            return
        self.algo_parameters = None

    def _save_iterations_file_name(self) -> str:
        """
        :return: The name of the file where the iterations are saved.
        :rtype: str
        """
        return f"__{self.modelName}.iter"

    def _audit(self) -> None:
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
            total = self.weight.get_value_c(
                database=self.database, aggregation=True, prepare_ids=True
            )
            s_size = self.database.get_sample_size()
            ratio = s_size / total
            if np.abs(ratio - 1) >= 0.01:
                the_warning = (
                    f"The sum of the weights ({total}) is different from "
                    f"the sample size ({self.database.get_sample_size()}). "
                    f"Multiply the weights by {ratio} to reconcile the two."
                )
                list_of_warnings.append(the_warning)
        if list_of_warnings:
            logger.warning("\n".join(list_of_warnings))
        if list_of_errors:
            logger.warning("\n".join(list_of_errors))
            raise BiogemeError("\n".join(list_of_errors))

    def _generate_draws(self, number_of_draws: int) -> None:
        """If Monte-Carlo integration is involved in one of the formulas, this
           function instructs the database to generate the draws.

        Args:
            number_of_draws: self explanatory (int)
        """

        # Draws
        self.monte_carlo = self.id_manager.requires_draws
        if self.monte_carlo:
            self.database.generate_draws(
                self.id_manager.draw_types(),
                self.id_manager.draws.names,
                number_of_draws,
            )

    def _prepare_database_for_formula(self) -> None:
        # Rebuild the map for panel data
        if self.database.is_panel():
            self.database.build_panel_map()

    @property
    def free_beta_names(self) -> list[str]:
        """Returns the names of the parameters that must be estimated

        :return: list of names of the parameters
        :rtype: list(str)
        """
        return self.id_manager.free_betas.names

    @property
    def freeBetaNames(self) -> list[str]:
        logger.warning("Obsolete syntax. Use free_beta_names instead of freeBetaNames")
        return self.id_manager.free_betas.names

    def number_unknown_parameters(self) -> int:
        """Returns the number of parameters that must be estimated

        :return: number of parameters
        :rtype: int
        """
        return len(self.id_manager.free_betas.names)

    def get_beta_values(self) -> dict[str, float]:
        """Returns a dict with the initial values of Beta. Typically
            useful for simulation.

        :return: dict with the initial values of the Beta
        :rtype: dict(str: float)
        """
        all_betas = {}
        if self.log_like is not None:
            all_betas.update(self.log_like.get_beta_values())

        if self.weight is not None:
            all_betas.update(self.weight.get_beta_values())

        for formula in self.formulas.values():
            all_betas.update(formula.get_beta_values())

        return all_betas

    def get_bounds_on_beta(self, beta_name: str) -> tuple[float, float]:
        """Returns the bounds on the parameter as defined by the user.

        :param beta_name: name of the parameter
        :type beta_name: string
        :return: lower bound, upper bound
        :rtype: tuple
        :raises BiogemeError: if the name of the parameter is not found.
        """
        index = self.id_manager.free_betas.indices.get(beta_name)
        if index is None:
            raise BiogemeError(f"Unknown parameter {beta_name}")
        return self.id_manager.bounds[index]

    @deprecated(get_bounds_on_beta)
    def getBoundsOnBeta(self, beta_name: str) -> tuple[float, float]:
        pass

    def calculate_null_loglikelihood(self, avail: dict[int, ExpressionOrNumeric]):
        """Calculate the log likelihood of the null model that predicts equal
        probability for each alternative

        :param avail: list of expressions to evaluate the availability
                      conditions for each alternative. If 1 is provided, it is always available.
        :type avail: list of :class:`biogeme.expressions.Expression`

        :return: value of the log likelihood
        :rtype: float

        """
        expression = -log(bioMultSum(avail))

        self.nullLogLike = expression.get_value_c(
            database=self.database,
            aggregation=True,
            prepare_ids=True,
        )
        return self.nullLogLike

    @deprecated(calculate_null_loglikelihood)
    def calculateNullLoglikelihood(self, avail: dict[int, ExpressionOrNumeric]):
        pass

    def calculate_init_likelihood(self) -> float:
        """Calculate the value of the log likelihood function

        The default values of the parameters are used.

        :return: value of the log likelihood.
        :rtype: float.
        """
        # Value of the loglikelihood for the default values of the parameters.
        self.initLogLike = self.calculate_likelihood(
            self.id_manager.free_betas_values, scaled=False
        )
        return self.initLogLike

    @deprecated(calculate_init_likelihood)
    def calculateInitLikelihood(self) -> float:
        pass

    def calculate_likelihood(
        self, x: np.ndarray | list[float], scaled: bool, batch: float | None = None
    ):
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

        :raises BiogemeError: if calculation with batch is requested
        """

        if batch is not None:
            raise BiogemeError("Calculation with batch not yet implemented")

        if len(x) != len(self.id_manager.free_betas_values):
            error_msg = (
                f"Input vector must be of length "
                f"{len(self.id_manager.free_betas_values)} and "
                f"not {len(x)}"
            )
            raise ValueError(error_msg)

        self._prepare_database_for_formula()
        f = self.theC.calculateLikelihood(x, self.id_manager.fixed_betas_values)

        logger.debug(
            f"Log likelihood (N = {self.database.get_sample_size()}): {f:10.7g}"
        )

        if scaled:
            return f / float(self.database.get_sample_size())

        return f

    @deprecated(calculate_likelihood)
    def calculateLikelihood(
        self, x: np.ndarray, scaled: bool, batch: float | None = None
    ):
        pass

    def report_array(self, array: np.ndarray, with_names: bool = True) -> str:
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
            names = self.free_beta_names
            report = ", ".join(
                [
                    f"{name}={value:.2g}"
                    for name, value in zip(names[:length], array[:length])
                ]
            )
            return report
        report = ", ".join([f"{value:.2g}" for value in array[:length]])
        return report

    def calculate_likelihood_and_derivatives(
        self,
        x: np.ndarray | list[float],
        scaled: bool,
        hessian: bool = False,
        bhhh: bool = False,
        batch: float | None = None,
    ) -> BiogemeFunctionOutputSmartOutputProxy:
        """Calculate the value of the log likelihood function
        and its derivatives.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :param scaled: if True, the results are divided by the number of
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
            raise BiogemeError("Calculation with batch not yet implemented")

        n = len(x)
        if n != self.id_manager.number_of_free_betas:
            error_msg = (
                f"Input vector must be of length "
                f"{self.id_manager.number_of_free_betas} and not {len(x)}"
            )
            raise ValueError(error_msg)
        self._prepare_database_for_formula()

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
            f"Log likelihood (N = {self.database.get_sample_size()}): {f:10.7g}"
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

        elif self.save_iterations:
            if self.bestIteration is None:
                self.bestIteration = f
            if f >= self.bestIteration:
                with open(
                    self._save_iterations_file_name(),
                    "w",
                    encoding="utf-8",
                ) as pf:
                    for i, v in enumerate(x):
                        print(
                            f"{self.id_manager.free_betas.names[i]} = {v}",
                            file=pf,
                        )

        if scaled:
            sample_size = float(self.database.get_sample_size())
            if sample_size == 0:
                raise BiogemeError(f"Sample size is {sample_size}")

            result = BiogemeFunctionOutput(
                function=f / sample_size,
                gradient=np.asarray(g) / sample_size,
                hessian=np.asarray(h) / sample_size,
                bhhh=np.asarray(bh) / sample_size,
            )
            return BiogemeFunctionOutputSmartOutputProxy(result)
        result = BiogemeFunctionOutput(
            function=f,
            gradient=np.asarray(g),
            hessian=np.asarray(h),
            bhhh=np.asarray(bh),
        )
        return BiogemeFunctionOutputSmartOutputProxy(result)

    @deprecated(calculate_likelihood_and_derivatives)
    def calculateLikelihoodAndDerivatives(
        self,
        x: np.ndarray,
        scaled: bool,
        hessian: bool = False,
        bhhh: bool = False,
        batch: float | None = None,
    ) -> FunctionOutput:
        pass

    def likelihood_finite_difference_hessian(
        self, x: np.ndarray | list[float]
    ) -> np.ndarray:
        """Calculate the hessian of the log likelihood function using finite
        differences.

        May be useful when the analytical hessian has numerical issues.

        :param x: vector of values for the parameters.
        :type x: list(float)

        :return: finite differences approximation of the hessian.
        :rtype: numpy.array

        :raises ValueError: if the length of the list x is incorrect

        """

        def the_function(the_x: np.ndarray) -> FunctionOutput:
            return self.calculate_likelihood_and_derivatives(
                the_x, scaled=False, hessian=False, bhhh=False
            )

        return biogeme.tools.derivatives.findiff_h(the_function, np.asarray(x))

    @deprecated(likelihood_finite_difference_hessian)
    def likelihoodFiniteDifferenceHessian(self, x: np.ndarray) -> np.ndarray:
        pass

    def check_derivatives(self, beta: np.ndarray | list[float], verbose: bool = False):
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

        def the_function(x: np.ndarray) -> FunctionOutput:
            """Wrapper function to use tools.checkDerivatives"""
            the_function_output: FunctionOutput = (
                self.calculate_likelihood_and_derivatives(
                    x, scaled=False, hessian=True, bhhh=False
                )
            )
            return the_function_output

        return biogeme.tools.derivatives.check_derivatives(
            the_function,
            np.asarray(beta),
            self.id_manager.free_betas.names,
            verbose,
        )

    @deprecated(check_derivatives)
    def checkDerivatives(self, beta: np.ndarray | list[float], verbose: bool = False):
        pass

    def _load_saved_iteration(self) -> None:
        """Reads the values of the parameters from a text file where each line
        has the form name_of_beta = value_of_beta, and use these values in all
        formulas.

        """
        filename = self._save_iterations_file_name()
        betas = {}
        try:
            with open(filename, encoding="utf-8") as fp:
                for line in fp:
                    ell = line.split("=")
                    betas[ell[0].strip()] = float(ell[1])
            self.change_init_values(betas)
            logger.info(f"Parameter values restored from {filename}")
        except OSError:
            logger.info(f"Cannot read file {filename}. Statement is ignored.")

    def set_random_init_values(self, default_bound: float = 100.0) -> None:
        """Modifies the initial values of the parameters in all formulas,
        using randomly generated values. The value is drawn from a
        uniform distribution on the interval defined by the
        bounds.

        :param default_bound: If the upper bound is missing, it is
            replaced by this value. If the lower bound is missing, it is
            replaced by the opposite of this value. Default: 100.
        :type default_bound: float
        """
        random_betas = {
            name: np.random.uniform(
                low=-default_bound if beta.lb is None else beta.lb,
                high=default_bound if beta.ub is None else beta.ub,
            )
            for name, beta in self.id_manager.free_betas.expressions.items()
        }
        self.change_init_values(random_betas)

    @deprecated(set_random_init_values)
    def setRandomInitValues(self, default_bound: float = 100.0) -> None:
        pass

    def change_init_values(self, betas: dict[str, float]) -> None:
        """Modifies the initial values of the parameters in all formula

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        if self.log_like is not None:
            self.log_like.change_init_values(betas)
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
        selected_configurations: set[Configuration] = None,
        quick_estimate: bool = False,
        recycle: bool = False,
        run_bootstrap: bool = False,
    ) -> dict[str, res.bioResults]:
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
            self.short_names = biogeme.tools.unique_ids.ModelNames(
                prefix=self.modelName
            )

        if self.log_like is None:
            raise BiogemeError("No log likelihood function has been specified")

        if selected_configurations is None:
            number_of_specifications = self.log_like.number_of_multiple_expressions()
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
                raise ValueOutOfRange(error_msg)

            the_iterator = iter(self.log_like)
        else:
            the_iterator = SelectedExpressionsIterator(
                self.log_like, selected_configurations
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
                results = b.quick_estimate(recycle=recycle)
            else:
                results = b.estimate(recycle=recycle, run_bootstrap=run_bootstrap)

            configurations[config_id] = results

        return configurations

    def recycled_estimation(
        self,
        run_bootstrap: bool = False,
        **kwargs,
    ) -> res.bioResults:

        return self.estimate(recycle=True, run_bootstrap=run_bootstrap, **kwargs)

    @deprecated_parameters(obsolete_params={'bootstrap': 'run_bootstrap'})
    def estimate(
        self,
        recycle: bool = False,
        run_bootstrap: bool = False,
        **kwargs,
    ) -> res.bioResults:
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
        if self.log_like is None:
            raise BiogemeError("No log likelihood function has been specified")

        if kwargs.get("bootstrap") is not None:
            error_msg = (
                'Parameter "bootstrap" is deprecated. In order to perform '
                "bootstrapping, set parameter run_bootstrap=True in the estimate "
                "function, and specify the number of bootstrap draws in the "
                "biogeme.toml file [e.g. bootstrap_samples=100]."
            )
            raise BiogemeError(error_msg)
        if kwargs.get("algorithm") is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if kwargs.get("algo_parameters") is not None:
            error_msg = (
                'The parameter "algo_parameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if kwargs.get("algoParameters") is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if self.modelName == DEFAULT_MODEL_NAME:
            logger.warning(
                f"You have not defined a name for the model. "
                f"The output .py are named from the model name. "
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
                        f"Several pickle .py are available for "
                        f"this model: {pickle_files}. "
                        f"The file {pickle_to_read} "
                        f"is used to load the results."
                    )
                    logger.warning(warning_msg)
                results = res.bioResults(
                    pickle_file=pickle_to_read,
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
        if len(self.id_manager.free_betas.names) == 0:
            raise BiogemeError(
                f"There is no parameter to estimate"
                f" in the formula: {self.log_like}."
            )

        if self.save_iterations:
            logger.info(
                f"*** Initial values of the parameters are "
                f"obtained from the file {self._save_iterations_file_name()}"
            )
            self._load_saved_iteration()

        self.calculate_init_likelihood()
        self.bestIteration = None

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(np.array(self.id_manager.free_betas_values))
        xstar, optimization_messages, convergence = output
        # Running time of the optimization algorithm
        optimization_messages["Optimization time"] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimization_messages
        self.convergence = convergence

        f_g_h_b: BiogemeFunctionOutput = self.calculate_likelihood_and_derivatives(
            xstar, scaled=False, hessian=True, bhhh=True
        )
        if not np.isfinite(f_g_h_b.hessian).all():
            warning_msg = (
                "Numerical problems in calculating "
                "the analytical hessian. Finite differences"
                " is tried instead."
            )
            logger.warning(warning_msg)
            fin_diff_hessian = self.likelihood_finite_difference_hessian(xstar)
            if not np.isfinite(f_g_h_b.hessian).all():
                logger.warning(
                    "Numerical problems with finite difference hessian as well."
                )
            else:
                f_g_h_b = BiogemeFunctionOutput(
                    function=f_g_h_b.function,
                    gradient=f_g_h_b.gradient,
                    hessian=fin_diff_hessian,
                    bhhh=f_g_h_b.bhhh,
                )

        # numpy array, of size B x K,
        # where
        #        - B is the number of bootstrap iterations
        #        - K is the number pf parameters to estimate
        self.bootstrap_results = None
        if run_bootstrap:
            start_time = datetime.now()

            logger.info(
                f"Re-estimate the model {self.bootstrap_samples} "
                f"times for bootstrapping"
            )
            self.bootstrap_results = np.empty(
                shape=[self.bootstrap_samples, len(xstar)]
            )
            current_logger_level = logger.level
            # Temporarily stop reporting log messages
            logger.setLevel(logging.WARNING)
            for b in tqdm(range(self.bootstrap_samples)):
                if self.database.is_panel():
                    sample = self.database.sample_individual_map_with_replacement()
                    self.theC.setDataMap(sample)
                else:
                    sample = self.database.sample_with_replacement()
                    self.theC.setData(sample)
                x_br, _, _ = self.optimize(xstar)
                self.bootstrap_results[b] = x_br

            # Time needed to generate the bootstrap results
            self.bootstrap_time = datetime.now() - start_time
            logger.setLevel(current_logger_level)
        raw_results = res.RawResults(
            self, xstar, f_g_h_b, bootstrap=self.bootstrap_results
        )
        r = res.bioResults(
            raw_results,
            identification_threshold=self.identification_threshold,
        )

        estimated_betas = r.get_beta_values()
        for f in self.formulas.values():
            f.change_init_values(estimated_betas)

        if not r.algorithm_has_converged():
            logger.warning(
                'It seems that the optimization algorithm did not converge. '
                'Therefore, the results may not correspond to the maximum '
                'likelihood estimator. Check the specification of the model, '
                'or the criteria for convergence of the algorithm.'
            )

        if self.generate_html:
            r.write_html(self.only_robust_stats)
        if self.generate_pickle:
            r.write_pickle()
        return r

    def quick_estimate(self, **kwargs) -> res.bioResults:
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
        if self.log_like is None:
            raise BiogemeError("No log likelihood function has been specified")

        if kwargs.get("algorithm") is not None:
            error_msg = (
                'The parameter "algorithm" is deprecated. Instead, define the '
                'parameter "optimization_algorithm" in section "[Estimation]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if kwargs.get("algo_parameters") is not None:
            error_msg = (
                'The parameter "algo_parameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if kwargs.get("algoParameters") is not None:
            error_msg = (
                'The parameter "algoParameters" is deprecated. Instead, define the '
                'parameters "max_iterations" and "tolerance" in section '
                '"[SimpleBounds]" '
                "of the TOML parameter file"
            )
            raise BiogemeError(error_msg)

        if len(self.id_manager.free_betas.names) == 0:
            raise BiogemeError(
                f"There is no parameter to estimate"
                f" in the formula: {self.log_like}."
            )

        self._set_function_parameters()
        self._set_algorithm_parameters()

        start_time = datetime.now()
        #        yep.start('profile.out')

        #        yep.stop()

        output = self.optimize(np.array(self.id_manager.free_betas_values))

        xstar, optimization_messages, convergence = output
        # Running time of the optimization algorithm
        optimization_messages["Optimization time"] = datetime.now() - start_time
        # Information provided by the optimization algorithm after completion.
        self.optimizationMessages = optimization_messages
        self.convergence = convergence

        f = self.calculate_likelihood(xstar, scaled=False)

        f_g_h_b = BiogemeFunctionOutput(
            function=f, gradient=None, hessian=None, bhhh=None
        )
        raw_results = res.RawResults(
            self,
            xstar,
            f_g_h_b,
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

    @deprecated(quick_estimate)
    def quickEstimate(self, **kwargs) -> res.bioResults:
        pass

    def validate(
        self,
        estimation_results: res.bioResults,
        validation_data: list[db.EstimationValidation],
    ) -> list[pd.DataFrame]:
        """Perform out-of-sample validation.

        The function performs the following tasks:

          - each slice defines a validation set (the slice itself)
            and an estimation set (the rest of the data),
          - the model is re-estimated on the estimation set,
          - the estimated model is applied on the validation set,
          - the value of the log likelihood for each observation is reported.

        :param estimation_results: results of the model estimation based on the
            full data.
        :type estimation_results: biogeme.results.bioResults

        :param validation_data: list of estimation and validation data sets
        :type validation_data: list(tuple(pandas.DataFrame, pandas.DataFrame))

        :return: a list containing as many items as slices. Each item
                 is the result of the simulation on the validation set.
        :rtype: list(pandas.DataFrame)

        :raises BiogemeError: An error is raised if the database is structured
            as panel data.
        """
        if self.database.is_panel():
            raise BiogemeError("Validation for panel data is not yet implemented")

        keep_database = self.database

        all_simulation_results = []
        count = 0
        for v in validation_data:
            count += 1
            # v[0] is the estimation data set
            database = db.Database("Estimation data", v.estimation)
            self.log_like.change_init_values(estimation_results.get_beta_values())
            est_biogeme = BIOGEME(database, self.log_like)
            est_biogeme.modelName = f"{self.modelName}_val_est_{count}"
            results = est_biogeme.estimate()
            simulate = {"Loglikelihood": self.log_like}
            sim_biogeme = BIOGEME(
                db.Database("Validation data", v.validation),
                simulate,
            )
            sim_biogeme.modelName = f"{self.modelName}_val_sim_{count}"
            sim_result = sim_biogeme.simulate(results.get_beta_values())
            all_simulation_results.append(sim_result)

        self.database = keep_database
        if self.generate_pickle:
            fname = f'{self.modelName}_validation'
            pickle_file_name = bf.get_new_file_name(fname, 'pickle')
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(all_simulation_results, f)
            logger.info(f"Simulation results saved in file {pickle_file_name}")

        return all_simulation_results

    def optimize(self, starting_values: np.ndarray = None) -> opt.OptimizationResults:
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
        if self.log_like.requires_draws() and self.number_of_draws <= 1000:
            warning_msg = f'The number of draws ({self.number_of_draws}) is low. The results may not be meaningful.'
            logger.warning(warning_msg)
        self._set_algorithm_parameters()
        the_function = NegativeLikelihood(
            dimension=self.id_manager.number_of_free_betas,
            like=self.calculate_likelihood,
            like_derivatives=self.calculate_likelihood_and_derivatives,
            parameters=self.function_parameters,
        )

        if starting_values is None:
            starting_values = np.array(self.id_manager.free_betas_values)

        algorithm_name = (
            'simple_bounds'
            if self.optimization_algorithm == 'automatic'
            else self.optimization_algorithm
        )
        the_algorithm = opt.algorithms.get(algorithm_name)
        if the_algorithm is None:
            err = f'Algorithm {self.optimization_algorithm} is not found in the optimization package'
            raise BiogemeError(err)

        # logger.debug(''.join(traceback.format_stack()))
        variable_names = (
            self.free_beta_names
            if len(starting_values) <= self.max_number_parameters_to_report
            else None
        )

        results = the_algorithm(
            fct=the_function,
            init_betas=starting_values,
            bounds=self.id_manager.bounds,
            variable_names=variable_names,
            parameters=self.algo_parameters,
        )

        return results

    def beta_values_dict_to_list(
        self, beta_dict: dict[str, float] | None = None
    ) -> list[float]:
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
            raise BiogemeError(err)
        for x in beta_dict.keys():
            if x not in self.id_manager.free_betas.names:
                logger.warning(f"Parameter {x} not present in the model.")

        beta_list = []
        for x in self.id_manager.free_betas.names:
            v = beta_dict.get(x)
            if v is None:
                err = f"Incomplete dict. The value of {x} is not provided."
                raise BiogemeError(err)

            beta_list.append(v)
        return beta_list

    @deprecated_parameters(obsolete_params={'theBetaValues': 'the_beta_values'})
    def simulate(self, the_beta_values: dict[str, float] | None) -> pd.DataFrame:
        """Applies the formulas to each row of the database.

        :param the_beta_values: values of the parameters to be used in
                the calculations. If None, the default values are
                used. Default: None.
        :type the_beta_values: dict(str, float)

        :return: a pandas data frame with the simulated value. Each
              row corresponds to a row in the database, and each
              column to a formula.

        :rtype: Pandas data frame

        Example::

              # Read the estimation results from a file
              results = res.bioResults(pickle_file = 'myModel.pickle')
              # Simulate the formulas using the nominal values
              simulatedValues = biogeme.simulate(beta_values)

        :raises BiogemeError: if the number of parameters is incorrect

        :raises BiogemeError: if theBetaValues is None.
        """
        if the_beta_values is None:
            current_beta_values = self.get_beta_values()
            err = (
                f'Contrarily to previous versions of Biogeme, '
                f'the values of Beta must '
                f'now be explicitly mentioned. If they have been estimated, they can be obtained from '
                f'results.getBetaValues(). If not, used the default values: {current_beta_values}'
            )
            raise BiogemeError(err)

        beta_values = self.beta_values_dict_to_list(the_beta_values)

        if self.database.is_panel():
            for f in self.formulas.values():
                count = f.count_panel_trajectory_expressions()
                if count != 1:
                    the_error = (
                        f"For panel data, the expression must "
                        f"contain exactly one PanelLikelihoodTrajectory "
                        f"operator. It contains {count}: {f}"
                    )
                    raise BiogemeError(the_error)
            output = pd.DataFrame(index=self.database.individualMap.index)
        else:
            output = pd.DataFrame(index=self.database.data.index)
        formulas_signature = [v.get_signature() for v in self.formulas.values()]

        if self.database.is_panel():
            self.database.build_panel_map()
            self.theC.setDataMap(self.database.individualMap)

        for v in self.formulas.values():
            list_of_errors, list_of_warnings = v.audit(database=self.database)
            if list_of_warnings:
                logger.warning("\n".join(list_of_warnings))
            if list_of_errors:
                logger.warning("\n".join(list_of_errors))
                raise BiogemeError("\n".join(list_of_errors))

        result = self.theC.simulateSeveralFormulas(
            formulas_signature,
            beta_values,
            self.id_manager.fixed_betas_values,
            self.database.data,
            self.number_of_threads,
            self.database.get_sample_size(),
        )
        for key, r in zip(self.formulas.keys(), result):
            output[key] = r
        return output

    def confidence_intervals(
        self, beta_values: list[dict[str, float]], interval_size: float = 0.9
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate confidence intervals on the simulated quantities


        :param beta_values: array of parameters values to be used in
               the calculations. Typically, it is a sample drawn from
               a distribution.
        :type beta_values: list(dict(str: float))

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
                results = res.bioResults(pickle_file = 'myModel.pickle')
                # Retrieve the names of the betas parameters that have been
                # estimated
                betas = biogeme.freeBetaNames

                # Draw 100 realization of the distribution of the estimators
                b = results.getBetasForSensitivityAnalysis(betas, size = 100)

                # Simulate the formulas using the nominal values
                simulatedValues = biogeme.simulate(beta_values)

                # Calculate the confidence intervals for each formula
                left, right = biogeme.confidenceIntervals(b, 0.9)

        :rtype: tuple of two Pandas dataframes.

        """
        list_of_results = []
        for b in beta_values:
            r = self.simulate(b)
            list_of_results += [r]
        all_results = pd.concat(list_of_results)
        r = (1.0 - interval_size) / 2.0
        left = all_results.groupby(level=0).quantile(r)
        right = all_results.groupby(level=0).quantile(1.0 - r)
        return left, right

    @deprecated(confidence_intervals)
    def confidenceIntervals(
        self, beta_values: list[dict[str, float]], interval_size: float = 0.9
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def __str__(self) -> str:
        r = f"{self.modelName}: database [{self.database.name}]"
        r += str(self.formulas)
        return r

    def files_of_type(self, extension: str, all_files: bool = False) -> list[str]:
        """Identify the list of .py with a given extension in the
        local directory

        :param extension: extension of the requested .py (without
            the dot): 'pickle', or 'html'
        :type extension: str
        :param all_files: if all_files is False, only .py containing
            the name of the model are identified. If all_files is
            True, all .py with the requested extension are
            identified.
        :type all_files: bool

        :return: list of .py with the requested extension.
        :rtype: list(str)
        """
        if all_files:
            pattern = f"*.{extension}"
            return glob.glob(pattern)
        pattern1 = f"{self.modelName}.{extension}"
        pattern2 = f"{self.modelName}~*.{extension}"
        files = glob.glob(pattern1) + glob.glob(pattern2)
        return files
