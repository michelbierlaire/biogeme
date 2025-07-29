"""
Implementation of the main Biogeme class

:author: Michel Bierlaire
:date: Tue Mar 26 16:45:15 2019

It combines the database and the model specification.
"""

from __future__ import annotations

import difflib
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from biogeme.biogeme_logging import suppress_logs
from biogeme.calculator import (
    CallableExpression,
    CompiledFormulaEvaluator,
    MultiRowEvaluator,
    function_from_compiled_formula,
)
from biogeme.calculator.simple_formula import evaluate_simple_expression
from biogeme.catalog import (
    CentralController,
    Configuration,
    SelectedConfigurationsIterator,
)
from biogeme.constants import LOG_LIKE, WEIGHT
from biogeme.database import Database
from biogeme.default_parameters import ParameterValue
from biogeme.deprecated import deprecated_parameters
from biogeme.dict_of_formulas import check_validity, get_expression
from biogeme.draws import RandomNumberGeneratorTuple
from biogeme.exceptions import BiogemeError, ValueOutOfRange
from biogeme.expressions import (
    Expression,
    ExpressionOrNumeric,
    MultipleSum,
    log,
)
from biogeme.filenames import get_new_file_name
from biogeme.function_output import (
    FunctionOutput,
)
from biogeme.likelihood import AlgorithmResults, model_estimation
from biogeme.likelihood.bootstrap import bootstrap
from biogeme.model_elements import ModelElements
from biogeme.optimization import OptimizationAlgorithm, algorithms
from biogeme.parameters import (
    DEFAULT_FILE_NAME as DEFAULT_PARAMETER_FILE_NAME,
    Parameters,
)
from biogeme.results_processing import (
    RawEstimationResults,
    generate_html_file,
)
from biogeme.results_processing.estimation_results import EstimationResults
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.tools import (
    CheckDerivativesResults,
    ModelNames,
    check_derivatives,
    safe_serialize_array,
)
from biogeme.tools.files import files_of_type
from biogeme.validation import ValidationResult, cross_validate_model

DEFAULT_MODEL_NAME = 'biogeme_model_default_name'
logger = logging.getLogger(__name__)


class BIOGEME:
    """Main class that combines the database and the model
        specification.

    It works in two modes: estimation and simulation.

    The following attributes are imported from the parameter file.


    """

    # Type hints for dynamically injected attributes
    __annotations__ = {
        'seed': int,
        'tolerance': float,
        'steptol': float,
        'max_iterations': int,
        'number_of_threads': int,
        'enlarging_factor': float,
        'initial_radius': float,
        'infeasible_cg': bool,
        'second_derivatives': int,
        'dogleg': bool,
        'bootstrap_samples': int,
        'save_iterations': bool,
        'generate_html': bool,
        'generate_yaml': bool,
        'optimization_algorithm': str,
        'maximum_number_catalog_expressions': int,
        'max_number_parameters_to_report': int,
    }

    def __init__(
        self,
        database: Database,
        formulas: Expression | dict[str, Expression],
        random_number_generators: dict[str:RandomNumberGeneratorTuple] | None = None,
        user_notes: str | None = None,
        parameters: str | Parameters | None = None,
        **kwargs,
    ):
        """Constructor

        :param database: choice data.
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

        :param random_number_generators: user defined random number generators.
        :param user_notes: these notes will be included in the report file.
        :param parameters: name of the .toml file where the parameters are read, or Parameters object containing
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

        database.reset_indices()
        self.parameter_file: str = self.biogeme_parameters.file_name
        self.html_filename: str | None = None
        self.yaml_filename: str | None = None

        # We allow the values of the parameters to be set with arguments
        self.biogeme_parameters.set_several_parameters(dict_of_parameters=kwargs)

        for name in self.biogeme_parameters.parameter_names:
            self._define_property(name)

        self.maximum_number_of_observations_per_individual: int | None = None
        self.database = database

        self.log_like_name: str = LOG_LIKE
        """ Keywords used for the name of the loglikelihood formula.
        Default: 'log_like'"""

        self.log_like_valid_names: list[str] = [LOG_LIKE, 'loglike']

        self.weight_name = WEIGHT
        """Keyword used for the name of the weight formula. Default: 'weight'
        """

        self.weight_valid_names = [WEIGHT, 'weights']

        self.model_name = DEFAULT_MODEL_NAME
        """Name of the model. Default: 'biogemeModelDefaultName'
        """

        self.second_derivatives_mode = SecondDerivativesMode(
            self.calculating_second_derivatives
        )

        if self.seed != 0:
            np.random.seed(self.seed)

        self.short_names: ModelNames | None = None

        self.formulas = self._normalize_formulas(formulas)

        self.user_notes = user_notes  #: User notes

        self.init_loglikelihood = None  #: Init value of the likelihood function

        self.null_loglikelihood = None  #: Log likelihood of the null model

        self.best_iteration = None  #: Store the best iteration found so far.

        self._model_elements = ModelElements(
            expressions=self.formulas,
            database=self.database,
            number_of_draws=self.biogeme_parameters.get_value(name='number_of_draws'),
            user_defined_draws=random_number_generators,
            use_jit=self.biogeme_parameters.get_value(name='use_jit'),
        )
        self._function_evaluator = (
            CompiledFormulaEvaluator(
                model_elements=self._model_elements,
                second_derivatives_mode=self.second_derivatives_mode,
                numerically_safe=self.numerically_safe,
            )
            if self.contains_log_likelihood()
            else None
        )

    @classmethod
    def from_configuration_and_controller(
        cls,
        config_id: str,
        central_controller: CentralController,
        database: Database,
        user_notes: str | None = None,
        parameters: str | Parameters | None = None,
        **kwargs,
    ) -> BIOGEME:
        """Obtain the Biogeme object corresponding to the
        configuration of a multiple expression

        :param config_id: identifier of the configuration

        :param central_controller: central controller for the multiple expression containing all the catalogs.

        :param database: database to be passed to the Biogeme object

        :param user_notes: these notes will be included in the report file.

        :param parameters: object with the parameters

        """
        if central_controller.all_configurations:
            # We verify that the configuration is valid
            the_set = central_controller.all_configurations_ids
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
        central_controller.set_configuration(the_configuration)
        if user_notes is None:
            user_notes = the_configuration.get_html()
        flat_expression = central_controller.expression.deep_flat_copy()
        return cls(
            database=database,
            formulas=flat_expression,
            user_notes=user_notes,
            parameters=parameters,
            **kwargs,
        )

    @classmethod
    def from_configuration(
        cls,
        config_id: str,
        multiple_expression: Expression,
        database: Database,
        user_notes: str | None = None,
        parameters: str | Parameters | None = None,
        **kwargs,
    ) -> BIOGEME:
        """Obtain the Biogeme object corresponding to the
        configuration of a multiple expression
        :param config_id: identifier of the configuration

        :param multiple_expression: multiple expression containing the catalog.

        :param database: database to be passed to the Biogeme object

        :param user_notes: these notes will be included in the report file.

        :param parameters: object with the parameters

        """
        central_controller = CentralController(expression=multiple_expression)
        return cls.from_configuration_and_controller(
            config_id=config_id,
            central_controller=central_controller,
            database=database,
            user_notes=user_notes,
            parameters=parameters,
            **kwargs,
        )

    def _define_property(self, name: str):
        def getter(obj):
            return obj.biogeme_parameters.get_value(name=name)

        def setter(obj, value):
            error_msg = (
                f"Direct assignment to '{name}' is not allowed. Please set the value in the "
                f"constructor or in the .toml file."
            )
            raise BiogemeError(error_msg)

        prop = property(getter, setter)
        setattr(self.__class__, name, prop)

    @property
    def generate_pickle(self) -> bool:
        warnings.warn(
            "'generate_pickle' is deprecated. Use 'generate_yaml' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return False

    @generate_pickle.setter
    def generate_pickle(self, value: bool) -> None:
        warnings.warn(
            "'generate_pickle' is deprecated. Use 'generate_yaml' instead. This statement is ignored",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def modelName(self) -> str:
        return self.model_name

    @modelName.setter
    def modelName(self, value: str) -> None:
        warnings.warn(
            "'modelName' is deprecated. Please use 'model_name' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_name = value

    @property
    def expressions_registry(self):
        return self._model_elements.expressions_registry

    @property
    def log_like(self) -> Expression | None:
        return get_expression(
            dict_of_formulas=self.formulas, valid_keywords=self.log_like_valid_names
        )

    def contains_log_likelihood(self) -> bool:
        return self.log_like is not None

    @property
    def weight(self) -> Expression | None:
        return get_expression(
            dict_of_formulas=self.formulas, valid_keywords=self.weight_valid_names
        )

    @property
    def model_elements(self) -> ModelElements | None:
        return self._model_elements

    @property
    def function_evaluator(self) -> CompiledFormulaEvaluator:
        return self._function_evaluator

    def _normalize_formulas(
        self, formulas: Expression | dict[str, Expression]
    ) -> dict[str, Expression]:
        """
        Normalize user input to a dictionary of expressions.
        Raises BiogemeError if input is invalid.

        :param formulas: a single expression or a dictionary of named expressions.
        :return: a dictionary mapping names to expressions.
        """
        if isinstance(formulas, Expression):
            return {self.log_like_name: formulas}
        if isinstance(formulas, dict):
            check_validity(formulas)
            return formulas
        raise BiogemeError(
            f'Invalid type for formulas: {type(formulas)}. Expected Expression or dict.'
        )

    def is_model_complex(self) -> bool:
        """Check if the model is potentially complex to estimate"""
        return self.log_like.is_complex()

    @property
    def loglike(self) -> Expression:
        """For backward compatibility"""
        return self.log_like

    @property
    def function_parameters(self) -> dict[str, ParameterValue]:
        """Prepare the parameters for the function"""
        return {
            "tolerance": self.biogeme_parameters.get_value('tolerance'),
            "steptol": self.biogeme_parameters.get_value('steptol'),
        }

    @property
    def algo_parameters(self) -> dict[str, ParameterValue]:
        """Prepare the parameters for the optimization algorithm."""
        common_bounds_params = {
            'infeasibleConjugateGradient': self.biogeme_parameters.get_value(
                'infeasible_cg'
            ),
            'radius': self.biogeme_parameters.get_value('initial_radius'),
            'enlargingFactor': self.biogeme_parameters.get_value('enlarging_factor'),
            'maxiter': self.biogeme_parameters.get_value('max_iterations'),
            'max_number_parameters_to_report': self.biogeme_parameters.get_value(
                'max_number_parameters_to_report'
            ),
        }

        if self.optimization_algorithm == 'automatic':
            if (
                self.is_model_complex()
                or self.second_derivatives_mode == SecondDerivativesMode.NEVER
            ):
                logger.info(
                    'As the model is rather complex, we cancel the calculation of second derivatives. '
                    'If you want to control the parameters, change the algorithm from "automatic" '
                    'to "simple_bounds" in the TOML file.'
                )
                algo_parameters = common_bounds_params | {
                    "proportionAnalyticalHessian": 0,
                }
            else:
                logger.info(
                    'As the model is not too complex, we activate the calculation of second derivatives. '
                    'To change this behavior, modify the algorithm to "simple_bounds" in the TOML file.'
                )
                algo_parameters = common_bounds_params | {
                    "proportionAnalyticalHessian": 1,
                }
            return algo_parameters

        if self.optimization_algorithm == "simple_bounds":
            algo_parameters = common_bounds_params | {
                "proportionAnalyticalHessian": self.biogeme_parameters.get_value(
                    'second_derivatives'
                ),
            }
            if (
                self.second_derivatives_mode == SecondDerivativesMode.FINITE_DIFFERENCES
                and self.biogeme_parameters.get_value('second_derivatives') > 0
            ):
                warning = (
                    f'The proportion of the analytical hessian is not zero, and the second derivatives are approximated by '
                    f'finite difference. It may not be the desired configuration.'
                )
                logger.warning(warning)
            if (
                self.second_derivatives_mode == SecondDerivativesMode.NEVER
                and self.biogeme_parameters.get_value('second_derivatives') > 0
            ):
                error_msg = (
                    f'The proportion of the analytical hessian is not zero, and the second derivatives cannot be '
                    f'evaluated. The parameters "calculating_second_derivatives" and "second_derivatives" are inconsistent.'
                )
            return algo_parameters

        if self.optimization_algorithm in {
            "simple_bounds_newton",
            "simple_bounds_BFGS",
        }:
            return common_bounds_params

        if self.optimization_algorithm in {"TR-newton", "TR-BFGS"}:
            algo_parameters = common_bounds_params | {
                "dogleg": self.biogeme_parameters.get_value('dogleg'),
                "radius": self.biogeme_parameters.get_value('initial_radius'),
                "maxiter": self.biogeme_parameters.get_value('max_iterations'),
            }
            return algo_parameters

        if self.optimization_algorithm in {"LS-newton", "LS-BFGS"}:
            return common_bounds_params

        return common_bounds_params

    @property
    def optimization_parameters(self) -> dict[str, ParameterValue]:
        return self.function_parameters | self.algo_parameters

    def _save_iterations_file_name(self) -> str:
        """
        :return: The name of the file where the iterations are saved.
        :rtype: str
        """
        return f"__{self.model_name}.iter"

    @property
    def free_betas_names(self) -> list[str]:
        """Returns the names of the parameters that must be estimated

        :return: list of names of the parameters
        :rtype: list(str)
        """
        return self.expressions_registry.free_betas_names

    def number_unknown_parameters(self) -> int:
        """Returns the number of parameters that must be estimated

        :return: number of parameters
        :rtype: int
        """
        return self.expressions_registry.number_of_free_betas

    def calculate_null_loglikelihood(self, avail: dict[int, ExpressionOrNumeric]):
        """Calculate the log likelihood of the null model that predicts equal
        probability for each alternative

        :param avail: list of expressions to evaluate the availability
                      conditions for each alternative. If 1 is provided, it is always available.
        :type avail: list of :class:`biogeme.expressions.Expression`

        :return: value of the log likelihood
        :rtype: float

        """
        expression = -log(MultipleSum(avail))
        result = evaluate_simple_expression(
            expression=expression,
            database=self.database,
            numerically_safe=False,
            use_jit=self.use_jit,
        )
        self.null_loglikelihood = result
        return self.null_loglikelihood

    def calculate_init_likelihood(self) -> float:
        """Calculate the value of the log likelihood function

        The default values of the parameters are used.

        :return: value of the log likelihood.
        :rtype: float.
        """
        # Value of the loglikelihood for the default values of the parameters.

        result = self.function_evaluator.evaluate(
            the_betas=self.expressions_registry.free_betas_init_values,
            gradient=False,
            hessian=False,
            bhhh=False,
        )
        return result.function

    def _get_likelihood_function(
        self,
    ) -> CallableExpression:
        return function_from_compiled_formula(
            the_compiled_function=self.function_evaluator,
            the_betas=self.expressions_registry.free_betas_init_values,
        )

    def report_array(self, array: np.ndarray, with_names: bool = True) -> str:
        """Reports the entries of the array up to the maximum number

        :param array: array to report
        :type array: numpy.array

        :param with_names: if True, the names of the parameters are included
        :type with_names: bool

        :return: string reporting the values
        :rtype: str
        """
        length = min(
            array.size,
            self.biogeme_parameters.get_value('max_number_parameters_to_report'),
        )
        if with_names:
            names = self.expressions_registry.free_betas_names
            report = ", ".join(
                [
                    f"{name}={value:.2g}"
                    for name, value in zip(names[:length], array[:length])
                ]
            )
            return report
        report = ", ".join([f"{value:.2g}" for value in array[:length]])
        return report

    def _load_saved_iteration(self) -> dict[str, float]:
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
            return betas
        except OSError:
            logger.info(f"Cannot read file {filename}. Statement is ignored.")
            return {}

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
            beta.name: np.random.uniform(
                low=-default_bound if beta.lower_bound is None else beta.lower_bound,
                high=default_bound if beta.upper_bound is None else beta.upper_bound,
            )
            for beta in self.expressions_registry.free_betas
        }
        self.change_init_values(random_betas)

    def change_init_values(self, betas: dict[str, float]) -> None:
        """Modifies the initial values of the parameters in all formula

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        for expression in self.formulas.values():
            expression.change_init_values(betas)
        for beta in self.expressions_registry.free_betas:
            value = betas.get(beta.name)
            if value is not None:
                beta.init_value = value

    def _load_saved_estimates(self) -> EstimationResults:
        yaml_files = files_of_type('yaml', name=self.model_name)
        yaml_files.sort()
        if yaml_files:
            yaml_to_read = yaml_files[-1]
            if len(yaml_files) > 1:
                warning_msg = (
                    f"Several files .yaml are available for "
                    f"this model: {yaml_files}. "
                    f"The file {yaml_to_read} "
                    f"is used to load the results."
                )
                logger.warning(warning_msg)
            results = EstimationResults.from_yaml_file(filename=yaml_to_read)
            logger.warning(
                f'Estimation results read from {yaml_to_read}. '
                f'There is no guarantee that they correspond '
                f'to the specified model.'
            )
            return results
        raise BiogemeError(f'No yaml file has been found for model {self.model_name}')

    def _bootstrap(self, estimated_parameters: dict[str, float]) -> list[np.ndarray]:
        number_of_bootstrap_samples = self.biogeme_parameters.get_value(
            'bootstrap_samples'
        )
        if number_of_bootstrap_samples == 0:
            return []

        logger.info(
            f"Re-estimate the model {self.biogeme_parameters.get_value('bootstrap_samples')} "
            f"times for bootstrapping"
        )
        with suppress_logs(level=logging.WARNING):
            # For some reason, the self.optimization_parameters cannot be used as such in the function call below.
            # I have no clue why.
            opt_parameters = self.optimization_parameters
            bootstrap_results: list[AlgorithmResults] = bootstrap(
                number_of_bootstrap_samples=number_of_bootstrap_samples,
                the_algorithm=self._algorithm_configuration(),
                modeling_elements=self.model_elements,
                parameters=opt_parameters,
                starting_values=estimated_parameters,
                second_derivatives_mode=self.biogeme_parameters.get_value(
                    name='calculating_second_derivatives'
                ),
                numerically_safe=self.numerically_safe,
                number_of_jobs=self.biogeme_parameters.get_value(name='number_of_jobs'),
                use_jit=self.use_jit,
            )

        return [result.solution for result in bootstrap_results]

    def retrieve_saved_estimates(self) -> EstimationResults | None:
        """
        Attempt to retrieve previously saved estimation results from a YAML file.

        :return:
            An EstimationResults object if a saved result is found. If no file is found or loading fails,
            None is returned and a warning is logged.

        :raises BiogemeError:
            Raised internally by _load_saved_estimates if loading fails, and is caught to log a warning instead.
        """
        try:
            return self._load_saved_estimates()
        except BiogemeError as e:
            logger.warning(e)
            return None

    def estimate(
        self,
        starting_values: dict[str, float] | None = None,
        recycle: bool = False,
        run_bootstrap: bool = False,
        **kwargs,
    ) -> EstimationResults:
        """Estimate the parameters of the model(s).

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
                "bootstrapping, bootstrap_samples=100 to a positive number in the biogeme.toml file ["
                "e.g. bootstrap_samples=100]."
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

        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise BiogemeError(
                f'Ignoring unexpected arguments passed to estimate(): {unexpected}. '
                f'This method does not accept any parameters. Use biogeme.toml or the BIOGEME object to set parameters.'
            )

        if self.log_like is None:
            raise BiogemeError("No log likelihood function has been specified")

        if recycle:
            try:
                return self._load_saved_estimates()
            except BiogemeError as e:
                logger.warning(e)

        if self.model_name == DEFAULT_MODEL_NAME:
            logger.warning(
                f'You have not defined a name for the model. '
                f'The output files are named from the model name. '
                f'The default is [{DEFAULT_MODEL_NAME}]'
            )

        if self.expressions_registry.number_of_free_betas == 0:
            raise BiogemeError(
                f"There is no parameter to estimate"
                f" in the formula: {self.log_like}."
            )

        save_iteration_file_name = None
        saved_starting_values = {}
        if self.biogeme_parameters.get_value('save_iterations'):
            logger.info(
                f"*** Initial values of the parameters are "
                f"obtained from the file {self._save_iterations_file_name()}"
            )
            save_iteration_file_name = self._save_iterations_file_name()
            saved_starting_values = self._load_saved_iteration()

        if starting_values is None:
            starting_values = saved_starting_values.copy()
        else:
            for key, value in saved_starting_values.items():
                starting_values.setdefault(key, value)
        logger.info(f'Starting values for the algorithm: {starting_values}')
        init_log_likelihood = self.calculate_init_likelihood()
        logger.debug(f'Init log likelihood: {init_log_likelihood}')

        # For some reason, the self.optimization_parameters cannot be used as such in the function call below.
        # I have no clue why.
        opt_parameters = self.optimization_parameters

        algorithm_results: AlgorithmResults = model_estimation(
            the_algorithm=self._algorithm_configuration(),
            function_evaluator=self.function_evaluator,
            parameters=opt_parameters,
            some_starting_values=starting_values,
            save_iterations_filename=save_iteration_file_name,
        )

        optimal_betas = self.expressions_registry.get_named_betas_values(
            algorithm_results.solution
        )

        calculate_hessian = self.second_derivatives_mode != SecondDerivativesMode.NEVER
        f_g_h_b: FunctionOutput = self.function_evaluator.evaluate(
            the_betas=optimal_betas,
            gradient=True,
            hessian=calculate_hessian,
            bhhh=True,
        )

        if run_bootstrap:
            start_time = datetime.now()
            bootstrap_results = self._bootstrap(estimated_parameters=optimal_betas)
            # Time needed to generate the bootstrap results
            bootstrap_time = datetime.now() - start_time
        else:
            bootstrap_results = []
            bootstrap_time = None

        clean_optimization_messages = {}
        for key, value in algorithm_results.optimization_messages.items():
            if isinstance(value, np.floating):
                clean_optimization_messages[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_optimization_messages[key] = value.tolist()
            else:
                clean_optimization_messages[key] = value

        null_log_likelihood = (
            float(self.null_loglikelihood)
            if self.null_loglikelihood is not None
            else None
        )
        the_hessian = (
            None if f_g_h_b.hessian is None else safe_serialize_array(f_g_h_b.hessian)
        )
        raw_estimation_results = RawEstimationResults(
            model_name=self.model_name,
            user_notes=self.user_notes,
            beta_names=self.expressions_registry.free_betas_names,
            beta_values=algorithm_results.solution.tolist(),
            lower_bounds=[bound[0] for bound in self.expressions_registry.bounds],
            upper_bounds=[bound[1] for bound in self.expressions_registry.bounds],
            gradient=safe_serialize_array(f_g_h_b.gradient),
            hessian=the_hessian,
            bhhh=safe_serialize_array(f_g_h_b.bhhh),
            null_log_likelihood=null_log_likelihood,
            initial_log_likelihood=init_log_likelihood,
            final_log_likelihood=f_g_h_b.function,
            data_name=self.model_elements.database.name,
            sample_size=self.model_elements.sample_size,
            number_of_observations=self.model_elements.number_of_observations,
            monte_carlo=self.expressions_registry.requires_draws,
            number_of_draws=int(self.model_elements.draws_management.number_of_draws),
            types_of_draws=self.model_elements.draws_management.draw_types,
            number_of_excluded_data=self.model_elements.database.number_of_excluded_data,
            draws_processing_time=self.model_elements.draws_management.processing_time,
            optimization_messages=clean_optimization_messages,
            convergence=algorithm_results.convergence,
            bootstrap=[one_result.tolist() for one_result in bootstrap_results],
            bootstrap_time=bootstrap_time,
        )
        estimation_results = EstimationResults(
            raw_estimation_results=raw_estimation_results
        )
        estimated_betas = estimation_results.get_beta_values()
        for f in self.formulas.values():
            f.change_init_values(estimated_betas)

        if not estimation_results.algorithm_has_converged:
            logger.warning(
                'It seems that the optimization algorithm did not converge. '
                'Therefore, the results may not correspond to the maximum '
                'likelihood estimator. Check the specification of the model, '
                'or the criteria for convergence of the algorithm.'
            )
        if self.biogeme_parameters.get_value('generate_html'):
            self.html_filename = get_new_file_name(self.model_name, 'html')
            generate_html_file(
                filename=self.html_filename, estimation_results=estimation_results
            )
        if self.biogeme_parameters.get_value('generate_yaml'):
            self.yaml_filename = get_new_file_name(self.model_name, 'yaml')
            estimation_results.dump_yaml_file(filename=self.yaml_filename)
        return estimation_results

    def quick_estimate(self) -> EstimationResults:
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
        # For some reason, the self.optimization_parameters cannot be used as such in the function call below.
        # I have no clue why.
        save_iteration_file_name = None
        starting_values = {}
        if self.biogeme_parameters.get_value('save_iterations'):
            logger.info(
                f"*** Initial values of the parameters are "
                f"obtained from the file {self._save_iterations_file_name()}"
            )
            save_iteration_file_name = self._save_iterations_file_name()
            starting_values = self._load_saved_iteration()

        # For some reason, the self.optimization_parameters cannot be used as such in the function call below.
        # I have no clue why.
        opt_parameters = self.optimization_parameters

        algorithm_results: AlgorithmResults = model_estimation(
            the_algorithm=self._algorithm_configuration(),
            function_evaluator=self.function_evaluator,
            parameters=opt_parameters,
            some_starting_values=starting_values,
            save_iterations_filename=save_iteration_file_name,
        )

        optimal_betas = self.expressions_registry.get_named_betas_values(
            algorithm_results.solution
        )
        f_g_h_b: FunctionOutput = self.function_evaluator.evaluate(
            the_betas=optimal_betas,
            gradient=False,
            hessian=False,
            bhhh=False,
        )

        clean_optimization_messages = {}
        for key, value in algorithm_results.optimization_messages.items():
            if isinstance(value, np.floating):
                clean_optimization_messages[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_optimization_messages[key] = value.tolist()
            else:
                clean_optimization_messages[key] = value

        null_log_likelihood = (
            float(self.null_loglikelihood)
            if self.null_loglikelihood is not None
            else None
        )
        raw_estimation_results = RawEstimationResults(
            model_name=self.model_name,
            user_notes=self.user_notes,
            beta_names=self.expressions_registry.free_betas_names,
            beta_values=algorithm_results.solution.tolist(),
            lower_bounds=[bound[0] for bound in self.expressions_registry.bounds],
            upper_bounds=[bound[1] for bound in self.expressions_registry.bounds],
            gradient=[],
            hessian=[[]],
            bhhh=[[]],
            null_log_likelihood=null_log_likelihood,
            initial_log_likelihood=None,
            final_log_likelihood=f_g_h_b.function,
            data_name=self.model_elements.database.name,
            sample_size=self.model_elements.sample_size,
            number_of_observations=self.model_elements.number_of_observations,
            monte_carlo=self.expressions_registry.requires_draws,
            number_of_draws=int(self.model_elements.draws_management.number_of_draws),
            types_of_draws=self.model_elements.draws_management.draw_types,
            number_of_excluded_data=self.model_elements.database.number_of_excluded_data,
            draws_processing_time=self.model_elements.draws_management.processing_time,
            optimization_messages=clean_optimization_messages,
            convergence=algorithm_results.convergence,
            bootstrap=[],
            bootstrap_time=None,
        )
        return EstimationResults(raw_estimation_results=raw_estimation_results)

    def estimate_catalog(
        self,
        selected_configurations: set[Configuration] = None,
        quick_estimate: bool = False,
        recycle: bool = False,
        run_bootstrap: bool = False,
    ) -> dict[str, EstimationResults]:
        """Estimate all or selected versions of a model with Catalog's,
        corresponding to multiple specifications.

        :param selected_configurations: set of configurations. If
            None, all configurations are considered.
        :param quick_estimate: if True, the final statistics are not calculated.
        :param recycle: if True, the results are read from the pickle
            file, if it exists. If False, the estimation is performed.
        :param run_bootstrap: if True, bootstrapping is applied.
        :return: object containing the estimation results associated
            with the name of each specification, as well as a
            description of each configuration

        """
        if self.short_names is None:
            self.short_names = ModelNames(prefix=self.modelName)

        if self.log_like is None:
            raise BiogemeError("No log likelihood function has been specified")

        central_controller = CentralController(
            expression=self.log_like,
            maximum_number_of_configurations=self.biogeme_parameters.get_value(
                'maximum_number_catalog_expressions'
            ),
        )
        if selected_configurations is None:
            number_of_specifications = central_controller.number_of_configurations()
            logger.info(f'Estimating {number_of_specifications} models.')

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

            the_iterator = SelectedConfigurationsIterator(
                the_central_controller=central_controller
            )
        else:
            the_iterator = SelectedConfigurationsIterator(
                the_central_controller=central_controller,
                selected_configurations=selected_configurations,
            )
        configurations = {}
        for config in the_iterator:
            config_id = config.get_string_id()
            b = self.__class__.from_configuration(
                config_id=config_id,
                multiple_expression=self.log_like,
                database=self.database,
                user_notes=self.user_notes,
                parameters=self.biogeme_parameters,
            )
            b.model_name = self.short_names(config_id)
            results = b.retrieve_saved_estimates() if recycle else None
            if results is None:
                results = (
                    b.quick_estimate()
                    if quick_estimate
                    else b.estimate(recycle=recycle, run_bootstrap=run_bootstrap)
                )

            configurations[config_id] = results

        return configurations

    def validate(
        self,
        estimation_results: EstimationResults,
        slices: int,
        groups: str | None = None,
    ) -> list[ValidationResult]:
        """
        Perform out-of-sample validation of the model.

        The validation procedure operates by dividing the dataset into a number of slices.
        For each slice:
          - The slice is used as the validation set.
          - The remaining data forms the estimation set.
          - The model is re-estimated on the estimation set.
          - The model is applied to the validation set to compute the log likelihood.

        :param estimation_results: Estimation results obtained from the full dataset.
        :param slices: Number of data splits to create for cross-validation.
        :param groups: Optional column name used to group data entries (e.g., panel data). If provided, splitting preserves groups.

        :return: List of validation results, one for each data slice.
        :raises BiogemeError: If the dataset is structured as panel data and incompatible with validation.
        """
        # For some reason, the self.optimization_parameters cannot be used as such in the function call below.
        # I have no clue why.
        parameters = self.optimization_parameters
        parameters['calculating_second_derivatives'] = (
            self.biogeme_parameters.get_value(name='calculating_second_derivatives')
        )
        return cross_validate_model(
            the_algorithm=self._algorithm_configuration(),
            modeling_elements=self.model_elements,
            parameters=parameters,
            starting_values=estimation_results.get_beta_values(),
            slices=slices,
            numerically_safe=self.numerically_safe,
            groups=groups,
        )

    def _algorithm_configuration(self) -> OptimizationAlgorithm:
        if self.biogeme_parameters.get_value('optimization_algorithm') == "automatic":
            self.biogeme_parameters.set_value('optimization_algorithm', 'simple_bounds')
        return algorithms.get(self.optimization_algorithm)

    def simulate(self, the_beta_values: dict[str, float] | None) -> pd.DataFrame:
        """Evaluate all simulation formulas on each row of the database using the specified parameter values.

        :param the_beta_values: Dictionary mapping parameter names to values.
                                If None, an exception is raised. Use results.get_beta_values()
                                after estimation or provide explicit values.

        :return: A pandas DataFrame where each row corresponds to an observation in the database,
                 and each column corresponds to a simulation formula.

        :raises BiogemeError: If the_beta_values is None or if the number of parameters is incorrect.
        """
        if the_beta_values is None:
            current_beta_values = self.expressions_registry.free_betas
            err = (
                f'Contrarily to previous versions of Biogeme, '
                f'the values of Beta must '
                f'now be explicitly mentioned. If they have been estimated, they can be obtained from '
                f'results.get_beta_values(). If not, use the default values: {current_beta_values}'
            )
            raise BiogemeError(err)

        if not isinstance(the_beta_values, dict):
            error_msg = f'the_beta_values must be a dict, and not an object of type {type(the_beta_values)}'
            raise BiogemeError(error_msg)

        the_evaluator: MultiRowEvaluator = MultiRowEvaluator(
            model_elements=self.model_elements,
            numerically_safe=self.numerically_safe,
            use_jit=self.use_jit,
        )

        results = the_evaluator.evaluate(the_beta_values)
        return results

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
                results = EstimationEResults.from_yaml_file(filename = 'my_model.yaml')
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
        for b in tqdm(beta_values):
            r = self.simulate(b)
            list_of_results += [r]
        all_results = pd.concat(list_of_results)
        r = (1.0 - interval_size) / 2.0
        left = all_results.groupby(level=0).quantile(r)
        right = all_results.groupby(level=0).quantile(1.0 - r)
        return left, right

    def __str__(self) -> str:
        r = f"{self.model_name}: database [{self.model_elements.database.name}]"
        r += str(self.formulas)
        return r

    @deprecated_parameters({'beta': None})
    def check_derivatives(self, verbose: bool = False) -> CheckDerivativesResults:
        """Verifies the implementation of the derivatives.

        It compares the analytical version with the finite differences
        approximation.

        :param verbose: if True, the comparisons are reported. Default: False.

        :return: f, g, h, gdiff, hdiff where

            - f is the value of the function,
            - g is the analytical gradient,
            - h is the analytical hessian,
            - gdiff is the difference between the analytical and the
              finite differences gradient,
            - hdiff is the difference between the analytical and the
              finite differences hessian,

        """

        starting_values = (
            self.model_elements.expressions_registry.complete_dict_of_free_beta_values(
                the_betas={}
            )
        )

        the_log_likelihood: CallableExpression = function_from_compiled_formula(
            the_compiled_function=self.function_evaluator,
            the_betas=starting_values,
        )

        def the_function(x: np.ndarray) -> FunctionOutput:
            """Wrapper function to use tools.checkDerivatives"""
            the_function_output: FunctionOutput = the_log_likelihood(
                x, gradient=True, hessian=True, bhhh=False
            )
            return the_function_output

        return check_derivatives(
            the_function,
            np.asarray(
                self.model_elements.expressions_registry.list_of_free_betas_init_values
            ),
            self.model_elements.expressions_registry.free_betas_names,
            verbose,
        )
