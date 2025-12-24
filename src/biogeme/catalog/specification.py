######
# TO DO: combine the validity conditions with the user defined validity
# Do something with the validity message (why it is invalid)


"""Model specification in a multiple expression context

Michel Bierlaire
Sat Jun 28 2025, 12:13:22

Implements a model specification in a multiple expression context (using Catalogs)
"""
from __future__ import annotations

import logging
from typing import Callable

from biogeme.catalog import CentralController, Configuration
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.parameters import get_default_value
from biogeme.results_processing import EstimationResults
from biogeme.tools import ModelNames
from biogeme.validity import Validity
from biogeme_optimization.pareto import SetElement

logger = logging.getLogger(__name__)


class Specification:
    """Implements a specification"""

    database: Database | None = None  #: :class:`biogeme.database.Database` object
    all_results: dict[str:EstimationResults] = {}
    central_controller: CentralController | None = None
    """
        function that generates all the objectives:
        fct(bioResults) -> list[floatNone]
    """
    user_defined_validity_check = None
    """
    function that checks the validity of the results
    """
    generic_name = 'default_name'  #: short name for file names
    maximum_number_parameters: int = get_default_value(
        name='maximum_number_parameters', section='AssistedSpecification'
    )
    model_names: ModelNames | None = None

    def __init__(
        self,
        configuration: Configuration,
    ):
        """Creates a specification from a  configuration

        :param configuration: configuration of the multiple expression
        """
        if not isinstance(configuration, Configuration):
            error_msg = 'Ctor needs an object of type Configuration'
            raise BiogemeError(error_msg)
        self.configuration = configuration
        self.validity = None
        logger.debug(f'{self.maximum_number_parameters=}')

        self._estimate()
        assert (
            self.validity is not None
        ), 'Validity must be set by the _estimate function'

    @classmethod
    def from_string_id(cls, configuration_id: str):
        """Constructor using a configuration"""
        return cls(Configuration.from_string(configuration_id))

    def configure_expression(self) -> None:
        """Configure the expression to the current configuration"""
        self.central_controller.set_configuration(self.configuration)

    @classmethod
    def default_specification(cls) -> Specification:
        """Alternative constructor for generate the default specification"""
        if cls.central_controller is None:
            raise BiogemeError('No central controller has been defined')
        cls.central_controller.reset_selection()
        the_config = cls.central_controller.get_configuration()
        return cls(the_config)

    @property
    def config_id(self) -> str:
        """Defined config_id as a property"""
        return self.configuration.get_string_id()

    @config_id.setter
    def config_id(self, value: str) -> None:
        self.configuration = Configuration.from_string(value)

    def get_results(self) -> EstimationResults:
        """Obtain the estimation results of the specification"""
        the_results = self.all_results.get(self.config_id)
        if the_results is None:
            error_msg = f'No result is available for specification {self.config_id}'
            raise BiogemeError(error_msg)
        return the_results

    def __repr__(self) -> str:
        return str(self.config_id)

    def _estimate(self) -> None:
        """Estimate the parameter of the current specification, if not already done"""
        if self.central_controller is None:
            error_msg = 'No central controller has been provided for the model.'
            raise BiogemeError(error_msg)
        if self.database is None:
            error_msg = 'No database has been provided for the estimation.'
            raise BiogemeError(error_msg)
        if __class__.model_names is None:
            __class__.model_names = ModelNames(prefix=self.generic_name)
        if self.config_id in self.all_results:
            results = self.all_results.get(self.config_id)
        else:
            logger.debug(f'****** Estimate {self.config_id}')
            from biogeme.biogeme import BIOGEME

            the_biogeme: BIOGEME = BIOGEME.from_configuration(
                config_id=self.config_id,
                multiple_expression=self.expression,
                database=self.database,
                generate_html=False,
                generate_yaml=False,
                save_iterations=False,
            )
            number_of_parameters = the_biogeme.number_unknown_parameters()
            logger.info(
                f'Model with {number_of_parameters} unknown parameters [max: {self.maximum_number_parameters}]'
            )

            if number_of_parameters > self.maximum_number_parameters:
                logger.info(
                    f'Invalid as it exceeds the maximum number of parameters: {self.maximum_number_parameters}'
                )
                self.validity = Validity(
                    status=False,
                    reason=(
                        f'Too many parameters: {number_of_parameters} > '
                        f'{self.maximum_number_parameters}'
                    ),
                )
                self.all_results[self.config_id] = None
                return

            the_biogeme.model_name = __class__.model_names(self.config_id)
            logger.info(f'*** Estimate {the_biogeme.model_name}')
            results = the_biogeme.quick_estimate()
        self.all_results[self.config_id] = results
        if not results.algorithm_has_converged:
            self.validity = Validity(
                status=False, reason=f'Optimization algorithm has not converged'
            )
            return

        if self.user_defined_validity_check is not None:
            self.validity = self.user_defined_validity_check(results)
        else:
            self.validity = Validity(status=True, reason='')

    def describe(self) -> str:
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        the_results = self.get_results()
        if the_results is None:
            return f'Invalid model: {self.validity.reason}'
        return f'{the_results.short_summary()}'

    def get_element(
        self, multi_objectives: Callable[[EstimationResults | None], list[float]]
    ) -> SetElement:
        """Obtains the element from the Pareto set corresponding to a specification

        :param multi_objectives: function calculating the objectives
            from the estimation results
        :type multi_objectives: fct(biogeme.results.bioResults) --> list[float]

        :return: element from the Pareto set
        :rtype: biogeme.pareto.SetElement

        """
        the_id = self.config_id
        the_results = self.get_results()
        the_objectives = multi_objectives(the_results)
        element = SetElement(the_id, the_objectives)
        logger.debug(f'{element=}')
        return element
