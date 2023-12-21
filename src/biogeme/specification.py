######
# TO DO: combine the validity conditions with the user defined validity
# Do something with the validity message (why it is invalid)


"""Model specification in a multiple expression context

:author: Michel Bierlaire
:date: Mon Apr 10 12:33:18 2023

Implements a model specification in a multiple expression context (using Catalogs)
"""
import logging
from typing import NamedTuple
from biogeme_optimization.pareto import SetElement
import biogeme.biogeme as bio
from biogeme import tools
from biogeme.configuration import Configuration
from biogeme.parameters import biogeme_parameters
import biogeme.exceptions as excep
from biogeme.validity import Validity

logger = logging.getLogger(__name__)


class Specification:
    """Implements a specification"""

    database = None  #: :class:`biogeme.database.Database` object
    all_results = {}  #: dict(str: `biogeme.results.bioResults`)
    expression = None  #: :class:`biogeme.expressions.Expression` object
    """
        function that generates all the objectives:
        fct(bioResults) -> list[floatNone]
    """
    user_defined_validity_check = None
    """
    function that checks the validity of the results
    """
    generic_name = 'default_name'  #: short name for file names

    def __init__(self, configuration):
        """Creates a specification from a  configuration

        :param configuration: configuration of the multiple expression
        :type configuration: biogeme.configuration.Configuration
        """
        if not isinstance(configuration, Configuration):
            error_msg = 'Ctor needs an object of type Configuration'
            raise excep.BiogemeError(error_msg)
        self.configuration = configuration
        self.model_names = None
        self.validity = None
        self._estimate()
        assert (
            self.validity is not None
        ), 'Validity must be set by the _estimate function'

    @classmethod
    def from_string_id(cls, configuration_id):
        """Constructor using a configuration"""
        return cls(Configuration.from_string(configuration_id))

    def configure_expression(self):
        """Configure the expression to the current configuration"""
        self.expression.configure_catalogs(self.configuration)

    @classmethod
    def default_specification(cls):
        """Alternative constructor for generate the default specification"""
        cls.expression.reset_expression_selection()
        the_config = cls.expression.current_configuration()
        return cls(the_config)

    @property
    def config_id(self):
        """Defined config_id as a property"""
        return self.configuration.get_string_id()

    @config_id.setter
    def config_id(self, value):
        self.configuration = Configuration.from_string(value)

    def get_results(self):
        """Obtain the estimation results of the specification"""
        the_results = self.all_results.get(self.config_id)
        if the_results is None:
            error_msg = f'No result is available for specification {self.config_id}'
            raise excep.BiogemeError(error_msg)
        return the_results

    def __repr__(self):
        return str(self.config_id)

    def _estimate(self):
        """Estimate the parameter of the current specification, if not already done

        :param quick_estimate: if True, a "quick estimate" is
            performed, in the sense that the final statistics are not
            calculated
        :type quick_estimate: bool
        """
        if self.expression is None:
            error_msg = 'No expression has been provided for the model.'
            raise excep.BiogemeError(error_msg)
        if self.database is None:
            error_msg = 'No database has been provided for the estimation.'
            raise excep.BiogemeError(error_msg)
        if self.model_names is None:
            self.model_names = tools.ModelNames(prefix=self.generic_name)

        if self.config_id in self.all_results:
            results = self.all_results.get(self.config_id)
        else:
            logger.debug(f'****** Estimate {self.config_id}')
            the_biogeme = bio.BIOGEME.from_configuration(
                config_id=self.config_id,
                expression=self.expression,
                database=self.database,
            )
            number_of_parameters = the_biogeme.number_unknown_parameters()
            maximum_number_parameters = biogeme_parameters.get_value(
                name='maximum_number_parameters', section='AssistedSpecification'
            )
            if number_of_parameters > maximum_number_parameters:
                self.validity = Validity(
                    status=False,
                    reason=(
                        f'Too many parameters: {number_of_parameters} > '
                        f'{maximum_number_parameters}'
                    ),
                )
                return
            the_biogeme.modelName = self.model_names(self.config_id)
            logger.info(f'*** Estimate {the_biogeme.modelName}')
            the_biogeme.generate_html = False
            the_biogeme.generate_pickle = False
            results = the_biogeme.quickEstimate()
        self.all_results[self.config_id] = results
        if not results.algorithm_has_converged():
            self.validity = Validity(
                status=False, reason=(f'Optimization algorithm has not converged')
            )
            return

        if self.user_defined_validity_check is not None:
            self.validity = self.user_defined_validity_check(results)
        else:
            self.validity = Validity(status=True, reason='')

    def describe(self):
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        the_results = self.get_results()
        return f'{the_results.short_summary()}'

    def get_element(self, multi_objectives):
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
