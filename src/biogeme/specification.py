"""File specification.py

:author: Michel Bierlaire
:date: Mon Apr 10 12:33:18 2023

Implements a model specification in a multiple expression context (using Catalogs)
"""
import logging
import biogeme.biogeme as bio
from biogeme import tools
from biogeme.configuration import Configuration
import biogeme.exceptions as excep
from biogeme.pareto import SetElement

logger = logging.getLogger(__name__)


class Specification:
    """Implements a specification"""

    database = None  #: :class:`biogeme.database.Database` object
    pareto = None  #: :class:`biogeme.paretoPareto` object
    all_results = {}  #: dict(str: `biogeme.results.bioResults`)
    expression = None  #: :class:`biogeme.expressions.Expression` object
    multi_objectives = None
    generic_name = 'default_name'  #: short name for file names
    """
        function that generates all the objectives:
        fct(bioResults) -> list[floatNone]
    """

    def __init__(self, configuration):
        """Creates a specification from a  configuration

        :param configuration: configuration of the multiple expression
        :type configuration: biogeme.configuration.Configuration
        """
        if not isinstance(configuration, Configuration):
            error_msg = 'Ctor needs an object of type Configuration'
            raise excep.biogemeError(error_msg)
        if self.pareto is None:
            raise excep.biogemeError('Pareto set has not been initialized.')
        self.configuration = configuration
        self.element = self.pareto.get_element_from_id(self.config_id)
        if self.element is None:
            self.estimate()
        self.model_names = None

    @classmethod
    def from_string_id(cls, configuration_id):
        """Constructor using a configuration"""
        return cls(Configuration.from_string(configuration_id))

    @classmethod
    def get_pareto_ids(cls):
        """Returns a list with the Pareto optimal models

        :return: list with the Pareto optimal models
        :rtype: list[str]

        """
        return [element.element_id for element in cls.pareto.pareto]

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

    def __repr__(self):
        return str(self.config_id)

    def get_element(self):
        """Implementation of abstract method"""
        return self.element

    def estimate(self):
        """Estimate the parameter of the current specification"""
        if self.model_names is None:
            self.model_names = tools.ModelNames(prefix=self.generic_name)

        the_config = self.expression.current_configuration()
        self.configure_expression()
        logger.debug(f'Estimate {self.config_id}')
        userNotes = the_config.get_html()
        b = bio.BIOGEME(self.database, self.expression, userNotes=userNotes)
        b.modelName = self.model_names(self.config_id)
        b.generate_html = False
        b.generate_pickle = False
        logger.info(f'*** Estimate {b.modelName}')
        results = b.quickEstimate()
        self.all_results[b.modelName] = results
        if self.multi_objectives is None:
            error_msg = (
                'No function has been provided to calculate the objectives to minimize'
            )
            raise excep.biogemeError(error_msg)
        self.element = SetElement(self.config_id, self.multi_objectives(results))
        self.pareto.add(self.element)

    def describe(self):
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        return f'{self.get_element()}'
