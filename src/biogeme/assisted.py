"""File assisted.py

:author: Michel Bierlaire
:date: Sun Mar 19 17:06:29 2023

New version of the assisted specification using Catalogs

"""
import logging
import random
from biogeme import vns
from biogeme import biogeme
import biogeme.exceptions as excep
from biogeme.pareto import SetElement
from biogeme.elementary_expressions import TypeOfElementaryExpression
from biogeme.vns import ParetoClass
from biogeme.configuration import Configuration

logger = logging.getLogger(__name__)


class Specification:
    """Implements a specification"""

    database = None  #: :class:`biogeme.database.Database` object
    pareto = None  #: :class:`biogeme.paretoPareto` object
    all_results = {}  #: dict(str: `biogeme.results.bioResults`)
    expression = None  #: :class:`biogeme.expressions.Expression` object
    multi_objectives = None
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
        self.configure_expression()
        logger.debug(f'Estimate {self.config_id}')
        b = biogeme.BIOGEME(self.database, self.expression)
        b.modelName = self.config_id
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


# Operators


def modify_several_catalogs(selected_catalogs, element, step):
    """Modify several catalogs in the specification

    :param selected_catalogs: names of the catalogs to be modified. If
        a name does not appear in the current configuration, it is
        ignored.
    :type selected_catalogs: set(str)

    :param element: ID of the current specification
    :type element: class SetElement

    :param step: shift to apply to each catalog
    :type step: int

    :param size: number of catalofs to modify
    :type size: int

    :return: Id of the neighbor, and actual number of modifications
    :rtype: tuple(SetElement, int)

    """
    specification = Specification.from_string_id(element.element_id)
    number_of_modifications = specification.expression.modify_catalogs(
        selected_catalogs, step=step, circular=True
    )
    new_specification = Specification(specification.expression.current_configuration())
    return new_specification.get_element(), number_of_modifications


def modify_random_catalogs(element, step, size=1):
    """Modify several catalogs in the specification

    :param element: ID of the current specification
    :type element: class SetElement

    :param step: shift to apply to each catalog
    :type step: int

    :param size: number of catalofs to modify
    :type size: int

    :return: Id of the neighbor, and actual number of modifications
    :rtype: tuple(SetElement, int)
    """
    specification = Specification.from_string_id(element.element_id)
    the_catalogs = list(specification.configuration.set_of_catalogs())
    actual_size = max(size, len(the_catalogs))
    the_selected_catalogs = set(random.choices(the_catalogs, k=actual_size))
    return modify_several_catalogs(
        selected_catalogs=the_selected_catalogs, element=element, step=step
    )


def increase_configuration(catalog_name, element, size=1):
    """Change the configuration of a catalog

    :param catalog_name: name of the catalog to modify
    :type catalog_name: str

    :param element: ID of the current specification
    :type element: class SetElement

    :param size: number of increments to apply
    :type size: int

    :return: Representation of the new specification, and number of
        changes actually made
    :rtype: tuple(class SetElement, int)

    """
    return modify_several_catalogs(
        selected_catalogs={catalog_name}, element=element, step=size
    )


def decrease_configuration(catalog_name, element, size=1):
    """Change the configuration of a catalog

    :param catalog_name: name of the catalog to modify
    :type catalog_name: str

    :param element: ID of the current specification
    :type element: class SetElement

    :param size: number of increments to apply
    :type size: int

    :return: Representation of the new specification, and number of
        changes actually made
    :rtype: tuple(class SetElement, int)

    """
    return modify_several_catalogs(
        selected_catalogs={catalog_name}, element=element, step=-size
    )


def increment_several_catalogs(element, size=1):
    """Increment several catalogs in the specification

    :param element: ID of the current specification
    :type element: class SetElement

    :param size: number of catalofs to modify
    :type size: int

    :return: Id of the neighbor, and actual number of modifications
    :rtype: tuple(SetElement, int)
    """
    return modify_random_catalogs(element=element, step=1, size=size)


def decrement_several_catalogs(element, size=1):
    """Increment several catalogs in the specification

    :param element: ID of the current specification
    :type element: class SetElement

    :param size: number of catalofs to modify
    :type size: int

    :return: Id of the neighbor, and actual number of modifications
    :rtype: tuple(SetElement, int)
    """
    return modify_random_catalogs(element=element, step=1, size=size)


def generate_operator(name, the_function):
    """Generate the operator function that complies with the
        requirements, from a function that takes the name of the catalog
        as an argument.

    :param name: name of the catalog
    :type name: str

    :param the_function: a function that takes the catalog name, the
        current element and the size ar argument, and that returns a
        neighbor.
    :type the_function: function(str, SetElement, int)

    :return: a function that takes the current element and the size ar
        argument, and that returns a neighbor. It can be used as an
        operator for the VNS algorithm.
    :rtype: function(SetElement, int)

    """

    def the_operator(element, size):
        return the_function(name, element, size)

    return the_operator


class AssistedSpecification(vns.ProblemClass):
    """Class defining assisted specification problem for the VNS algorithm."""

    def __init__(
        self,
        biogeme_object,
        multi_objectives,
        pareto_file_name,
        max_neighborhood=20,
    ):
        """Ctor

        :param biogeme_object: object containnig the loglikelihood and the database
        :type biogeme_object: biogeme.biogeme.BIOGEME

        :param multi_objectives: function calculating the objectives to minimize
        :type multi_objectives: fct(biogeme.results.bioResults) --> list[float]

        :param pareto_file_name: file where to read and write the Pareto solutions
        :type pareto_file_name: str

        :param max_neighborhood: maximum number of neighborhood
            investigated by the algorithm
        :type max_neighborhood: int

        """
        logger.debug('Ctor assisted specification')
        self.biogeme_object = biogeme_object
        self.multi_objectives = staticmethod(multi_objectives)
        self.pareto = ParetoClass(
            max_neighborhood=max_neighborhood, pareto_file=pareto_file_name
        )
        self.expression = biogeme_object.loglike
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise excep.biogemeError(error_msg)
        self.database = biogeme_object.database
        Specification.expression = self.expression
        Specification.database = self.database
        Specification.multi_objectives = staticmethod(multi_objectives)
        self.operators = {}
        all_catalogs = list(
            self.expression.dict_of_catalogs(ignore_synchronized=True).keys()
        )
        for catalog in all_catalogs:
            self.operators[f'Increase {catalog}'] = generate_operator(
                catalog, increase_configuration
            )
            self.operators[f'Decrease {catalog}'] = generate_operator(
                catalog, decrease_configuration
            )
        self.operators['Increment several catalogs'] = increment_several_catalogs
        self.operators['Decrement several catalogs'] = decrement_several_catalogs

        logger.debug('Ctor assisted specification: Done')
        super().__init__(self.operators)
        logger.debug('Ctor assisted specification: Done')

    def is_valid(self, element):
        """In this implementation, we condider all models to be valid.

        :param element: representation of the specification
        :type element: class SetElement

        :return: True if the model is valid
        :rtype: bool
        """
        return True, None

    def reestimate(self):
        """The assisted specification uses quickEstimate to estimate
        the models. A complete estimation is necessary to obtain the
        full estimation results.

        """
        new_results = {}
        for element in self.pareto.pareto:
            config_id = element.element_id
            logger.debug(f'REESTIMATE {config_id}')
            config = Configuration.from_string(config_id)
            self.expression.configure_catalogs(config)
            self.expression.locked = True
            b = biogeme.BIOGEME(self.database, self.expression)
            b.modelName = config_id
            the_result = b.estimate()
            new_results[config_id] = the_result
            self.expression.locked = False
        return new_results

    def run(self, number_of_neighbors=20):
        """Runs the VNS algorithm

        :return: object containing the estimation results associated
            with the  description of each configuration
        :rtype: dict(str: bioResults)

        """
        logger.debug('Run assisted specification')
        logger.debug('Pareto solutions BEFORE')
        for elem in self.pareto.pareto:
            logger.debug(elem.element_id)
        # We first try to estimate all possible configurations
        Specification.database = self.biogeme_object.database
        Specification.expression = self.biogeme_object.loglike
        Specification.pareto = self.pareto
        logger.debug('Default specification')
        default_specification = Specification.default_specification()
        logger.debug('Default specification: done')
        pareto_before = self.pareto.length_of_all_sets()

        # Check if we can estimate everything
        number_of_specifications = (
            self.biogeme_object.loglike.number_of_multiple_expressions()
        )
        maximum_number = self.biogeme_object.maximum_number_catalog_expressions
        if number_of_specifications <= maximum_number:
            logger.info('We consider all possible combinations of the catalogs.')
            for c in self.biogeme_object.loglike:
                the_config = c.current_configuration()
                _ = Specification(the_config)
        else:
            logger.info(
                f'The number of possible specifications [{number_of_specifications}] '
                f'exceeds the maximum number [{maximum_number}]. '
                f'A heuristc algorithm is applied.'
            )

            self.pareto = vns.vns(
                problem=self,
                first_solutions=[default_specification.get_element()],
                pareto=self.pareto,
                number_of_neighbors=number_of_neighbors,
            )
        logger.debug('Pareto solutions AFTER')
        for elem in self.pareto.pareto:
            logger.debug(elem.element_id)

        pareto_after = self.pareto.length_of_all_sets()

        self.pareto.dump()
        logger.info(f'Pareto file has been updated: {self.pareto.filename}')
        logger.info(
            f'Before the algorithm: {pareto_before[1]} models, '
            f'with {pareto_before[0]} Pareto.'
        )
        logger.info(
            f'After the algorithm: {pareto_after[1]} models, '
            f'with {pareto_after[0]} Pareto.'
        )

        logger.info('Non dominated models are reestimated to obtain the statistics.')
        reestimated_models = self.reestimate()
        logger.debug('RESTIMATED MODELS')
        for key in reestimated_models:
            logger.debug(key)
        logger.info('Information about the Pareto set')
        self.log_statistics()
        return reestimated_models

    def log_statistics(self):
        """Report some statistics about the process in the logger"""
        for msg in self.statistics():
            logger.info(msg)

    def statistics(self):
        """Report some statistics about the process

        :return: tuple of messages, possibly empty.
        :rtype: tuple(str)
        """
        if self.pareto is None:
            return tuple()
        msg = (
            f'Initial Pareto: {self.pareto.size_init_pareto} ',
            f'Initial considered: {self.pareto.size_init_considered} ',
            f'Final Pareto: {len(self.pareto.pareto)} ',
            f'Condidered: {len(self.pareto.considered)} ',
            f'Removed: {len(self.pareto.removed)}',
        )
        return msg

    def plot(
        self,
        objective_x=0,
        objective_y=1,
        label_x=None,
        label_y=None,
        margin_x=5,
        margin_y=5,
        ax=None,
    ):
        """Plot the members of the set according to two
            objective functions.  They  determine the x- and
            y-coordinate of the plot.

        :param objective_x: index of the objective function to use for the x-coordinate.
        :param objective_x: int

        :param objective_y: index of the objective function to use for the y-coordinate.
        :param objective_y: int

        :param label_x: label for the x_axis
        :type label_x: str

        :param label_y: label for the y_axis
        :type label_y: str

        :param margin_x: margin for the x axis
        :type margin_x: int

        :param margin_y: margin for the y axis
        :type margin_y: int

        :param ax: matplotlib axis for the plot
        :type ax: matplotlib.Axes

        """

        self.pareto.plot(
            objective_x, objective_y, label_x, label_y, margin_x, margin_y, ax
        )
