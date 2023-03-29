"""File assisted.py

:author: Michel Bierlaire
:date: Sun Mar 19 17:06:29 2023

New version of the assisted specification using Catalogs

"""
import logging
from biogeme import vns
from biogeme import biogeme
import biogeme.exceptions as excep
from biogeme.multiple_expressions import (
    string_id_to_configuration,
    configuration_to_string_id,
)
from biogeme.results import pareto_optimal
from biogeme.pareto import SetElement
from biogeme.vns import ParetoClass

logger = logging.getLogger(__name__)


class Specification:
    """Implements a specification"""

    database = None #: :class:`biogeme.database.Database` object
    pareto = None #: :class:`biogeme.paretoPareto` object
    all_results = {} #: dict(str: `biogeme.results.bioResults`)
    expression = None #: :class:`biogeme.expressions.Expression` object
    multi_objectives = None
    """
        function that generates all the objectives:
        fct(bioResults) -> list[floatNone]
    """
    

    def __init__(self, configuration_string):
        """Creates a specification from a string configuration"""
        if self.pareto is None:
            raise excep.biogemeError('Pareto set has not been initialized.')
        self.config_id = configuration_string
        self.element = self.pareto.get_element_from_id(self.config_id)
        if self.element is None:
            self.estimate()

    @classmethod
    def get_pareto_ids(cls):
        """ Returns a list with the Pareto optimal models

        :return: list with the Pareto optimal models
        :rtype: list[str]
        
        """
        return [
            element.element_id for element in cls.pareto.pareto
        ]

    @classmethod
    def from_configuration(cls, configuration):
        """ Constructor using a configuration
        """
        config_id = configuration_to_string_id(configuration)
        return cls(config_id)

    def configure_expression(self):
        """ Configure the expression to the current configuration
        """
        the_config = string_id_to_configuration(self.config_id)
        self.expression.configure_catalogs(the_config)

    @classmethod
    def default_specification(cls):
        """Alternative constructor for generate the default specification"""
        cls.expression.reset_expression_selection()
        the_config = cls.expression.current_configuration()
        the_id = configuration_to_string_id(the_config)
        return cls(the_id)

    def __repr__(self):
        return str(self.code_id())

    def get_element(self):
        """Implementation of abstract method"""
        return self.element

    def estimate(self):
        """Estimate the parameter of the current specification"""
        self.configure_expression()
        logger.debug(f'Estimate {self.code_id()}')
        b = biogeme.BIOGEME(self.database, self.expression)
        b.modelName = self.code_id()
        b.generate_html = False
        b.generate_pickle = False
        logger.info(f'*** Estimate {b.modelName}')
        results = b.quickEstimate()
        self.all_results[b.modelName] = results
        if self.multi_objectives is None:
            error_msg = 'No function has been provided to calculate the objectives to minimize'
            raise excep.biogemeError(error_msg)
        self.element = SetElement(
            self.code_id(), self.multi_objectives(results)
        )
        self.pareto.add(self.element)

    def describe(self):
        """Short description of the solution. Used for reporting.

        :return: short description of the solution.
        :rtype: str
        """
        return f'{self.get_element()}'

    def code_id(self):
        """Provide a string ID for the specification

        :return: identifier of the solution. Used to organize the Pareto set.
        :rtype: str
        """
        return self.config_id


# Operators
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
    logger.debug(f'Increase config for {catalog_name} {size=}')
    logger.debug(f'Current config: {element.element_id}')
    specification = Specification(element.element_id)
    try:
        specification.expression.increment_catalog(catalog_name, step=size)
    except excep.valueOutOfRange:
        logger.debug('Value out of range')

        return specification.get_element(), 0

    new_specification = Specification.from_configuration(
        specification.expression.current_configuration()
    )
    logger.debug(f'New config: {new_specification.get_element().element_id}')

    return new_specification.get_element(), size


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
    logger.debug(f'Decrease config for {catalog_name} {size=}')
    logger.debug(f'Current config: {element.element_id}')
    specification = Specification(element.element_id)
    try:
        specification.expression.increment_catalog(catalog_name, step=-size)
    except excep.valueOutOfRange:
        return specification.get_element(), 0

    new_specification = Specification.from_configuration(
        specification.expression.current_configuration()
    )

    return new_specification.get_element(), size


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

    def __init__(self, biogeme_object, multi_objectives, pareto_file_name):
        """Ctor

        :param biogeme_object: object containnig the loglikelihood and the database
        :type biogeme_object: biogeme.biogeme.BIOGEME

        :param multi_objectives: function calculating the objectives to minimize
        :type multi_objectives: fct(biogeme.results.bioResults) --> list[float]

        :param pareto_file_name: file where to read and write the Pareto solutions
        :type pareto_file_name: str
        """
        logger.debug('Ctor assisted specification')
        self.biogeme_object = biogeme_object
        self.multi_objectives = staticmethod(multi_objectives)
        self.pareto_file_name = pareto_file_name
        self.pareto = None
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

    def reestimate(self, model_ids):
        """The assisted specification uses quickEstimate to estimate
        the models. A complete estimation is necessary to obtain the
        full estimation results.

        """

        new_results = {}
        for config_id in model_ids:
            config = string_id_to_configuration(config_id)
            self.expression.configure_catalogs(config)
            b = biogeme.BIOGEME(self.database, self.expression)
            b.modelName = config_id
            the_result = b.estimate()
            new_results[config_id] = the_result

        return new_results

    def run(self, max_neighborhood=20, number_of_neighbors=20, reestimate=True):
        """Runs the VNS algorithm

        :return: object containing the estimation results associated
            with the  description of each configuration
        :rtype: dict(str: bioResults)

        """
        logger.debug('Run assisted specification')
        # We first try to estimate all possible configurations
        try:
            all_results = self.biogeme_object.estimate_catalog(quick_estimate=True)
        except excep.valueOutOfRange:
            logger.debug('Prepare the heuristic')
            Specification.database = self.biogeme_object.database
            Specification.expression = self.biogeme_object.loglike
            self.pareto = ParetoClass(
                max_neighborhood=max_neighborhood, pareto_file=self.pareto_file_name
            )
            Specification.pareto = self.pareto
            logger.debug('Default specification')
            default_specification = Specification.default_specification()
            logger.debug('Default specification: done')
            logger.debug('Run the heuristic')
            pareto_before = self.pareto.length_of_all_sets()
            self.pareto = vns.vns(
                problem=self,
                first_solutions=[default_specification.get_element()],
                pareto=self.pareto,
                max_neighborhood=max_neighborhood,
                number_of_neighbors=number_of_neighbors,
            )
            pareto_after = self.pareto.length_of_all_sets()
            
            self.pareto.dump(self.pareto_file_name)
            logger.info(f'Pareto file has been updated: {self.pareto_file_name}')
            logger.info(f'Before the algorithm: {pareto_before[1]} models, with {pareto_before[0]} Pareto.')
            logger.info(f'After the algorithm: {pareto_after[1]} models, with {pareto_after[0]} Pareto.')
            logger.debug('Run the heuristic: done')

            if reestimate:
                return self.reestimate(Specification.get_pareto_ids())
            # If it is not asked to restimate, the partial results are returned
            all_results = {
                element.element_id: Specification.all_results[element.element_id]
                for element in self.pareto.pareto
            }
            return all_results

        non_dominated_models = pareto_optimal(all_results)
        if reestimate:
            return self.reestimate(non_dominated_models.keys())
        return non_dominated_models

    def statistics(self):
        """Report some statistics about the process"""
        if self.pareto is None:
            return ''
        msg = (
            f'Initial Pareto: {self.pareto.size_init_pareto} '
            f'Initial considered: {self.pareto.size_init_considered} '
            f'Final Pareto: {len(self.pareto.pareto)} '
            f'Condidered: {len(self.pareto.considered)} '
            f'Removed: {len(self.pareto.removed)}'
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
