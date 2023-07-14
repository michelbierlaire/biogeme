"""File assisted.py

:author: Michel Bierlaire
:date: Sun Mar 19 17:06:29 2023

New version of the assisted specification using Catalogs

"""
import logging
import random
from datetime import date
from biogeme_optimization.vns import vns, ParetoClass
from biogeme_optimization.pareto import Pareto, SetElement, DATE_TIME_STRING
from biogeme_optimization.neighborhood import Neighborhood
from biogeme import biogeme, tools
import biogeme.version as bv
import biogeme.exceptions as excep
from biogeme.configuration import Configuration
from biogeme.specification import Specification

logger = logging.getLogger(__name__)


def get_element(specification, multi_objectives):
    """Obtains the element from the Pareto set corresponding to a specification

    :param specification: model specification
    :type specification: biogeme.specification.Specification

    :param multi_objectives: function calculating the objectives from the estimation results
    :type multi_objectives: fct(biogeme.results.bioResults) --> list[float]

    :return: element from the Pareto set
    :rtype: biogeme.pareto.SetElement
    """

    the_id = specification.config_id
    the_results = specification.get_results()
    the_objectives = multi_objectives(the_results)
    element = SetElement(the_id, the_objectives)
    logger.debug(f'{element=}')
    return element


# Operators


def modify_several_catalogs(selected_catalogs, element, step):
    """Modify several catalogs in the specification

    :param pareto: object managing the Pareto set
    :type pareto: biogeme.pareto.Pareto

    :param selected_catalogs: names of the catalogs to be modified. If
        a name does not appear in the current configuration, it is
        ignored.
    :type selected_catalogs: set(str)

    :param element: ID of the current specification
    :type element: class SetElement

    :param step: shift to apply to each catalog
    :type step: int

    :param size: number of catalogs to modify
    :type size: int

    :return: specification of the neighbor, and actual number of modifications
    :rtype: tuple(Specification, int)

    """
    specification = Specification.from_string_id(element.element_id)
    number_of_modifications = specification.expression.modify_catalogs(
        selected_catalogs, step=step, circular=True
    )
    new_specification = Specification(specification.expression.current_configuration())
    return new_specification, number_of_modifications


def modify_random_catalogs(element, step, size=1):
    """Modify several catalogs in the specification

    :param element: ID of the current specification
    :type element: class SetElement

    :param step: shift to apply to each catalog
    :type step: int

    :param size: number of catalofs to modify
    :type size: int

    :return: specification of the neighbor, and actual number of modifications
    :rtype: tuple(Specification, int)
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

    :return: new specification, and number of
        changes actually made
    :rtype: tuple(Specification, int)

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

    :return: new specification, and number of changes actually made
    :rtype: tuple(Specification, int)

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

    :return: specification of the neighbor, and actual number of modifications
    :rtype: tuple(Specification, int)
    """
    return modify_random_catalogs(element=element, step=1, size=size)


def decrement_several_catalogs(element, size=1):
    """Increment several catalogs in the specification

    :param element: ID of the current specification
    :type element: class SetElement

    :param size: number of catalofs to modify
    :type size: int

    :return: specification of the neighbor, and actual number of modifications
    :rtype: tuple(Specification, int)
    """
    return modify_random_catalogs(element=element, step=1, size=size)


class ParetoPostProcessing:
    """Class to process an existing Pareto set."""

    def __init__(
        self,
        biogeme_object,
        pareto_file_name,
    ):
        """Ctor

        :param biogeme_object: object containnig the loglikelihood and the database
        :type biogeme_object: biogeme.biogeme.BIOGEME

        :param pareto_file_name: file where to read and write the Pareto solutions
        :type pareto_file_name: str

        """
        self.biogeme_object = biogeme_object
        self.pareto = Pareto(filename=pareto_file_name)
        self.expression = biogeme_object.loglike
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise excep.BiogemeError(error_msg)
        self.database = biogeme_object.database
        self.model_names = None

    def reestimate(self, recycle):
        """The assisted specification uses quickEstimate to estimate
        the models. A complete estimation is necessary to obtain the
        full estimation results.

        """
        if self.model_names is None:
            self.model_names = tools.ModelNames(prefix=self.biogeme_object.modelName)

        new_results = {}
        for element in self.pareto.pareto:
            config_id = element.element_id
            logger.debug(f'REESTIMATE {config_id}')
            config = Configuration.from_string(config_id)
            self.expression.configure_catalogs(config)
            self.expression.locked = True
            user_notes = config.get_html()
            the_biogeme = biogeme.BIOGEME(
                self.database, self.expression, userNotes=user_notes
            )
            the_biogeme.modelName = self.model_names(config_id)
            the_result = the_biogeme.estimate(recycle=recycle)
            new_results[config_id] = the_result
            self.expression.locked = False
        return new_results

    def log_statistics(self):
        """Report some statistics about the process in the logger"""
        for msg in self.pareto.statistics():
            logger.info(msg)

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

        return self.pareto.plot(
            objective_x, objective_y, label_x, label_y, margin_x, margin_y, ax
        )


class AssistedSpecification(Neighborhood):
    """Class defining assisted specification problem for the VNS algorithm."""

    def __init__(
        self,
        biogeme_object,
        multi_objectives,
        pareto_file_name,
        validity=None,
    ):
        """Ctor

        :param biogeme_object: object containnig the loglikelihood and the database
        :type biogeme_object: biogeme.biogeme.BIOGEME

        :param multi_objectives: function calculating the objectives to minimize
        :type multi_objectives: fct(biogeme.results.bioResults) --> list[float]

        :param pareto_file_name: file where to read and write the Pareto solutions
        :type pareto_file_name: str

        :param validity: function verifying that the estimation results are valid
        :type validity: fct(biogeme.results.bioResults) --> bool

        """
        logger.debug('Ctor assisted specification')
        self.biogeme_object = biogeme_object
        self.parameters = biogeme_object.toml.parameters
        Specification.generic_name = biogeme_object.modelName
        self.multi_objectives = staticmethod(multi_objectives)
        self.validity = None if validity is None else staticmethod(validity)
        largest_neighborhood = self.parameters.get_value(
            name='largest_neighborhood', section='AssistedSpecification'
        )
        self.pareto = ParetoClass(
            max_neighborhood=largest_neighborhood, pareto_file=pareto_file_name
        )
        self.pareto.comments = [
            f'Biogeme {bv.getVersion()} [{bv.versionDate}]',
            f'File {self.pareto.filename} created on {DATE_TIME_STRING}',
            f'{bv.author}, {bv.department}, {bv.university}',
        ]
        self.expression = biogeme_object.loglike
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise excep.BiogemeError(error_msg)
        self.database = biogeme_object.database
        Specification.expression = self.expression
        Specification.database = self.database
        self.operators = {}
        all_catalogs = list(
            self.expression.dict_of_catalogs(ignore_synchronized=True).keys()
        )
        for catalog in all_catalogs:
            self.operators[f'Increase {catalog}'] = self.generate_named_operator(
                catalog, increase_configuration
            )
            self.operators[f'Decrease {catalog}'] = self.generate_named_operator(
                catalog, decrease_configuration
            )
        self.operators['Increment several catalogs'] = self.generate_operator(
            increment_several_catalogs
        )
        self.operators['Decrement several catalogs'] = self.generate_operator(
            decrement_several_catalogs
        )

        logger.debug('Ctor assisted specification: Done')
        super().__init__(self.operators)
        logger.debug('Ctor assisted specification: Done')

    def generate_operator(self, the_function):
        """Generate the operator function that complies with the
            requirements, from a function that takes the name of the catalog
            as an argument.

        :param the_function: a function that takes the catalog name, the
            current element and the size as argument, and that returns a
            neighbor specification.
        :type the_function: function(SetElement, int) --> Specification

        :return: a function that takes the current element and the size as
            argument, and that returns a neighbor. It can be used as an
            operator for the VNS algorithm.
        :rtype: function(SetElement, int)

        """

        def the_operator(element, size):
            specification, modifications = the_function(element, size)
            new_element = get_element(specification, self.multi_objectives)
            return new_element, modifications

        return the_operator

    def generate_named_operator(self, name, the_function):
        """Generate the operator function that complies with the
            requirements, from a function that takes the name of the catalog
            as an argument.

        :param name: name of the catalog
        :type name: str

        :param the_function: a function that takes the catalog name, the
            current element and the size as argument, and that returns a
            neighbor specification.
        :type the_function: function(str, SetElement, int)

        :return: a function that takes the current element and the size as
            argument, and that returns a neighbor. It can be used as an
            operator for the VNS algorithm.
        :rtype: function(SetElement, int)

        """

        def the_operator(element, size):
            specification, modifications = the_function(name, element, size)
            new_element = get_element(specification, self.multi_objectives)
            return new_element, modifications

        return the_operator

    def is_valid(self, element):
        """Check the validity of the solution.

        :param element: solution to be checked
        :type element: :class:`biogeme.pareto.SetElement`

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. why contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """
        if not isinstance(element, SetElement):
            raise excep.BiogemeError(
                f'Wrong type {type(element)} instead of SetElement'
            )

        if self.validity is None:
            return True, None

        specification = Specification.from_string_id(element.element_id)
        results = specification.get_results()
        print(f'{self.validity=}')
        print(f'{results=}')
        return self.validity(results)

    def run(self):
        """Runs the VNS algorithm

        :return: doct with the estimation results of the Pareto optimal models
        :rtype: dict[biogeme.results.bioResults]

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
        if number_of_specifications is not None:
            logger.info('We consider all possible combinations of the catalogs.')
            for c in self.biogeme_object.loglike:
                the_config = c.current_configuration()
                _ = Specification(the_config)
        else:
            logger.info(
                f'The number of possible specifications '
                f'exceeds the maximum number [{maximum_number}]. '
                f'A heuristic algorithm is applied.'
            )

            default_element = get_element(default_specification, self.multi_objectives)
            number_of_neighbors = self.parameters.get_value(
                name='number_of_neighbors', section='AssistedSpecification'
            )
            maximum_attempts = self.parameters.get_value(
                name='maximum_attempts', section='AssistedSpecification'
            )
            logger.debug(f'{default_element=}')
            self.pareto = vns(
                problem=self,
                first_solutions=[default_element],
                pareto=self.pareto,
                number_of_neighbors=number_of_neighbors,
                maximum_attempts=maximum_attempts,
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

        # Postprocessing
        logger.info(
            'VNS algorithm completed. Postprocessing of the Pareto optimal solutions'
        )

        post_processing = ParetoPostProcessing(
            biogeme_object=self.biogeme_object, pareto_file_name=self.pareto.filename
        )
        estimation_results = post_processing.reestimate(recycle=False)
        post_processing.log_statistics()
        return estimation_results
