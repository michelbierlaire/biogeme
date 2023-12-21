"""File assisted.py

:author: Michel Bierlaire
:date: Sun Mar 19 17:06:29 2023

New version of the assisted specification using Catalogs

"""
import logging
from biogeme_optimization.vns import vns, ParetoClass
from biogeme_optimization.pareto import Pareto, SetElement, DATE_TIME_STRING
from biogeme_optimization.neighborhood import Neighborhood
from biogeme.biogeme import BIOGEME
from biogeme import tools
from biogeme.parameters import biogeme_parameters
import biogeme.version as bv
import biogeme.exceptions as excep
from biogeme.configuration import Configuration
from biogeme.specification import Specification

logger = logging.getLogger(__name__)


# Operators


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

    def reestimate(self, recycle=False):
        """The assisted specification uses quickEstimate to estimate
        the models. A complete estimation is necessary to obtain the
        full estimation results.

        """
        if self.model_names is None:
            self.model_names = tools.ModelNames(prefix=self.biogeme_object.modelName)

        all_results = {}
        for element in self.pareto.pareto:
            config_id = element.element_id
            the_biogeme = BIOGEME.from_configuration(
                config_id=config_id,
                expression=self.expression,
                database=self.database,
                parameter_file=self.biogeme_object.parameter_file,
            )
            _ = Configuration.from_string(config_id)
            the_biogeme.modelName = self.model_names(config_id)
            logger.debug(f'REESTIMATE {config_id}')
            the_result = the_biogeme.estimate(recycle=recycle)
            all_results[config_id] = the_result
        return all_results

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
        :type objective_x: int

        :param objective_y: index of the objective function to use for the y-coordinate.
        :type objective_y: int

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

        :param validity: function verifying that the estimation
            results are valid. It must return a bool and an explanation
            why if it is invalid, or None otherwise

        :type validity: fct(biogeme.results.bioResults) --> Validity

        """
        logger.debug('Ctor assisted specification')
        self.biogeme_object = biogeme_object
        self.central_controller = self.biogeme_object.loglike.set_central_controller()
        Specification.generic_name = biogeme_object.modelName
        self.multi_objectives = staticmethod(multi_objectives)
        Specification.user_defined_validity_check = (
            None if validity is None else staticmethod(validity)
        )
        largest_neighborhood = biogeme_parameters.get_value(
            name='largest_neighborhood', section='AssistedSpecification'
        )
        self.pareto = ParetoClass(
            max_neighborhood=largest_neighborhood, pareto_file=pareto_file_name
        )
        self.pareto.comments = [
            f'Biogeme {bv.getVersion()} [{bv.versionDate}]',
            f'File {self.pareto.filename} created on {DATE_TIME_STRING}',
            f'{bv.AUTHOR}, {bv.DEPARTMENT}, {bv.UNIVERSITY}',
        ]
        self.expression = biogeme_object.loglike
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise excep.BiogemeError(error_msg)
        self.database = biogeme_object.database
        Specification.expression = self.expression
        Specification.database = self.database
        self.operators = {
            name: self.generate_operator(operator)
            for name, operator in self.central_controller.prepare_operators().items()
        }
        super().__init__(self.operators)

    def generate_operator(self, function):
        """Defines an operator that takes a SetElement as an argument, to
            comply with the interface of the VNS algorithm.

        :param function: one of the function implementing the
            operators from the central controller
        :type function: function(str, int) --> str, int

        :return: operator
        :rtype: function(SetElement, int) --> SetElement, int

        """

        def the_operator(element, step):
            the_new_config, number_of_modifications = function(
                current_config=element.element_id,
                step=step,
            )
            new_specification = Specification(the_new_config)
            return (
                new_specification.get_element(self.multi_objectives),
                number_of_modifications,
            )

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

        specification = Specification.from_string_id(element.element_id)
        return specification.validity

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
        the_element = default_specification.get_element(self.multi_objectives)
        Specification.pareto.add(the_element)

        logger.info(f'{default_specification=}')
        logger.debug('Default specification: done')
        pareto_before = self.pareto.length_of_all_sets()

        # Check if we can estimate everything
        number_of_specifications = (
            self.biogeme_object.loglike.number_of_multiple_expressions()
        )
        maximum_number = self.biogeme_object.maximum_number_catalog_expressions
        if number_of_specifications <= maximum_number:
            logger.info('We consider all possible combinations of the catalogs.')
            for index, configuration in enumerate(self.biogeme_object.loglike):
                logger.info(f'Model {index}/{number_of_specifications}')
                the_config = configuration.current_configuration()
                the_specification = Specification(the_config)
                the_element = the_specification.get_element(self.multi_objectives)
                Specification.pareto.add(the_element)
                Specification.pareto.dump()
        else:
            logger.info(
                f'The number of possible specifications [{number_of_specifications}] '
                f'exceeds the maximum number [{maximum_number}]. '
                f'A heuristic algorithm is applied.'
            )

            default_element = default_specification.get_element(self.multi_objectives)
            number_of_neighbors = biogeme_parameters.get_value(
                name='number_of_neighbors', section='AssistedSpecification'
            )
            maximum_attempts = biogeme_parameters.get_value(
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
        estimation_results = post_processing.reestimate()
        post_processing.log_statistics()
        return estimation_results
