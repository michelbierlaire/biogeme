"""File assisted.py

:author: Michel Bierlaire
:date: Sun Mar 19 17:06:29 2023

New version of the assisted specification using Catalogs

"""

import logging
from typing import Callable

import biogeme.tools.unique_ids
import biogeme.version as bv
from biogeme.biogeme import BIOGEME
from biogeme.catalog import CentralController, Configuration, ControllerOperator
from biogeme.catalog.specification import Specification
from biogeme.exceptions import BiogemeError
from biogeme.results_processing import EstimationResults
from biogeme_optimization.neighborhood import Neighborhood, Operator as VnsOperator
from biogeme_optimization.pareto import DATE_TIME_STRING, Pareto, SetElement
from biogeme_optimization.vns import ParetoClass, vns
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


# Operators


class ParetoPostProcessing:
    """Class to process an existing Pareto set."""

    def __init__(
        self,
        biogeme_object: BIOGEME,
        pareto_file_name: str,
    ):
        """Ctor

        :param biogeme_object: object containing the loglikelihood and the database
        :type biogeme_object: biogeme.biogeme.BIOGEME

        :param pareto_file_name: file where to read and write the Pareto solutions
        :type pareto_file_name: str

        """
        self.biogeme_object = biogeme_object
        self.pareto = Pareto(filename=pareto_file_name)
        self.expression = biogeme_object.log_like
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise BiogemeError(error_msg)
        self.database = biogeme_object.database
        self.model_names = None
        self.central_controller = CentralController(
            expression=biogeme_object.log_like,
            maximum_number_of_configurations=biogeme_object.biogeme_parameters.get_value(
                'maximum_number_catalog_expressions'
            ),
        )

    def reestimate(self, recycle: bool = False) -> dict[str, EstimationResults]:
        """The assisted specification uses quickEstimate to estimate
        the models. A complete estimation is necessary to obtain the
        full estimation results.

        """
        if self.model_names is None:
            self.model_names = biogeme.tools.unique_ids.ModelNames(
                prefix=self.biogeme_object.model_name
            )

        all_results = {}
        for element in self.pareto.pareto:
            config_id = element.element_id
            the_biogeme = BIOGEME.from_configuration_and_controller(
                config_id=config_id,
                central_controller=self.central_controller,
                database=self.database,
                parameters=self.biogeme_object.biogeme_parameters,
            )
            _ = Configuration.from_string(config_id)
            the_biogeme.model_name = self.model_names(config_id)
            the_result = the_biogeme.estimate(recycle=recycle)
            all_results[config_id] = the_result
        return all_results

    def log_statistics(self) -> None:
        """Report some statistics about the process in the logger"""
        for msg in self.pareto.statistics():
            logger.info(msg)

    def plot(
        self,
        objective_x: int = 0,
        objective_y: int = 1,
        label_x: str | None = None,
        label_y: str | None = None,
        margin_x: int = 5,
        margin_y: int = 5,
        ax: Axes | None = None,
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
        biogeme_object: BIOGEME,
        multi_objectives: Callable[[EstimationResults | None], list[float]],
        pareto_file_name: str,
        validity: Callable[[EstimationResults], bool] | None = None,
    ):
        """Ctor

        :param biogeme_object: object containing the loglikelihood and the database
        :param multi_objectives: function calculating the objectives to minimize
        :param pareto_file_name: file where to read and write the Pareto solutions
        :param validity: function verifying that the estimation
            results are valid. It must return a bool and an explanation
            why if it is invalid, or None otherwise

        """
        self.multi_objectives = multi_objectives
        logger.debug('Ctor assisted specification')
        self.biogeme_object = biogeme_object
        self.biogeme_parameters = biogeme_object.biogeme_parameters

        self.central_controller = CentralController(
            expression=self.biogeme_object.log_like,
            maximum_number_of_configurations=self.biogeme_parameters.get_value(
                name='maximum_number_catalog_expressions'
            ),
        )
        Specification.generic_name = biogeme_object.model_name
        Specification.user_defined_validity_check = (
            None if validity is None else staticmethod(validity)
        )
        Specification.central_controller = self.central_controller
        largest_neighborhood = self.biogeme_parameters.get_value(
            name='largest_neighborhood', section='AssistedSpecification'
        )
        self.pareto = ParetoClass(
            max_neighborhood=largest_neighborhood, pareto_file=pareto_file_name
        )
        self.pareto.comments = [
            f'Biogeme {bv.get_version()} [{bv.versionDate}]',
            f'File {self.pareto.filename} created on {DATE_TIME_STRING}',
            f'{bv.AUTHOR}, {bv.DEPARTMENT}, {bv.UNIVERSITY}',
        ]
        self.expression = biogeme_object.log_like
        if self.expression is None:
            error_msg = 'No log likelihood function is defined'
            raise BiogemeError(error_msg)
        self.database = biogeme_object.database
        Specification.maximum_number_parameters = self.biogeme_parameters.get_value(
            name='maximum_number_parameters', section='AssistedSpecification'
        )
        Specification.expression = self.expression
        Specification.database = self.database
        self.operators = {
            name: self.generate_operator(operator)
            for name, operator in self.central_controller.prepare_operators().items()
        }
        super().__init__(self.operators)

    def generate_operator(self, function: ControllerOperator) -> VnsOperator:
        """Defines an operator that takes a SetElement as an argument, to
            comply with the interface of the VNS algorithm.

        :param function: one of the function implementing the
            operators from the central controller
        :type function: function(str, int) --> str, int

        :return: operator
        :rtype: function(SetElement, int) --> SetElement, int

        """

        def the_operator(element: SetElement, step: int) -> tuple[SetElement, int]:
            the_new_configuration, number_of_modifications = function(
                Configuration.from_string(element.element_id),
                step,
            )
            new_specification = Specification(configuration=the_new_configuration)
            return (
                new_specification.get_element(self.multi_objectives),
                number_of_modifications,
            )

        return the_operator

    def is_valid(self, element: SetElement) -> tuple[bool, str]:
        """Check the validity of the solution.

        :param element: solution to be checked
        :type element: :class:`biogeme.pareto.SetElement`

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. why contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """
        if not isinstance(element, SetElement):
            raise BiogemeError(f'Wrong type {type(element)} instead of SetElement')

        specification = Specification.from_string_id(element.element_id)
        return specification.validity

    def run(self) -> dict[str, EstimationResults]:
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
        Specification.expression = self.biogeme_object.log_like
        Specification.pareto = self.pareto
        logger.debug('Default specification')
        default_specification = Specification.default_specification()
        the_element = default_specification.get_element(self.multi_objectives)
        Specification.pareto.add(the_element)

        logger.info(f'{default_specification=}')
        logger.debug('Default specification: done')
        pareto_before = self.pareto.length_of_all_sets()

        # Check if we can estimate everything
        number_of_specifications = self.central_controller.number_of_configurations()
        maximum_number = self.biogeme_object.maximum_number_catalog_expressions
        if number_of_specifications <= maximum_number:
            logger.info('We consider all possible combinations of the catalogs.')
            the_iterator = self.central_controller.expression_configuration_iterator()
            for index, the_config in enumerate(the_iterator):
                logger.info(f'Model {index}/{number_of_specifications}')
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
            number_of_neighbors = self.biogeme_parameters.get_value(
                name='number_of_neighbors', section='AssistedSpecification'
            )
            maximum_attempts = self.biogeme_parameters.get_value(
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
