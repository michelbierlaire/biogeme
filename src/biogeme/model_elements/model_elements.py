from __future__ import annotations

import pandas as pd

from biogeme.audit_tuple import AuditTuple, merge_audit_tuples
from biogeme.constants import LOG_LIKE, WEIGHT
from biogeme.database import Database, audit_dataframe
from biogeme.draws import DrawsManagement, RandomNumberGeneratorTuple
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, audit_expression
from biogeme.expressions.base_expressions import LogitTuple
from biogeme.expressions_registry import ExpressionRegistry
from biogeme.function_output import FunctionOutput, NamedFunctionOutput
from biogeme.model_elements.audit import audit_chosen_alternative, audit_variables
from .database_adapter import ModelElementsAdapter


class ModelElements:
    """
    Container for all key components required to define and estimate a model,
    using an adapter-based design.

    :param expressions: Dict of expressions to be evaluated.
    :param use_jit: Whether to use just-in-time compilation from Jax.
    :param adapter: Adapter implementing the model elements interface.
    :param number_of_draws: Number of Monte Carlo draws.
    :param draws_management: Optional object managing the draws.
    :param user_defined_draws: dict with user defined draw generators.
    :param expressions_registry: Optional expressions registry.
    """

    loglikelihood_name: str = LOG_LIKE
    weight_name: str = WEIGHT

    def __init__(
        self,
        expressions: dict[str, Expression],
        use_jit: bool,
        adapter: ModelElementsAdapter,
        number_of_draws: int | None = None,
        draws_management: DrawsManagement | None = None,
        user_defined_draws: dict[str, RandomNumberGeneratorTuple] | None = None,
        expressions_registry: ExpressionRegistry | None = None,
    ):
        self.use_jit = use_jit
        self._adapter = adapter
        self.expressions = expressions
        self._adapter.prepare(self.expressions)
        self._database = self._adapter.database
        self._database.register_listener(self.on_database_update)
        self.number_of_draws: int | None = number_of_draws
        self._draws_management = draws_management
        self.user_defined_draws = user_defined_draws
        self._expressions_registry = expressions_registry
        # Validate inputs: only one must be provided
        if self.number_of_draws and self._draws_management:
            raise ValueError(
                "One of 'number_of_draws' or 'draws_management' must be provided (or none of them)."
            )

    @property
    def expressions_registry(self) -> ExpressionRegistry:
        if self._expressions_registry is None:
            self._expressions_registry = self._adapter.build_registry(self.expressions)
        return self._expressions_registry

    @property
    def draws_management(self) -> DrawsManagement:
        if self._draws_management is None:
            self._draws_management = DrawsManagement(
                sample_size=self.sample_size,
                number_of_draws=self.number_of_draws,
                user_generators=self.user_defined_draws,
            )
            if self.expressions_registry.requires_draws:
                self._draws_management.generate_draws(
                    draw_types=self.expressions_registry.draw_types(),
                    variable_names=self.expressions_registry.draws_names,
                )
            if self._draws_management.sample_size != self.sample_size:
                error_msg = f'Inconsistent sizes: database[{self.sample_size}] and draws [{self.draws_management.sample_size}]'
                raise BiogemeError(error_msg)
        return self._draws_management

    @property
    def free_betas_names(self) -> list[str]:
        """Returns the names of the parameters that must be estimated

        :return: list of names of the parameters
        :rtype: list(str)
        """
        return self.expressions_registry.free_betas_names

    @property
    def database(self) -> Database:
        return self._adapter.database

    @classmethod
    def from_expression_and_weight(
        cls,
        log_like: Expression,
        adapter: ModelElementsAdapter,
        use_jit: bool,
        weight: Expression | None = None,
        number_of_draws: int = 0,
        draws_management: DrawsManagement | None = None,
        user_defined_draws: dict[str, RandomNumberGeneratorTuple] | None = None,
    ) -> ModelElements:
        """
        Alternative constructor for two expressions.

        :param log_like: Expression for the log-likelihood.
        :param weight: Expression for the weight.
        :param use_jit: use just-in-time compilation from Jax
        :param adapter: Adapter implementing the model elements interface.
        :param number_of_draws: Number of Monte Carlo draws.
        :param draws_management: Optional object managing the draws.
        :param user_defined_draws: dict with user defined draw generators.
        """
        expressions = (
            {cls.loglikelihood_name: log_like}
            if weight is None
            else {cls.loglikelihood_name: log_like, cls.weight_name: weight}
        )
        return cls(
            expressions=expressions,
            adapter=adapter,
            number_of_draws=number_of_draws,
            draws_management=draws_management,
            user_defined_draws=user_defined_draws,
            use_jit=use_jit,
        )

    def on_database_update(self, updated_index: pd.Index):
        """Update the draws object to remain consistent with the new database"""
        self.draws_management.remove_rows(updated_index)

    @property
    def sample_size(self) -> int:
        return self._adapter.sample_size

    @property
    def number_of_observations(self) -> int:
        return self._adapter.number_of_observations

    @property
    def loglikelihood(self) -> Expression | None:
        return self.expressions.get(self.loglikelihood_name)

    @property
    def weight(self) -> Expression | None:
        return self.expressions.get(self.weight_name)

    @property
    def formula_names(self) -> list[str]:
        return list(self.expressions.keys())

    def generate_named_output(
        self, function_output: FunctionOutput
    ) -> NamedFunctionOutput:
        """Assigns parameter name to the entries of the gradient and the hessian"""
        return NamedFunctionOutput(
            function_output=function_output,
            mapping=self.expressions_registry.free_betas_indices,
        )

    def audit(self) -> AuditTuple:
        """Audit the model elements"""

        # First, we audit the expressions.
        expression_audits = [
            audit_expression(expr) for expr in self.expressions.values()
        ]

        # Second, we audit the database
        database_audits = [audit_dataframe(data=self.database.dataframe)]

        # Then, we check the variables

        variables_audits = [
            audit_variables(expression=expr, database=self.database)
            for expr in self.expressions.values()
        ]

        # Finally, we verify the logit formula, if any.
        logit_audits = []
        if self.loglikelihood is not None:
            logits_to_check: list[LogitTuple] = self.loglikelihood.logit_choice_avail()
            logit_audits = (
                [
                    audit_chosen_alternative(
                        choice=logit_to_check.choice,
                        availability=logit_to_check.availabilities,
                        database=self.database,
                        use_jit=self.use_jit,
                    )
                    for logit_to_check in logits_to_check
                ]
                if self.loglikelihood is not None
                else []
            )

        return merge_audit_tuples(
            expression_audits + database_audits + variables_audits + logit_audits
        )
