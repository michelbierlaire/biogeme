from __future__ import annotations

import pandas as pd

from biogeme.audit_tuple import AuditTuple, display_messages, merge_audit_tuples
from biogeme.constants import LOG_LIKE, WEIGHT
from biogeme.database import Database, PanelDatabase, audit_dataframe
from biogeme.default_parameters import MISSING_VALUE
from biogeme.draws import DrawsManagement, RandomNumberGeneratorTuple
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, audit_expression
from biogeme.expressions.base_expressions import LogitTuple
from biogeme.expressions_registry import ExpressionRegistry
from biogeme.function_output import FunctionOutput, NamedFunctionOutput
from biogeme.model_elements.audit import audit_chosen_alternative, audit_variables


class ModelElements:
    """
    Container for all key components required to define and estimate a model.

    :param expressions: Dict of expressions to be evaluated.
    """

    def __init__(
        self,
        expressions: dict[str, Expression],
        use_jit: bool,
        database: Database | None,
        number_of_draws: int | None = None,
        draws_management: DrawsManagement | None = None,
        user_defined_draws: dict[str:RandomNumberGeneratorTuple] | None = None,
        expressions_registry: ExpressionRegistry = None,
    ):
        self.use_jit = use_jit
        if database is None:
            database = Database.dummy_database()
        self.panel_prepared: bool = False
        if database.panel_column is None:
            self._is_panel = False
            self._panel_database: PanelDatabase | None = None
            self._database: Database = database
        else:
            self._is_panel: bool = True
            self._panel_database = PanelDatabase(
                database=database, panel_column=database.panel_column
            )
            self._database: Database = database

        self.expressions: dict[str, Expression] = expressions
        self._flat_database: Database | None = None
        self.number_of_draws: int = number_of_draws
        self.draws_management = draws_management
        self.user_defined_draws = user_defined_draws
        self.expressions_registry = expressions_registry
        # Validate inputs: only one must be provided
        if self.number_of_draws and self.draws_management:
            raise ValueError(
                "One of 'number_of_draws' or 'draws_management' must be provided (or none of them)."
            )

        if self.expressions_registry is None:
            self.expressions_registry = ExpressionRegistry(
                self.expressions.values(), self.database
            )

        if self._database is not None:
            self.database.register_listener(self.on_database_update)

            if self.draws_management is None:
                self.draws_management = DrawsManagement(
                    sample_size=self.sample_size,
                    number_of_draws=self.number_of_draws,
                    user_generators=user_defined_draws,
                )
                if self.expressions_registry.requires_draws:
                    self.draws_management.generate_draws(
                        draw_types=self.expressions_registry.draw_types(),
                        variable_names=self.expressions_registry.draws_names,
                    )
            else:
                # Check consistency of sized
                if self.draws_management.sample_size != self.sample_size:
                    error_msg = f'Inconsistent sizes: database[{self.sample_size}] and draws [{self.draws_management.sample_size}]'
                    raise BiogemeError(error_msg)

        if self._is_panel:
            self._prepare_for_panel()
            self.panel_prepared: bool = True
        display_messages(self.audit())

    def is_panel(self) -> bool:
        return self._is_panel

    @property
    def database(self) -> Database:
        if not self.is_panel():
            return self._database
        if not self.panel_prepared:
            self._prepare_for_panel()
        return self._flat_database

    @classmethod
    def from_expression_and_weight(
        cls,
        log_like: Expression,
        database: Database,
        use_jit: bool,
        weight: Expression | None = None,
        number_of_draws: int = 0,
        draws_management: DrawsManagement | None = None,
        user_defined_draws: dict[str:RandomNumberGeneratorTuple] | None = None,
    ) -> ModelElements:
        """
        Alternative constructor for two expressions.

        :param log_like: Expression for the log-likelihood.
        :param weight: Expression for the weight.
        :param database: The database containing data.
        :param number_of_draws: Number of Monte Carlo draws.
        :param draws_management: Optional object managing the draws.
        """
        expressions = (
            {LOG_LIKE: log_like}
            if weight is None
            else {LOG_LIKE: log_like, WEIGHT: weight}
        )
        return cls(
            expressions=expressions,
            database=database,
            number_of_draws=number_of_draws,
            draws_management=draws_management,
            user_defined_draws=user_defined_draws,
            use_jit=use_jit,
        )

    def on_database_update(self, updated_index: pd.Index):
        """Update the draws object to remain consistent with the new database"""
        self.draws_management.remove_rows(updated_index)
        self.panel_prepared = False

    def _prepare_for_panel(self) -> None:

        flat_dataframe, maximum_number_of_observations_per_individual = (
            self._panel_database.flatten_database(missing_data=MISSING_VALUE)
        )
        self._flat_database = Database(
            name=f'flat {self._database.name}', dataframe=flat_dataframe
        )
        for expression in self.expressions.values():
            expression.set_maximum_number_of_observations_per_individual(
                max_number=maximum_number_of_observations_per_individual
            )
        self.expressions_registry = ExpressionRegistry(
            self.expressions.values(), self._flat_database
        )

    @property
    def sample_size(self) -> int:
        if self._database is None:
            return 0
        return self.database.num_rows()

    @property
    def number_of_observations(self) -> int:
        if self._database is None:
            return 0
        return self._database.num_rows()

    @property
    def loglikelihood(self) -> Expression | None:
        return self.expressions.get(LOG_LIKE)

    @property
    def weight(self) -> Expression | None:
        return self.expressions.get(WEIGHT)

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
        database_audits = (
            [audit_dataframe(data=self.database.dataframe)]
            if self._database is not None
            else []
        )

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
