from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from biogeme.audit_tuple import AuditTuple, merge_audit_tuples, display_messages
from biogeme.constants import LOG_LIKE, WEIGHT
from biogeme.database import Database
from biogeme.database.audit import audit_dataframe
from biogeme.draws import DrawsManagement
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, audit_expression
from biogeme.expressions.base_expressions import LogitTuple
from biogeme.expressions_registry import ExpressionRegistry
from biogeme.model_elements.audit import audit_chosen_alternative
from biogeme.default_parameters import MISSING_VALUE


class ModelElements:
    """
    Container for all key components required to define and estimate a model.

    :param expressions: Dict of expressions to be evaluated.
    """

    def __init__(
        self,
        expressions: dict[str, Expression],
        database: Database,
        number_of_draws: Optional[int] = 0,
        draws_management: Optional[DrawsManagement] = None,
        expressions_registry: ExpressionRegistry = None,
    ):
        self.expressions = expressions
        self._database = database
        self._flat_database = None
        self.number_of_draws = number_of_draws
        self.draws_management = draws_management
        self.expressions_registry = expressions_registry
        # Validate inputs: only one must be provided
        if self.number_of_draws and self.draws_management:
            raise ValueError(
                "One of 'number_of_draws' or 'draws_management' must be provided (or none of them)."
            )
        self._prepare_for_panel()

        if self.expressions_registry is None:
            self.expressions_registry = ExpressionRegistry(
                self.expressions.values(), self.database
            )

        self.database.register_listener(self.on_database_update)

        if self.draws_management is None:
            self.draws_management = DrawsManagement(
                sample_size=self.database.sample_size,
                number_of_draws=self.number_of_draws,
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

        display_messages(self.audit())

    def is_panel(self) -> bool:
        return self._database.panel_column is not None

    @property
    def database(self) -> Database:
        if not self.is_panel():
            return self._database
        if self._flat_database is None:
            raise BiogemeError('Flat database unavailable for panel data')
        return self._flat_database

    @classmethod
    def from_expression_and_weight(
        cls,
        log_like: Expression,
        database: Database,
        weight: Optional[Expression] = None,
        number_of_draws: Optional[int] = 0,
        draws_management: Optional[DrawsManagement] = None,
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
        )

    def on_database_update(self, updated_index: pd.Index):
        """Update the draws object to remain consistent with the new database"""
        self.draws_management.remove_rows(updated_index)
        self._prepare_for_panel()

    def _prepare_for_panel(self) -> None:
        """If the database contains panel data,  it is flattened"""
        if self._database.panel_column is None:
            return
        flat_dataframe, maximum_number_of_observations_per_individual = (
            self._database.flatten_database(missing_data=MISSING_VALUE)
        )
        self._flat_database = Database(
            name=f'flat {self._database.name}', dataframe=flat_dataframe
        )
        for expression in self.expressions.values():
            expression.set_maximum_number_of_observations_per_individual(
                max_number=maximum_number_of_observations_per_individual
            )

    @property
    def sample_size(self):
        return self.database.sample_size

    @property
    def loglikelihood(self) -> Expression | None:
        return self.expressions.get(LOG_LIKE)

    @property
    def weight(self) -> Expression | None:
        return self.expressions.get(WEIGHT)

    @property
    def formula_names(self) -> list[str]:
        return list(self.expressions.keys())

    def audit(self) -> AuditTuple:
        """Audit the model elements"""

        # First, we audit the expressions.
        expression_audits = [
            audit_expression(expr=expr) for expr in self.expressions.values()
        ]

        # Second, we audit the database
        database_audits = [audit_dataframe(data=self.database.dataframe)]

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
                    )
                    for logit_to_check in logits_to_check
                ]
                if self.loglikelihood is not None
                else []
            )

        return merge_audit_tuples(expression_audits + database_audits + logit_audits)
