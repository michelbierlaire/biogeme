"""
Database adapters (strategy objects) for ModelElements.

This module provides a small adapter interface that encapsulates the
preparation and exposure of the working database used to evaluate model
expressions, together with a factory for the corresponding
:class:`~biogeme.expressions_registry.ExpressionRegistry`.

Two concrete adapters are provided:

* :class:`RegularAdapter` – uses the database as-is (no transformation).
* :class:`FlatPanelAdapter` – flattens a panel database and updates
  expressions with the maximum number of observations per individual.

These adapters allow the :class:`ModelElements` container to remain generic
and free of conditional logic about database variants.

Michel Bierlaire
Tue Nov 11 2025, 17:40:56
"""

from __future__ import annotations

from typing import Protocol

from biogeme.database import Database, PanelDatabase
from biogeme.default_parameters import MISSING_VALUE
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression
from biogeme.expressions_registry import ExpressionRegistry


class ModelElementsAdapter(Protocol):
    """
    Strategy interface for providing a working database and registry.

    Any implementation must be able to prepare its internal state given
    the model expressions, expose the database on which expressions will
    be evaluated, and build an :class:`ExpressionRegistry` bound to that
    database.

    Methods
    -------
    prepare(expressions)
        Perform any one-time preparation needed before evaluation
        (e.g., flatten a panel DB, set expression metadata).

    database
        The :class:`~biogeme.database.Database` against which expressions
        are to be evaluated.

    build_registry(expressions)
        Construct an :class:`~biogeme.expressions_registry.ExpressionRegistry`
        connected to the working database.

    sample_size
        Number of data rows in the working database.

    number_of_observations
        Number of original observations in the source database
        (this may differ from ``sample_size`` for flattened panel data).

    :param expressions: Mapping of expression names to :class:`~biogeme.expressions.Expression`
        instances. Implementations may use this to attach metadata.
    """

    def prepare(self, expressions: dict[str, Expression]) -> None: ...
    @property
    def database(self) -> Database: ...
    def build_registry(
        self, expressions: dict[str, Expression]
    ) -> ExpressionRegistry: ...
    @property
    def sample_size(self) -> int: ...
    @property
    def number_of_observations(self) -> int: ...


class RegularAdapter:
    """
    Adapter for a regular (non-panel or non-flattened) database.

    This adapter is a thin wrapper around a :class:`~biogeme.database.Database`
    instance and performs no transformation.



    :note:
        * ``sample_size`` equals ``number_of_observations``.
        * :meth:`prepare` is a no-op.
    """

    def __init__(self, database: Database | None):
        """Ctor
        :param database: Input database. If ``None``, a dummy database is created.
        """
        self._db: Database = database or Database.dummy_database()

    def prepare(self, expressions: dict[str, Expression]) -> None:
        """
        No-op preparation for regular databases.

        :param expressions: Mapping of expression names to expressions (unused).
        """
        return

    @property
    def database(self) -> Database:
        """Working database (the input database)."""
        return self._db

    def build_registry(self, expressions: dict[str, Expression]) -> ExpressionRegistry:
        """
        Build an :class:`ExpressionRegistry` bound to the working database.

        :param expressions: Mapping of expression names to expressions.

        :return: Registry bound to :pyattr:`database`.
        :rtype: ExpressionRegistry
        """
        return ExpressionRegistry(expressions.values(), self._db)

    @property
    def sample_size(self) -> int:
        """Number of rows in the working database."""
        return self._db.num_rows()

    @property
    def number_of_observations(self) -> int:
        """Number of observations in the source database."""
        return self._db.num_rows()


class FlatPanelAdapter:
    """
    Adapter that flattens a panel database for expression evaluation.

    This adapter converts an input panel database into a flat database
    using :class:`~biogeme.database.PanelDatabase`. It also updates each
    expression with the maximum number of observations per individual.

    :note:
        * After :meth:`prepare`, :pyattr:`database` points to the flat database.
        * ``sample_size`` is the number of rows in the flat database.
          ``number_of_observations`` is the number of rows in the original database.
    """

    def __init__(self, database: Database | None):
        """Ctor

        :param database: Input *panel* database to be flattened. If ``None``, a dummy database
        is created. A :class:`BiogemeError` is raised if the database is not
        marked as panel.

        :ivar _orig: Original database passed at construction.
        :ivar _panel: :class:`PanelDatabase` used to perform the flattening.
        :ivar _flat: Resulting flat :class:`Database` after :meth:`prepare` is called.
        :ivar _max_obs: Maximum number of observations per individual computed during
        flattening.

        :raises BiogemeError: If the supplied database is not a panel database.


        """
        self._orig: Database = database or Database.dummy_database()
        if not self._orig.is_panel():
            raise BiogemeError("FlatPanelAdapter requires a panel database.")
        self._panel = PanelDatabase(
            database=self._orig, panel_column=self._orig.panel_column
        )
        self._flat: Database | None = None
        self._max_obs: int | None = None

    def prepare(self, expressions: dict[str, Expression]) -> None:
        """
        Flatten the panel database and update expressions.

        :param expressions: Mapping of expression names to expressions. Each expression is
            informed of the maximum number of observations per individual
            via ``set_maximum_number_of_observations_per_individual``.
        """
        flat_df, max_obs = self._panel.flatten_database(missing_data=MISSING_VALUE)
        self._flat = Database(name=f"flat {self._orig.name}", dataframe=flat_df)
        self._max_obs = max_obs
        for expr in expressions.values():
            expr.set_maximum_number_of_observations_per_individual(max_number=max_obs)

    @property
    def database(self) -> Database:
        """
        Working database, i.e., the flattened database.

        :return: The flat database created in :meth:`prepare`.
        :rtype: Database

        :raises BiogemeError: If :meth:`prepare` has not been called yet.
        """
        if self._flat is None:
            raise BiogemeError("Flat database not prepared. Call 'prepare' first.")
        return self._flat

    def build_registry(self, expressions: dict[str, Expression]) -> ExpressionRegistry:
        """
        Build an :class:`ExpressionRegistry` bound to the flat database.

        :param expressions: Mapping of expression names to expressions.

        :return: Registry bound to :pyattr:`database`.
        :rtype: ExpressionRegistry
        """
        return ExpressionRegistry(expressions.values(), self.database)

    @property
    def sample_size(self) -> int:
        """
        Number of rows in the working (flat) database.

        :return: Number of rows.
        :rtype: int
        """
        return self.database.num_rows()

    @property
    def number_of_observations(self) -> int:
        """
        Number of observations in the original (panel) database.

        :return: Number of observations.
        :rtype: int
        """
        return self._orig.num_rows()
