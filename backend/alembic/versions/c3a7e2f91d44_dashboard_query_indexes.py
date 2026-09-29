"""add dashboard query indexes

Revision ID: c3a7e2f91d44
Revises: 8f7d3c2a1b09
Create Date: 2026-09-11 05:15:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "c3a7e2f91d44"
down_revision: Union[str, Sequence[str], None] = "8f7d3c2a1b09"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_statistics_id_match", "statistics", ["id_match"], unique=False)
    op.create_index("ix_odds_id_match", "odds", ["id_match"], unique=False)
    op.create_index(
        "ix_prediction_ledger_cohort_fixture",
        "prediction_ledger",
        ["cohort", "fixture_id"],
        unique=False,
    )
    op.create_index("ix_prediction_ledger_created_at", "prediction_ledger", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_prediction_ledger_created_at", table_name="prediction_ledger")
    op.drop_index("ix_prediction_ledger_cohort_fixture", table_name="prediction_ledger")
    op.drop_index("ix_odds_id_match", table_name="odds")
    op.drop_index("ix_statistics_id_match", table_name="statistics")
