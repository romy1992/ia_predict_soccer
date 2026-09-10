"""official prediction ledger

Revision ID: 3e9c7a4b1d20
Revises: 74a17ea35149
Create Date: 2026-09-10 13:45:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "3e9c7a4b1d20"
down_revision: Union[str, Sequence[str], None] = "74a17ea35149"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("prediction_ledger", sa.Column("p_market_raw", sa.Float(), nullable=True))
    op.add_column("prediction_ledger", sa.Column("period", sa.String(), nullable=False, server_default="full_time"))
    op.add_column("prediction_ledger", sa.Column("line", sa.String(), nullable=True))
    op.add_column("prediction_ledger", sa.Column("source", sa.String(), nullable=False, server_default="manual"))
    op.add_column("prediction_ledger", sa.Column("cohort", sa.String(), nullable=False, server_default="legacy"))
    op.add_column("prediction_ledger", sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("prediction_ledger", sa.Column("odds_captured_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("prediction_ledger", sa.Column("bookmaker_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("prediction_ledger", sa.Column("league", sa.Integer(), nullable=True))
    op.add_column("prediction_ledger", sa.Column("capture_key", sa.String(length=160), nullable=True))
    op.execute("UPDATE prediction_ledger SET captured_at = created_at WHERE captured_at IS NULL")
    op.alter_column("prediction_ledger", "captured_at", nullable=False)
    op.create_index("uq_prediction_ledger_capture_key", "prediction_ledger", ["capture_key"], unique=True)
    op.create_index(
        "ix_prediction_ledger_cohort_captured_at",
        "prediction_ledger",
        ["cohort", "captured_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_ledger_cohort_captured_at", table_name="prediction_ledger")
    op.drop_index("uq_prediction_ledger_capture_key", table_name="prediction_ledger")
    for column in (
        "capture_key",
        "league",
        "bookmaker_count",
        "odds_captured_at",
        "captured_at",
        "cohort",
        "source",
        "line",
        "period",
        "p_market_raw",
    ):
        op.drop_column("prediction_ledger", column)
