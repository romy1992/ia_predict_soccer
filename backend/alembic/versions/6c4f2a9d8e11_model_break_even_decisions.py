"""persist model break-even decision metrics

Revision ID: 6c4f2a9d8e11
Revises: 3e9c7a4b1d20
Create Date: 2026-09-10 14:45:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6c4f2a9d8e11"
down_revision: Union[str, Sequence[str], None] = "3e9c7a4b1d20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for column in (
        sa.Column("model_void_odd", sa.Float(), nullable=True),
        sa.Column("market_fair_odd", sa.Float(), nullable=True),
        sa.Column("odds_edge_absolute", sa.Float(), nullable=True),
        sa.Column("odds_edge_percent", sa.Float(), nullable=True),
        sa.Column("expected_roi_percent", sa.Float(), nullable=True),
        sa.Column("play_threshold_odd", sa.Float(), nullable=True),
        sa.Column("min_edge_percent", sa.Float(), nullable=True),
        sa.Column("value_label", sa.String(), nullable=True),
        sa.Column("value_reason", sa.String(), nullable=True),
    ):
        op.add_column("prediction_ledger", column)


def downgrade() -> None:
    for column in (
        "value_reason",
        "value_label",
        "min_edge_percent",
        "play_threshold_odd",
        "expected_roi_percent",
        "odds_edge_percent",
        "odds_edge_absolute",
        "market_fair_odd",
        "model_void_odd",
    ):
        op.drop_column("prediction_ledger", column)
