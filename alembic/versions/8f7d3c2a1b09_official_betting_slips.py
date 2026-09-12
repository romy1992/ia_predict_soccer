"""add official betting slips

Revision ID: 8f7d3c2a1b09
Revises: 6c4f2a9d8e11
Create Date: 2026-09-10 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8f7d3c2a1b09"
down_revision: Union[str, Sequence[str], None] = "6c4f2a9d8e11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "betting_slips",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("capture_key", sa.String(length=160), nullable=False),
        sa.Column("reference_date", sa.String(length=10), nullable=False),
        sa.Column("profile", sa.String(length=24), nullable=False),
        sa.Column("initial_situation", sa.String(length=16), nullable=False),
        sa.Column("initial_reason", sa.String(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("combined_odd", sa.Float(), nullable=False),
        sa.Column("naive_probability", sa.Float(), nullable=True),
        sa.Column("adjusted_probability", sa.Float(), nullable=True),
        sa.Column("combined_model_void_odd", sa.Float(), nullable=True),
        sa.Column("combined_edge_absolute", sa.Float(), nullable=True),
        sa.Column("combined_edge_percent", sa.Float(), nullable=True),
        sa.Column("combined_expected_roi", sa.Float(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("combined_play_threshold", sa.Float(), nullable=True),
        sa.Column("slip_min_edge_percent", sa.Float(), nullable=True),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("potential_return", sa.Float(), nullable=True),
        sa.Column("effective_combined_odd", sa.Float(), nullable=True),
        sa.Column("actual_return", sa.Float(), nullable=True),
        sa.Column("realized_profit", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("correlation_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("capture_key"),
    )
    op.create_index("ix_betting_slips_reference_profile", "betting_slips", ["reference_date", "profile"])
    op.create_index("ix_betting_slips_status", "betting_slips", ["status"])

    op.create_table(
        "betting_slip_picks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("slip_id", sa.String(length=36), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("competition", sa.String(), nullable=True),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team", sa.String(), nullable=True),
        sa.Column("away_team", sa.String(), nullable=True),
        sa.Column("market", sa.String(), nullable=False),
        sa.Column("line", sa.String(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("p_model", sa.Float(), nullable=False),
        sa.Column("market_odd", sa.Float(), nullable=False),
        sa.Column("model_void_odd", sa.Float(), nullable=True),
        sa.Column("market_fair_odd", sa.Float(), nullable=True),
        sa.Column("odds_edge_absolute", sa.Float(), nullable=True),
        sa.Column("odds_edge_percent", sa.Float(), nullable=True),
        sa.Column("expected_roi", sa.Float(), nullable=True),
        sa.Column("situation", sa.String(length=16), nullable=False),
        sa.Column("bookmakers_count", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("policy_version", sa.String(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("final_score", sa.String(length=24), nullable=True),
        sa.Column("void_reason", sa.String(), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["prediction_id"], ["prediction_ledger.id_prediction"]),
        sa.ForeignKeyConstraint(["slip_id"], ["betting_slips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_betting_slip_picks_slip_position",
        "betting_slip_picks",
        ["slip_id", "position"],
        unique=True,
    )
    op.create_index("ix_betting_slip_picks_fixture", "betting_slip_picks", ["fixture_id"])


def downgrade() -> None:
    op.drop_index("ix_betting_slip_picks_fixture", table_name="betting_slip_picks")
    op.drop_index("ix_betting_slip_picks_slip_position", table_name="betting_slip_picks")
    op.drop_table("betting_slip_picks")
    op.drop_index("ix_betting_slips_status", table_name="betting_slips")
    op.drop_index("ix_betting_slips_reference_profile", table_name="betting_slips")
    op.drop_table("betting_slips")
