"""add betslip proposal snapshots

Revision ID: d9b4f6a21c73
Revises: c3a7e2f91d44
Create Date: 2026-09-11 10:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d9b4f6a21c73"
down_revision: Union[str, Sequence[str], None] = "c3a7e2f91d44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "betting_slip_proposal_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("snapshot_key", sa.String(length=64), nullable=False),
        sa.Column("logical_slip_id", sa.String(length=64), nullable=False),
        sa.Column("supersedes_id", sa.String(length=36), nullable=True),
        sa.Column("reference_date", sa.String(length=10), nullable=False),
        sa.Column("profile", sa.String(length=24), nullable=False),
        sa.Column("situation", sa.String(length=16), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("combined_odd", sa.Float(), nullable=False),
        sa.Column("adjusted_probability", sa.Float(), nullable=True),
        sa.Column("combined_model_void_odd", sa.Float(), nullable=True),
        sa.Column("combined_edge_absolute", sa.Float(), nullable=True),
        sa.Column("combined_expected_roi", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("correlation_version", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("is_latest", sa.Boolean(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["supersedes_id"],
            ["betting_slip_proposal_snapshots.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("snapshot_key"),
    )
    op.create_index(
        "ix_betting_slip_proposal_reference_profile",
        "betting_slip_proposal_snapshots",
        ["reference_date", "profile"],
    )
    op.create_index(
        "ix_betting_slip_proposal_lineage_latest",
        "betting_slip_proposal_snapshots",
        ["logical_slip_id", "is_latest"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_betting_slip_proposal_lineage_latest",
        table_name="betting_slip_proposal_snapshots",
    )
    op.drop_index(
        "ix_betting_slip_proposal_reference_profile",
        table_name="betting_slip_proposal_snapshots",
    )
    op.drop_table("betting_slip_proposal_snapshots")
