"""add shadow betslip settlement

Revision ID: e4c8a17d5b92
Revises: d9b4f6a21c73
Create Date: 2026-09-11 14:20:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4c8a17d5b92"
down_revision: Union[str, Sequence[str], None] = "d9b4f6a21c73"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("diversification_version", sa.String(), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_status", sa.String(length=16), nullable=False, server_default="PENDING"),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_stake", sa.Float(), nullable=False, server_default="1"),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_effective_odd", sa.Float(), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_return", sa.Float(), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_profit", sa.Float(), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_settlement", sa.JSON(), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("shadow_settled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column(
            "staking_policy_version",
            sa.String(),
            nullable=False,
            server_default="shadow_flat_unit_v1",
        ),
    )
    op.create_index(
        "ix_betting_slip_proposal_shadow_status",
        "betting_slip_proposal_snapshots",
        ["shadow_status", "is_latest"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_betting_slip_proposal_shadow_status",
        table_name="betting_slip_proposal_snapshots",
    )
    for column in (
        "staking_policy_version",
        "shadow_settled_at",
        "shadow_settlement",
        "shadow_profit",
        "shadow_return",
        "shadow_effective_odd",
        "shadow_stake",
        "shadow_status",
        "diversification_version",
    ):
        op.drop_column("betting_slip_proposal_snapshots", column)
