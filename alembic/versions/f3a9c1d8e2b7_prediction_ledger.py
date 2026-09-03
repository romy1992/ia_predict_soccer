"""prediction_ledger

Revision ID: f3a9c1d8e2b7
Revises: e51a8d7c9b21
Create Date: 2026-09-03 08:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a9c1d8e2b7'
down_revision: Union[str, Sequence[str], None] = 'e51a8d7c9b21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'prediction_ledger',
        sa.Column('id_prediction', sa.String(length=36), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('market', sa.String(), nullable=False),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('model_run_id', sa.String(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('policy_version', sa.String(), nullable=True),
        sa.Column('p_model', sa.Float(), nullable=True),
        sa.Column('p_market_fair', sa.Float(), nullable=True),
        sa.Column('odd', sa.Float(), nullable=True),
        sa.Column('fair_odd', sa.Float(), nullable=True),
        sa.Column('prob_edge', sa.Float(), nullable=True),
        sa.Column('ev', sa.Float(), nullable=True),
        sa.Column('decision', sa.String(), nullable=False),
        sa.Column('stake', sa.Float(), nullable=False),
        sa.Column('kickoff_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_settled', sa.Boolean(), nullable=False),
        sa.Column('settled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('settlement_status', sa.String(), nullable=True),
        sa.Column('actual_outcome', sa.String(), nullable=True),
        sa.Column('won', sa.Boolean(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id_prediction'),
    )
    op.create_index('ix_prediction_ledger_fixture_id', 'prediction_ledger', ['fixture_id'], unique=False)
    op.create_index('ix_prediction_ledger_market', 'prediction_ledger', ['market'], unique=False)
    op.create_index('ix_prediction_ledger_is_settled', 'prediction_ledger', ['is_settled'], unique=False)
    op.create_index('ix_prediction_ledger_kickoff_at', 'prediction_ledger', ['kickoff_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_prediction_ledger_kickoff_at', table_name='prediction_ledger')
    op.drop_index('ix_prediction_ledger_is_settled', table_name='prediction_ledger')
    op.drop_index('ix_prediction_ledger_market', table_name='prediction_ledger')
    op.drop_index('ix_prediction_ledger_fixture_id', table_name='prediction_ledger')
    op.drop_table('prediction_ledger')

