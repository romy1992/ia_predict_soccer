"""odds_snapshot

Revision ID: c92f3e6b7d11
Revises: 55bbb5f0a367
Create Date: 2026-09-02 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c92f3e6b7d11'
down_revision: Union[str, Sequence[str], None] = '55bbb5f0a367'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'odds_snapshot',
        sa.Column('id_snapshot', sa.String(length=40), nullable=False),
        sa.Column('id_match', sa.String(length=36), nullable=True),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('bookmaker', sa.String(), nullable=False),
        sa.Column('market', sa.String(), nullable=False),
        sa.Column('period', sa.String(), nullable=False),
        sa.Column('line', sa.String(), nullable=True),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('odd', sa.Float(), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['id_match'], ['match.id_match_fk']),
        sa.PrimaryKeyConstraint('id_snapshot'),
    )
    op.create_index('ix_odds_snapshot_fixture_id', 'odds_snapshot', ['fixture_id'], unique=False)
    op.create_index('ix_odds_snapshot_market', 'odds_snapshot', ['market'], unique=False)
    op.create_index('ix_odds_snapshot_captured_at', 'odds_snapshot', ['captured_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_odds_snapshot_captured_at', table_name='odds_snapshot')
    op.drop_index('ix_odds_snapshot_market', table_name='odds_snapshot')
    op.drop_index('ix_odds_snapshot_fixture_id', table_name='odds_snapshot')
    op.drop_table('odds_snapshot')
