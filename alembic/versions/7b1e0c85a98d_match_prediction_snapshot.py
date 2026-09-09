"""match_prediction_snapshot

Revision ID: 7b1e0c85a98d
Revises: a7d92c4f18e6
Create Date: 2026-09-09 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7b1e0c85a98d'
down_revision: Union[str, Sequence[str], None] = 'a7d92c4f18e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'match_prediction_snapshot',
        sa.Column('id_snapshot', sa.String(length=36), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('market', sa.String(), nullable=False),
        sa.Column('prediction', sa.Integer(), nullable=False),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('model_run_id', sa.String(), nullable=True),
        sa.Column('feature_fingerprint', sa.String(), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id_snapshot'),
    )
    op.create_index(
        'ix_match_prediction_snapshot_fixture_market_computed',
        'match_prediction_snapshot',
        ['fixture_id', 'market', 'computed_at'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_match_prediction_snapshot_fixture_market_computed', table_name='match_prediction_snapshot')
    op.drop_table('match_prediction_snapshot')
