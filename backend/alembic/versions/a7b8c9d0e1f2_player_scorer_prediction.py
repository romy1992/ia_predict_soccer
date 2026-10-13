"""player_scorer_prediction

Revision ID: a7b8c9d0e1f2
Revises: a1b2c3d4e5f6
Create Date: 2026-10-13 00:00:00.000000

Predizioni "il giocatore segna?" (mercato marcatori), append-only per revisione di formazione.
Tabella nuova, additiva.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a7b8c9d0e1f2'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'player_scorer_prediction',
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False),
        sa.Column('player_name', sa.String(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.String(), nullable=True),
        sa.Column('position', sa.String(), nullable=True),
        sa.Column('league', sa.Integer(), nullable=True),
        sa.Column('kickoff', sa.String(), nullable=True),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('lineup_type', sa.String(), nullable=False),
        sa.Column('model_version', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('scored', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('fixture_id', 'player_id', 'revision'),
    )
    op.create_index('ix_player_scorer_prediction_league', 'player_scorer_prediction', ['league'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_player_scorer_prediction_league', table_name='player_scorer_prediction')
    op.drop_table('player_scorer_prediction')
