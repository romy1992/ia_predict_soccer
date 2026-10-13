"""player_match_stats

Revision ID: a1b2c3d4e5f6
Revises: e5f6a7b8c9d0
Create Date: 2026-10-13 00:00:00.000000

Cantiere "giocatori che segnano": statistiche individuali per partita
(`fixtures/players`: minuti, tiri, rigori, rating...). Tabella nuova,
additiva (vedi `src/data/player/player_stats_models.py`).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INT_COLUMNS = [
    'team_id', 'minutes', 'shots_total', 'shots_on', 'goals', 'assists', 'saves',
    'passes_total', 'key_passes', 'dribbles_attempts', 'dribbles_success',
    'fouls_drawn', 'fouls_committed', 'yellow', 'red', 'pen_won', 'pen_committed',
    'pen_scored', 'pen_missed', 'offsides',
]


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'player_match_stats',
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        *[sa.Column(name, sa.Integer(), nullable=True) for name in _INT_COLUMNS],
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('captain', sa.Boolean(), nullable=True),
        sa.Column('substitute', sa.Boolean(), nullable=True),
        sa.Column('position', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('fixture_id', 'player_id'),
    )
    op.create_index('ix_player_match_stats_team_id', 'player_match_stats', ['team_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_player_match_stats_team_id', table_name='player_match_stats')
    op.drop_table('player_match_stats')
