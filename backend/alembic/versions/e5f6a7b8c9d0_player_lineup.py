"""player_lineup

Revision ID: e5f6a7b8c9d0
Revises: d1e2f3a4b5c6
Create Date: 2026-10-07 00:00:00.000000

Cantiere "giocatori che segnano": completa `player_match_event` con
l'universo completo di chi ha giocato una fixture (titolari + panchina),
necessario per costruire un dataset di training "ha segnato si/no" senza
la distorsione di vedere solo i giocatori con eventi (vedi docstring di
`src/data/player/player_lineup_models.py`). Tabella nuova, additiva,
nessun impatto sullo schema esistente.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'player_lineup',
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('player_name', sa.String(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.String(), nullable=True),
        sa.Column('position', sa.String(), nullable=True),
        sa.Column('is_starter', sa.Boolean(), nullable=False),
        sa.Column('formation', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('fixture_id', 'player_id'),
    )
    op.create_index('ix_player_lineup_team_id', 'player_lineup', ['team_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_player_lineup_team_id', table_name='player_lineup')
    op.drop_table('player_lineup')
