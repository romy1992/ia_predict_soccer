"""live_data_tables

Revision ID: a7d92c4f18e6
Revises: 9c4e7a21f6b3
Create Date: 2026-09-04 09:00:00.000000

LIVE-01: dataset LIVE distinto da `match`/`statistics`/`odds` (il dataset
pre-match) - tre tabelle nuove, additive, nessun impatto sullo schema
esistente ("compatibilita' DB storico").
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7d92c4f18e6'
down_revision: Union[str, Sequence[str], None] = '9c4e7a21f6b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'live_fixture_snapshot',
        sa.Column('id_snapshot', sa.String(length=36), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('league_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('elapsed_minute', sa.Integer(), nullable=True),
        sa.Column('elapsed_extra', sa.Integer(), nullable=True),
        sa.Column('home_team_id', sa.Integer(), nullable=True),
        sa.Column('away_team_id', sa.Integer(), nullable=True),
        sa.Column('home_goals', sa.Integer(), nullable=True),
        sa.Column('away_goals', sa.Integer(), nullable=True),
        sa.Column('raw_payload', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id_snapshot'),
    )
    op.create_index('ix_live_fixture_snapshot_fixture_id', 'live_fixture_snapshot', ['fixture_id'], unique=False)
    op.create_index('ix_live_fixture_snapshot_captured_at', 'live_fixture_snapshot', ['captured_at'], unique=False)

    op.create_table(
        'live_match_event',
        sa.Column('id_event', sa.String(length=40), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('elapsed_minute', sa.Integer(), nullable=True),
        sa.Column('elapsed_extra', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(), nullable=True),
        sa.Column('event_detail', sa.String(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.String(), nullable=True),
        sa.Column('player_name', sa.String(), nullable=True),
        sa.Column('assist_name', sa.String(), nullable=True),
        sa.Column('comments', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id_event'),
    )
    op.create_index('ix_live_match_event_fixture_id', 'live_match_event', ['fixture_id'], unique=False)
    op.create_index('ix_live_match_event_captured_at', 'live_match_event', ['captured_at'], unique=False)

    op.create_table(
        'live_fixture_stat_snapshot',
        sa.Column('id_snapshot', sa.String(length=36), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('stats', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id_snapshot'),
    )
    op.create_index(
        'ix_live_fixture_stat_snapshot_fixture_id', 'live_fixture_stat_snapshot', ['fixture_id'], unique=False
    )
    op.create_index(
        'ix_live_fixture_stat_snapshot_captured_at', 'live_fixture_stat_snapshot', ['captured_at'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_live_fixture_stat_snapshot_captured_at', table_name='live_fixture_stat_snapshot')
    op.drop_index('ix_live_fixture_stat_snapshot_fixture_id', table_name='live_fixture_stat_snapshot')
    op.drop_table('live_fixture_stat_snapshot')

    op.drop_index('ix_live_match_event_captured_at', table_name='live_match_event')
    op.drop_index('ix_live_match_event_fixture_id', table_name='live_match_event')
    op.drop_table('live_match_event')

    op.drop_index('ix_live_fixture_snapshot_captured_at', table_name='live_fixture_snapshot')
    op.drop_index('ix_live_fixture_snapshot_fixture_id', table_name='live_fixture_snapshot')
    op.drop_table('live_fixture_snapshot')
