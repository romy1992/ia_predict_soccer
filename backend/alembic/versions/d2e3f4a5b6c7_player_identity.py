"""player_identity

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-10-17 00:00:00.000000

Anagrafica unica dei giocatori (`player_identity`) e collegamenti ai provider esterni (`player_provider_link`).
Tabelle nuove, additive.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'd2e3f4a5b6c7'
down_revision: Union[str, Sequence[str], None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'player_identity',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('canonical_name', sa.String(), nullable=False),
        sa.Column('birth_date', sa.String(), nullable=True),
        sa.Column('position', sa.String(), nullable=True),
        sa.Column('team_api_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_player_identity_team_api_id', 'player_identity', ['team_api_id'])
    op.create_table(
        'player_provider_link',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('identity_id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('provider_player_id', sa.String(), nullable=False),
        sa.Column('provider_name', sa.String(), nullable=True),
        sa.Column('name_norm', sa.String(), nullable=True),
        sa.Column('provider_team', sa.String(), nullable=True),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('provider', 'provider_player_id', name='uq_player_provider_link'),
    )
    op.create_index('ix_player_provider_link_identity_id', 'player_provider_link', ['identity_id'])
    op.create_index('ix_player_provider_link_provider', 'player_provider_link', ['provider'])
    op.create_index('ix_player_provider_link_name_norm', 'player_provider_link', ['name_norm'])


def downgrade() -> None:
    op.drop_index('ix_player_provider_link_name_norm', table_name='player_provider_link')
    op.drop_index('ix_player_provider_link_provider', table_name='player_provider_link')
    op.drop_index('ix_player_provider_link_identity_id', table_name='player_provider_link')
    op.drop_table('player_provider_link')
    op.drop_index('ix_player_identity_team_api_id', table_name='player_identity')
    op.drop_table('player_identity')
