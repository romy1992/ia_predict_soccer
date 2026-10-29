"""admin_credential

Revision ID: e4f5a6b7c8d9
Revises: d2e3f4a5b6c7
Create Date: 2026-10-29 00:00:00.000000

Credenziali del pannello modificabili dal portale (riga unica). Tabella nuova, additiva.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'e4f5a6b7c8d9'
down_revision: Union[str, Sequence[str], None] = 'd2e3f4a5b6c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'admin_credential',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('admin_credential')
