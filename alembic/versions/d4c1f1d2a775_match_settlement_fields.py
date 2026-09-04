"""match_settlement_fields

Revision ID: d4c1f1d2a775
Revises: c92f3e6b7d11
Create Date: 2026-09-02 11:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4c1f1d2a775'
down_revision: Union[str, Sequence[str], None] = 'c92f3e6b7d11'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('match', sa.Column('is_settled', sa.Boolean(), nullable=True))
    op.add_column('match', sa.Column('settlement_status', sa.String(), nullable=True))
    op.add_column('match', sa.Column('settled_at', sa.String(), nullable=True))
    op.add_column('match', sa.Column('settlement_details', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('match', 'settlement_details')
    op.drop_column('match', 'settled_at')
    op.drop_column('match', 'settlement_status')
    op.drop_column('match', 'is_settled')
