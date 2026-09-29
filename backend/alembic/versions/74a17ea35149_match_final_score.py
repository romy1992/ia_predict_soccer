"""match_final_score

Revision ID: 74a17ea35149
Revises: 7b1e0c85a98d
Create Date: 2026-09-10 07:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74a17ea35149'
down_revision: Union[str, Sequence[str], None] = '7b1e0c85a98d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('match', sa.Column('score_home', sa.Integer(), nullable=True))
    op.add_column('match', sa.Column('score_away', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('match', 'score_away')
    op.drop_column('match', 'score_home')
