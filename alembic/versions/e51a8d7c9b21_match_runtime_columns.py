"""match_runtime_columns

Revision ID: e51a8d7c9b21
Revises: d4c1f1d2a775
Create Date: 2026-09-02 09:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = 'e51a8d7c9b21'
down_revision: Union[str, Sequence[str], None] = 'd4c1f1d2a775'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = inspector.get_columns(table_name)
    return any(col.get("name") == column_name for col in columns)


def upgrade() -> None:
    """Upgrade schema."""
    if not _column_exists("match", "current_league"):
        op.add_column("match", sa.Column("current_league", sa.Integer(), nullable=True))
    if not _column_exists("match", "league_match"):
        op.add_column("match", sa.Column("league_match", sa.Integer(), nullable=True))
    if not _column_exists("match", "status"):
        op.add_column("match", sa.Column("status", sa.String(), nullable=True))
    if not _column_exists("match", "mean_statistics"):
        op.add_column("match", sa.Column("mean_statistics", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    if _column_exists("match", "mean_statistics"):
        op.drop_column("match", "mean_statistics")
    if _column_exists("match", "status"):
        op.drop_column("match", "status")
    if _column_exists("match", "league_match"):
        op.drop_column("match", "league_match")
    if _column_exists("match", "current_league"):
        op.drop_column("match", "current_league")

