"""fotmob predicted lineups

Revision ID: c1d2e3f4a5b6
Revises: b8c9d0e1f2a3
Create Date: 2026-10-16 00:00:00.000000

Snapshot append-only delle formazioni previste FotMob e mappa fixture
API-Sports -> match FotMob. Nessuna modifica a `player_lineup`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fotmob_match_map",
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("fotmob_match_id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("mapped_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("fixture_id"),
    )
    op.create_index("ix_fotmob_match_map_fotmob_match_id", "fotmob_match_map", ["fotmob_match_id"])
    op.create_table(
        "fotmob_lineup_snapshot",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=True),
        sa.Column("fotmob_match_id", sa.Integer(), nullable=False),
        sa.Column("lineup_type", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_formation", sa.String(), nullable=True),
        sa.Column("away_formation", sa.String(), nullable=True),
        sa.Column("home_team_fotmob_id", sa.Integer(), nullable=True),
        sa.Column("away_team_fotmob_id", sa.Integer(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=True),
        sa.Column("away_team_id", sa.Integer(), nullable=True),
        sa.Column("reliability", sa.String(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("starter_count", sa.Integer(), nullable=False),
        sa.Column("mapped_starters", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fotmob_lineup_snapshot_fixture_id", "fotmob_lineup_snapshot", ["fixture_id"])
    op.create_index("ix_fotmob_lineup_snapshot_fotmob_match_id", "fotmob_lineup_snapshot", ["fotmob_match_id"])
    op.create_index("ix_fotmob_lineup_snapshot_fetched_at", "fotmob_lineup_snapshot", ["fetched_at"])
    op.create_index("ix_fotmob_lineup_snapshot_reliability", "fotmob_lineup_snapshot", ["reliability"])
    op.create_table(
        "fotmob_lineup_player",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("fotmob_player_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("shirt_number", sa.String(), nullable=True),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("market_value", sa.Integer(), nullable=True),
        sa.Column("api_player_id", sa.Integer(), nullable=True),
        sa.Column("api_team_id", sa.Integer(), nullable=True),
        sa.Column("map_method", sa.String(), nullable=True),
        sa.Column("historical_position", sa.String(), nullable=True),
        sa.Column("unavailability_type", sa.String(), nullable=True),
        sa.Column("expected_return", sa.String(), nullable=True),
        sa.Column("expected_return_date", sa.String(), nullable=True),
        sa.Column("return_certainty", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fotmob_lineup_player_snapshot_id", "fotmob_lineup_player", ["snapshot_id"])


def downgrade() -> None:
    op.drop_index("ix_fotmob_lineup_player_snapshot_id", table_name="fotmob_lineup_player")
    op.drop_table("fotmob_lineup_player")
    op.drop_index("ix_fotmob_lineup_snapshot_reliability", table_name="fotmob_lineup_snapshot")
    op.drop_index("ix_fotmob_lineup_snapshot_fetched_at", table_name="fotmob_lineup_snapshot")
    op.drop_index("ix_fotmob_lineup_snapshot_fotmob_match_id", table_name="fotmob_lineup_snapshot")
    op.drop_index("ix_fotmob_lineup_snapshot_fixture_id", table_name="fotmob_lineup_snapshot")
    op.drop_table("fotmob_lineup_snapshot")
    op.drop_index("ix_fotmob_match_map_fotmob_match_id", table_name="fotmob_match_map")
    op.drop_table("fotmob_match_map")
