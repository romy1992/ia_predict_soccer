"""match_date_match_index

Fix performance (dashboard): `DashboardService._fetch_matches` filtra ora
per range su `match.date_match` (bug fix, prima nessun filtro SQL: caricava
l'intero storico ad ogni richiesta dashboard, 22.6s misurati su 47k righe).
Il filtro riduce gia' drasticamente le righe caricate anche senza indice
(sequential scan su una tabella di decine di migliaia di righe e' comunque
sub-secondo su Postgres), ma un indice btree rende il range scan efficiente
anche quando lo storico continuera' a crescere. Migration puramente
additiva (CREATE INDEX), nessun rischio per i dati esistenti.

Revision ID: 9c4e7a21f6b3
Revises: f3a9c1d8e2b7
Create Date: 2026-09-03 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '9c4e7a21f6b3'
down_revision: Union[str, Sequence[str], None] = 'f3a9c1d8e2b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index('ix_match_date_match', 'match', ['date_match'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_match_date_match', table_name='match')

