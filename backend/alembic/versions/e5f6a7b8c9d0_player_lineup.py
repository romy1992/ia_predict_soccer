"""player_lineup (bridge)

Revision ID: e5f6a7b8c9d0
Revises: d1e2f3a4b5c6
Create Date: 2026-10-08 00:00:00.000000

BRIDGE, upgrade/downgrade NO-OP. Stesso identico scenario del bridge
precedente (d1e2f3a4b5c6_player_match_event.py): il DB "dev" condiviso e'
stato stampato con questa revision id da un `alembic upgrade head` lanciato
dal branch `claude/player-goalscorer-odds-backend` (cantiere "giocatori che
segnano"), che ha creato VERAMENTE la tabella `player_lineup`. Questo chain
(quello che il servizio `api` di Railway esegue all'avvio) non conosceva
quella revision id - stesso fallimento "Can't locate revision identified
by 'e5f6a7b8c9d0'", stesso identico fix: riconoscerla senza ripetere la DDL
(la tabella esiste gia').
"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: `player_lineup` esiste gia' (vedi docstring del modulo)."""
    pass


def downgrade() -> None:
    """No-op: il downgrade reale e' di competenza del chain che ha creato
    la tabella (`alembic/versions/e5f6a7b8c9d0_player_lineup.py` sul branch
    `claude/player-goalscorer-odds-backend`)."""
    pass
