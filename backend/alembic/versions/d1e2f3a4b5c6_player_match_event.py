"""player_match_event (bridge)

Revision ID: d1e2f3a4b5c6
Revises: b8e3f6a2c9d4
Create Date: 2026-10-07 00:00:00.000000

BRIDGE, upgrade/downgrade NO-OP. Il DB "dev" condiviso su Railway (STESSO
`DATABASE_URL` usato da questo chain `backend/alembic/` e dal chain radice
`alembic/` del branch `claude/player-goalscorer-odds` - vedi policy
DATABASE_URL in README_PLATFORM.md) e' stato stampato con questa identica
revision id da un `alembic upgrade head` lanciato da quel branch (cantiere
"giocatori che segnano"), che ha creato VERAMENTE la tabella
`player_match_event`. Questo chain (`backend/`, quello che il servizio
`api` di Railway esegue all'avvio) non conosceva quella revision id -
`alembic upgrade head` falliva con "Can't locate revision identified by
'd1e2f3a4b5c6'", healthcheck mai raggiunto, deploy FAILED.

Questo file esiste SOLO per far si' che questo chain riconosca la revision
gia' stampata nel DB e possa avanzare oltre - la tabella e' gia' stata
creata dall'altro chain, quindi qui non si ripete la DDL (altrimenti
"relation already exists").
"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = 'd1e2f3a4b5c6'
down_revision: Union[str, Sequence[str], None] = 'b8e3f6a2c9d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: `player_match_event` esiste gia' (vedi docstring del modulo)."""
    pass


def downgrade() -> None:
    """No-op: il downgrade reale e' di competenza del chain che ha creato
    la tabella (`alembic/versions/d1e2f3a4b5c6_player_match_event.py` sul
    branch `claude/player-goalscorer-odds`)."""
    pass
