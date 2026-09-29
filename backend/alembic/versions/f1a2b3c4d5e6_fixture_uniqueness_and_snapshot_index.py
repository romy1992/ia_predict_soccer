"""match.id_fixture unico + indice su odds_snapshot.id_match

Revision ID: f1a2b3c4d5e6
Revises: e4c8a17d5b92
Create Date: 2026-09-15 08:10:00.000000

Due indici mancanti che insieme spiegavano la lentezza di ogni job che
carica un `Match` dall'ORM, piu' il vincolo che impedisce alle righe
duplicate di ripresentarsi.

1) `ix_odds_snapshot_id_match` - `odds_snapshot.id_match` e' una FK verso
   `match.id_match_fk` e Postgres NON indicizza le FK da se'. La migration
   `c3a7e2f91d44` aveva aggiunto l'indice equivalente su `statistics` e
   `odds` ma non su `odds_snapshot`, che e' la tabella piu' grande del DB
   (311.779 righe / 113 MB, media 693 snapshot per match). Con
   `Match.odds_snapshots` in `lazy="selectin"` ogni caricamento di un
   Match faceva quindi un Parallel Seq Scan su tutta la tabella:

       EXPLAIN select * from odds_snapshot where id_match = '...'
       -> Parallel Seq Scan, Rows Removed by Filter: 309.606

   Misurato: 1,54 s per caricare UNA fixture, che su 193 fixture spiega i
   456 s di un `future_sync`. L'indice resta utile anche dopo il passaggio
   a `lazy="select"` (vedi `Match.odds_snapshots`), perche' il caricamento
   esplicito con `selectinload` fa la stessa query.

2) `uq_match_id_fixture` - `Match.id_fixture` aveva `unique=True`
   commentato nel modello, quindi niente impediva due righe per la stessa
   partita. Ne sono nate 116 (232 righe) da esecuzioni sovrapposte di
   `download_import_matches`, con l'effetto che una riga restava a `NS`
   per sempre e la Dashboard mostrava la partita "In diretta" a distanza
   di giorni. E' un indice UNICO (non un table constraint) di proposito:
   serve anche a `ON CONFLICT (id_fixture)` dell'upsert in
   `download_match_service.py`, che su un constraint di tabella non
   potrebbe fare inferenza. `id_fixture` resta NULLABLE e in Postgres un
   unique index ammette piu' NULL, quindi le righe legacy senza fixture
   (import storici da odds-api) non sono toccate.

PREREQUISITO: i duplicati vanno fusi PRIMA di questa migration, con
`python scripts/maintenance/dedup_match_id_fixture.py --apply`. Se sono
ancora a DB, `create_index(unique=True)` fallisce - ed e' il
comportamento voluto: meglio una migration che si ferma di un vincolo
applicato scartando righe in silenzio.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e4c8a17d5b92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_odds_snapshot_id_match", "odds_snapshot", ["id_match"], unique=False)
    op.create_index("uq_match_id_fixture", "match", ["id_fixture"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_match_id_fixture", table_name="match")
    op.drop_index("ix_odds_snapshot_id_match", table_name="odds_snapshot")
