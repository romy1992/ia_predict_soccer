"""
Connessione al db
| Caso d’uso                            | Comando Alembic                                             |
| ------------------------------------- | ----------------------------------------------------------- |
| Inizializzazione schema               | `revision --autogenerate -m "inizio"` + `upgrade head`      |
| Aggiungere/modificare campi o tabelle | `revision --autogenerate -m "descrizione"` + `upgrade head` |
| Rollback ultima versione              | `alembic downgrade -1`                                      |
| Tornare a specifica migrazione        | `alembic downgrade <revision_id>`                           |
| Verificare se serve nuova migrazione  | `alembic check`                                             |

"""
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from src.service_ia.config.app_config import load_app_config

_CFG = load_app_config()

# DATABASE_URL viene letto da una sola sorgente di configurazione condivisa.
DATABASE_URL = _CFG.database_url

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# scoped_session (bug fix 2026-09-06): ogni THREAD che chiama SessionLocal()
# ottiene una Session isolata (thread-local), mai la stessa istanza di un
# altro thread. Necessario perche' `CrudRepository` (vedi crud_repository.py)
# e' usato da repository "singleton di modulo" come `repo_match`/
# `repo_snapshot` in download_match_service.py, condivisi tra job
# schedulati DIVERSI (es. data_sync_today ogni 30 min + data_settlement
# ogni 60 min, che richiama a sua volta download_import_matches) che
# girano in thread APScheduler separati ma nello STESSO processo
# `soccer_scheduler`. Con una Session fissa unica (sessionmaker "nudo"),
# due job che finivano per sovrapporsi nel tempo causavano errori
# SQLAlchemy "concurrent operations are not permitted" / "Method close()
# can't be called here" su decine di fixture in un colpo solo, con
# conseguente violazione FK su odds_snapshot (partite mai salvate ma
# quote si', o viceversa) - vedi jobs_history.jsonl 2026-09-06 08:00 UTC.
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
