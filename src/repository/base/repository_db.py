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
from sqlalchemy.orm import sessionmaker

from src.service_ia.config.app_config import load_app_config

_CFG = load_app_config()

# DATABASE_URL viene letto da una sola sorgente di configurazione condivisa.
DATABASE_URL = _CFG.database_url

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
