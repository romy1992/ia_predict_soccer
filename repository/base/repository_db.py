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
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Calcola il path assoluto nella root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "my_database.db")
# DATABASE_URL = f"sqlite:///{DATABASE_PATH}" PER quello locale sqlite
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/match_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
