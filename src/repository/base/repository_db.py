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

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Calcola il path assoluto nella root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "my_database.db")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "properties", "config.env"))

# DATABASE_URL puo essere sovrascritto da config.env o variabili di ambiente di runtime
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/match_db")
# DATABASE_URL = f"sqlite:///{DATABASE_PATH}"  # fallback locale sqlite, se necessario

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
