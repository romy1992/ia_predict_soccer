"""Mercati "in osservazione" per la Schedina Oracle.

Un mercato puo' restare in `production` nel `ModelRegistry` (quindi ancora
"attivo" per Dashboard/pronostici singoli) ma essere temporaneamente escluso
dal MIX multi-mercato delle schedine, quando l'operatore vuole isolarne il
contributo senza ritirarlo del tutto dal registry (a differenza del ritiro
esplicito, es. corners 2026-09-21, gia' gestito da
`ModelRegistry.list_active_markets`).

Caso reale che ha motivato questo modulo (2026-09-28): 3 delle 4 linee
`cards` sono state promosse in produzione con `force=True` nonostante il
gate automatico le avesse respinte (`report_cards_promozione_forzata.md`) -
l'operatore vuole continuare a raccogliere un track record ufficiale su
quelle linee, ma non farle contribuire nel frattempo alle schedine
multi-mercato "di fiducia" (dove il loro eventuale rumore si spalmerebbe
sull'intero mix, indistinguibile dagli altri mercati).

Lista versionata e persistita su disco (stesso pattern di
`src/jobs/job_settings.py::get_job_settings`/`update_job_settings`: file
JSON con scrittura atomica via file temporaneo + `os.replace`, mai un
edit in-place che un lettore concorrente potrebbe leggere a meta').
Nessun mercato viene MAI escluso silenziosamente: l'esclusione e' sempre
un'entrata esplicita in questa lista, mai una soglia calcolata al volo.

Un mercato isolato resta comunque generabile ESPLICITAMENTE (es. una futura
sezione "schedine mono-mercato" che lo interroga da solo, o un chiamante che
passa `markets=["cards"]` di proposito): l'isolamento si applica SOLO al
default "tutti i mercati attivi" usato dal mix multi-mercato
(`BetslipService.generate_exploration_for_day` quando `markets is None`),
mai a una richiesta che nomina esplicitamente il mercato.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Default deliberatamente prudente (mai un edit silenzioso: una nuova
# esclusione/rimozione passa da `set_isolated_markets`, mai da qui):
# "cards" isolato dal 2026-09-28 per la promozione forzata di 3/4 linee
# (vedi docstring di modulo).
DEFAULT_ISOLATED_MARKETS: frozenset[str] = frozenset({"cards"})


def _isolated_markets_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(project_root, "best_models", "isolated_markets.json")


def get_isolated_markets() -> frozenset[str]:
    """Legge la lista corrente dei mercati isolati, sempre normalizzata
    (stringhe, minuscolo, senza spazi) - un file assente/vuoto/corrotto
    ritorna il default, mai un'eccezione propagata al chiamante (stesso
    principio di `job_settings.get_job_settings`)."""
    path = _isolated_markets_path()
    if not os.path.exists(path):
        return DEFAULT_ISOLATED_MARKETS
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("isolated_markets.json illeggibile (%s): uso i default", exc)
        return DEFAULT_ISOLATED_MARKETS
    if not isinstance(loaded, list):
        logger.warning("isolated_markets.json non e' una lista: uso i default")
        return DEFAULT_ISOLATED_MARKETS
    return frozenset(str(m).strip().lower() for m in loaded if str(m).strip())


def set_isolated_markets(markets: list[str]) -> frozenset[str]:
    """Sostituisce INTERAMENTE la lista (mai un merge parziale: a
    differenza dei toggle per-job di `job_settings`, qui un'unica lista
    esplicita e' piu' chiara di un dizionario di flag per un insieme di
    mercati che cambia raramente). Persistenza atomica (file temporaneo +
    `os.replace`), stesso principio di `job_settings.update_job_settings`."""
    normalized = sorted({str(m).strip().lower() for m in markets if str(m).strip()})
    with _LOCK:
        path = _isolated_markets_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    return frozenset(normalized)
