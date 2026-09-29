"""Percorsi dei file di modello: una sola regola, condivisa.

Fino al 2026-09-15 `best_models/` era piatta: tutti i `.pkl` di tutti i
mercati nella stessa cartella, distinti solo dal nome. Con la riorganizzazione
la cartella ha una forma:

    best_models/
        under_over/under_over_1_5/   i modelli della procedura nuova
        under_over/under_over_2_5/
        under_over/under_over_3_5/
        h2h/                         champion h2h (quote per esito)
        dc/                          champion dc (quote per esito)
        archivio/                    tutto quello della vecchia procedura
        registry/                    index.jsonl + metadati

Questo modulo esiste perche' quella forma richiede di trattare il percorso di
un modello come `best_models` + un percorso RELATIVO, non piu' come
`best_models` + un nome file. Tre punti del codice usavano `os.path.basename`
per ricostruire il percorso: con le sottocartelle appiattirebbero
`under_over/under_over_1_5/x.pkl` in `x.pkl`, e il file non si troverebbe piu'.

Le tre funzioni servono a tre momenti diversi:
- `to_container_path`   quando si SCRIVE nel registry (il percorso deve essere
                        quello visto dai container, `/app/best_models/...`,
                        non quello della macchina che ha addestrato);
- `resolve_model_path`  quando si LEGGE per caricare un modello (il percorso
                        registrato puo' essere in forma container, assoluta di
                        un'altra macchina, o puntare a un file nel frattempo
                        spostato);
- `destination_subdir`  quando si decide DOVE salvare un modello nuovo.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

BEST_MODELS_DIRNAME = "best_models"
CONTAINER_BEST_MODELS = "/app/best_models"
ARCHIVIO_DIRNAME = "archivio"

# Mercati rifatti con la procedura nuova Under/Over (quote separate per
# esito, 33 feature). Restano in `under_over/<mercato>/`. Non mescolare con
# h2h/dc: quelli hanno un numero di feature diverso e stanno in
# `best_models/<mercato>/`.
MERCATI_NUOVA_PROCEDURA = ("under_over_1_5", "under_over_2_5", "under_over_3_5")

# Cartella = nome mercato (`best_models/h2h`, `best_models/dc`). I pkl
# vecchi restano in `archivio/<mercato>/` e non si cancellano.
MERCATI_CARTELLA_EVENTO = ("h2h", "dc", "goal_no_goal")

# Mercati a linea (2026-09-16): `corners_line_8_5`, `cards_line_3_5`, ecc.
# (`LINE_MARKETS` in `filter_market_service.py`) - una cartella per FAMIGLIA
# (`best_models/corners/<mercato>/`, `best_models/cards/<mercato>/`), cosi'
# le linee della stessa famiglia restano raggruppate invece di finire nella
# radice piatta insieme a tutto il resto non ancora rifatto.
_RE_MERCATO_LINEA = re.compile(r"^(corners|cards)_line_\d+_\d+$")


def best_models_root() -> str:
    """Radice locale di `best_models`.

    Risolta ad ogni chiamata rispetto alla directory corrente, come gia' fa
    `ModelRegistry` (che usa `os.path.join("best_models", "registry")`): nei
    container la CWD e' `/app`, sulla macchina dell'operatore e' la radice del
    repo. Mai una costante calcolata all'import, che congelerebbe la CWD del
    momento in cui il modulo viene caricato.
    """
    return os.path.abspath(BEST_MODELS_DIRNAME)


def destination_subdir(market: str) -> str:
    """Sottocartella (relativa a `best_models`) dove salvare un modello nuovo.

    Under/Over 1.5/2.5/3.5: `under_over/<mercato>`.
    h2h e dc: `<mercato>` (cioe' `best_models/h2h`, `best_models/dc`).
    Tutti gli altri: radice, finche' non vengono rifatti.
    """
    if market in MERCATI_NUOVA_PROCEDURA:
        return os.path.join("under_over", market)
    if market in MERCATI_CARTELLA_EVENTO:
        return market
    linea = _RE_MERCATO_LINEA.match(market)
    if linea:
        return os.path.join(linea.group(1), market)
    return ""


def relative_to_best_models(path: str) -> Optional[str]:
    """Percorso relativo a `best_models`, con separatori `/`.

    Ritorna `None` se il percorso non sta sotto `best_models`, in nessuna
    delle due forme (locale o container). Serve a distinguere "so dove sta
    questo file dentro la cartella dei modelli" da "questo percorso e'
    qualcos'altro", senza indovinare.
    """
    if not path:
        return None

    normalizzato = str(path).replace("\\", "/").rstrip("/")
    container = CONTAINER_BEST_MODELS.replace("\\", "/")
    if normalizzato == container or normalizzato.startswith(container + "/"):
        return normalizzato[len(container):].lstrip("/") or None

    # Forma locale: qualunque macchina, qualunque radice. Si cerca l'ULTIMA
    # occorrenza di `best_models/` perche' un percorso puo' contenerla piu'
    # volte (es. un repo clonato in una cartella che si chiama cosi').
    marcatore = f"/{BEST_MODELS_DIRNAME}/"
    posizione = normalizzato.rfind(marcatore)
    if posizione != -1:
        return normalizzato[posizione + len(marcatore):].lstrip("/") or None

    # Percorso relativo scritto dalla radice del repo (`best_models/x.pkl`):
    # non ha lo slash iniziale, quindi il marcatore sopra non lo intercetta.
    prefisso = f"{BEST_MODELS_DIRNAME}/"
    if normalizzato.startswith(prefisso):
        return normalizzato[len(prefisso):].lstrip("/") or None
    return None


def is_archived_model_path(path: Optional[str]) -> bool:
    """True se `path` (in una qualunque delle forme gestite sopra -
    container, di un'altra macchina, o relativo dalla radice del repo)
    punta dentro `archivio/`.

    Un modello li' dentro resta CARICABILE (`resolve_model_path` lo trova
    comunque, la riorganizzazione non cancella nulla) ma e' quello che la
    procedura vecchia ha lasciato indietro - serve a `ModelRegistry.
    list_active_markets()` per smettere di offrirlo come mercato attivo
    in Dashboard senza doverlo rimuovere dal registry/storico."""
    relativo = relative_to_best_models(path) if path else None
    if relativo is None:
        return False
    return relativo == ARCHIVIO_DIRNAME or relativo.startswith(f"{ARCHIVIO_DIRNAME}/")


def to_container_path(path: str) -> str:
    """Percorso in forma container, PRESERVANDO le sottocartelle.

    `SaveLoad` registra il percorso assoluto della macchina su cui gira, ma
    l'app dell'operatore gira in Docker con `./best_models:/app/best_models`:
    una riga di registry con `C:\\Users\\...\\best_models\\x.pkl` da' "File
    modello non trovato" al primo utilizzo. Un percorso gia' in forma
    container resta identico.
    """
    if not path:
        return path
    if str(path).replace("\\", "/").startswith(CONTAINER_BEST_MODELS + "/"):
        return str(path).replace("\\", "/")

    relativo = relative_to_best_models(path)
    if relativo is None:
        # Non sta sotto best_models: non c'e' niente da riscrivere, e
        # inventare una destinazione sarebbe peggio del percorso originale.
        return path
    return f"{CONTAINER_BEST_MODELS}/{relativo}"


def _candidate_paths(path: str, root: str) -> Iterable[str]:
    """Percorsi locali da provare, dal piu' fedele al piu' tollerante."""
    yield path

    relativo = relative_to_best_models(path)
    if relativo is None:
        return

    # Stessa posizione relativa, radice locale: e' il caso normale dentro il
    # container e sulla macchina dell'operatore.
    yield os.path.join(root, *relativo.split("/"))

    # Il file e' stato spostato dalla riorganizzazione e una riga di registry
    # e' rimasta indietro. Si cerca lo stesso nome nelle posizioni previste
    # dal layout, senza mai scandire l'intero albero: un modello caricato da
    # una posizione inattesa e' meglio di una predizione mancante, ma solo se
    # resta un elenco chiuso e leggibile.
    nome = os.path.basename(relativo)
    yield os.path.join(root, nome)
    yield os.path.join(root, ARCHIVIO_DIRNAME, nome)
    for mercato in MERCATI_NUOVA_PROCEDURA:
        yield os.path.join(root, "under_over", mercato, nome)
        yield os.path.join(root, ARCHIVIO_DIRNAME, mercato, nome)
    for mercato in MERCATI_CARTELLA_EVENTO:
        yield os.path.join(root, mercato, nome)
        yield os.path.join(root, ARCHIVIO_DIRNAME, mercato, nome)


def resolve_model_path(path: Optional[str], root: Optional[str] = None) -> Optional[str]:
    """Percorso locale ESISTENTE per un modello, o `None`.

    Il valore registrato viene provato per primo e vince sempre quando esiste:
    il resolver non cambia mai il file servito finche' il registry e' allineato.
    Le alternative servono solo quando quel file non c'e' piu' - tipicamente
    una riga rimasta indietro dopo uno spostamento - e sostituiscono un "File
    modello non trovato" con il modello giusto trovato altrove.
    """
    if not path:
        return None

    radice = os.path.abspath(root) if root else best_models_root()
    visti: set[str] = set()
    for candidato in _candidate_paths(str(path), radice):
        if not candidato or candidato in visti:
            continue
        visti.add(candidato)
        if os.path.exists(candidato):
            return candidato
    return None
