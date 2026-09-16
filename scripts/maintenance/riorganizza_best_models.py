"""Da' una forma a `best_models/`: i modelli nuovi per mercato, il resto in archivio.

DA ESEGUIRE SULLA MACCHINA DELL'OPERATORE, dove sta il registry vero (quello
montato dai container con `./best_models:/app/best_models`). La sessione cloud
ha un `best_models/` inesistente: la cartella e' in `.gitignore` e non viaggia
per git, quindi qui non ci sarebbe niente da spostare.

DA COSA A COSA

    prima                              dopo
    best_models/                       best_models/
        under_over_1_5_champion_...        under_over/under_over_1_5/under_over_1_5_champion_...
        under_over_2_5_champion_...        under_over/under_over_2_5/...
        under_over_3_5_champion_...        under_over/under_over_3_5/...
        under_over_4_5_champion.pkl        archivio/under_over_4_5/under_over_4_5_champion.pkl
        goal_no_goal_champion.pkl          archivio/goal_no_goal/goal_no_goal_champion.pkl
        cards_champion.pkl                 archivio/cards/cards_champion.pkl
        corners_champion.pkl               archivio/corners/corners_champion.pkl
        h2h_champion.pkl                   archivio/h2h/h2h_champion.pkl
        cards_models/*.pkl                 archivio/cards/*.pkl
        corners_models/*.pkl               archivio/corners/*.pkl
        archivio/*.pkl                     archivio/<mercato>/*.pkl
        registry/                          registry/         (mai toccata)

Sotto `under_over/<mercato>/` va SOLO il modello in produzione della procedura
nuova (quote separate per esito, 33 feature) con il suo calibratore: sono
quelli che l'app usa davvero. Tutto il resto e' della vecchia procedura e va in
`archivio/<mercato>/`, con la stessa separazione per mercato: una cartella di
conservazione piatta con dentro decine di .pkl di mercati diversi e'
esattamente il problema da cui si parte.

ARCHIVIARE NON E' RITIRARE
Under/Over 4.5, goal_no_goal, corners e cards restano in PRODUZIONE nel
registry: qui si sposta il file e si riscrive la riga che lo referenzia, non
si cambia lo stage. Continuano a servire predizioni esattamente come prima,
solo da un percorso diverso. Quando verranno rifatti con la procedura nuova
prenderanno anche loro una cartella dedicata.

SICUREZZA
- parte in SIMULAZIONE: senza --apply stampa il piano e non scrive niente;
- prima di toccare il registry ne fa una copia in `registry/_backup_<data>_riorganizzazione/`;
- non sovrascrive mai un file gia' presente a destinazione: si ferma prima di
  spostare qualunque cosa, dopo aver detto se i due file sono lo stesso modello
  copiato due volte o due modelli diversi con lo stesso nome; con
  --risolvi-collisioni archivia quello della radice con un nome distinto
  (nome__<data di modifica>.pkl) invece di fermarsi;
- non cancella mai niente, solo sposta;
- sposta il file e riscrive la riga di registry insieme, mai una sola delle
  due (un file spostato con la riga vecchia da' "File modello non trovato");
- riscrive TUTTE le righe che puntano a un file spostato, non solo la prima:
  fino ai nomi con suffisso data piu' run condividevano lo stesso nome file;
- alla fine verifica che ogni riga punti a un file esistente e che i tre
  modelli in produzione si carichino davvero.

Uso:
    python scripts/maintenance/riorganizza_best_models.py            # simulazione
    python scripts/maintenance/riorganizza_best_models.py --apply    # esegue
    python scripts/maintenance/riorganizza_best_models.py --apply --risolvi-collisioni
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.model_paths import (  # noqa: E402
    ARCHIVIO_DIRNAME,
    CONTAINER_BEST_MODELS,
    MERCATI_NUOVA_PROCEDURA,
    destination_subdir,
    relative_to_best_models,
    to_container_path,
)

BEST = "best_models"

# Numero di feature della procedura nuova (quote separate per esito). Stesso
# criterio gia' usato da `promuovi_e_archivia.py` per decidere se un mercato e'
# gia' a posto: un modello a 69 feature e' della vecchia procedura, comunque si
# chiami il file.
FEATURE_NUOVA_PROCEDURA = 33

# Solo i file con queste estensioni vengono spostati: qualunque altra cosa
# nella radice (note, export, residui) resta dov'e', perche' spostare un file
# che non si e' capito cos'e' e' peggio che lasciarlo.
ESTENSIONI_MODELLO = (".pkl", ".joblib")


@dataclass(frozen=True)
class Spostamento:
    """Un file da spostare, con la posizione di partenza e di arrivo
    espresse come percorsi RELATIVI a `best_models`."""

    sorgente: str
    destinazione: str
    motivo: str


# ----------------------------------------------------------------------
# Pianificazione: funzioni pure, cosi' il piano si puo' verificare senza
# toccare il disco (e i test non hanno bisogno di un best_models vero).
# ----------------------------------------------------------------------
def mercato_di(relativo: str, mercati: Iterable[str]) -> str:
    """Mercato a cui appartiene un file, dal suo nome o dalla cartella.

    Si sceglie il prefisso PIU' LUNGO fra i mercati noti, altrimenti
    `under_over_1_5_champion.pkl` finirebbe sotto un ipotetico `under_over`
    invece che sotto la sua linea. Le cartelle gia' per mercato
    (`cards_models/`, `corners_models/`) valgono come indicazione quando il
    nome file non basta. Se non si riconosce niente, `altri`: meglio una
    cartella onesta che un mercato indovinato.
    """
    cartella, _, nome = relativo.rpartition("/")
    candidati = [m for m in mercati if nome.startswith(f"{m}_") or nome == m]
    if candidati:
        return max(candidati, key=len)
    ultima = cartella.rsplit("/", 1)[-1]
    if ultima.endswith("_models"):
        return ultima[: -len("_models")]

    # Ultima risorsa, e quella che copre i mercati non elencati nel registry:
    # il nome file e' sempre `<mercato>_champion...`, quindi il mercato e' cio'
    # che sta prima. Meglio di `altri` per un file perfettamente riconoscibile.
    prima_di_champion = os.path.splitext(nome)[0].split("_champion")[0]
    return prima_di_champion or "altri"


def pianifica(
    file_da_sistemare: Iterable[str],
    nuovi: Mapping[str, Iterable[str]],
    mercati: Iterable[str] = (),
) -> list[Spostamento]:
    """Piano di spostamento, a partire da percorsi RELATIVI a `best_models`.

    `nuovi` mappa mercato -> nomi dei file della procedura nuova (modello e
    calibratore in produzione): quelli vanno in `under_over/<mercato>/`. Tutto
    il resto va in `archivio/<mercato>/`, con la stessa separazione per mercato
    dei modelli in produzione - una cartella di conservazione piatta con dentro
    decine di `.pkl` di mercati diversi e' esattamente il problema da cui si
    parte. L'ordine dell'elenco in ingresso viene preservato, cosi' il piano
    stampato e' stabile fra una simulazione e l'esecuzione.
    """
    mercato_nuovo_per_nome: dict[str, str] = {}
    for mercato, nomi in nuovi.items():
        for nome in nomi:
            mercato_nuovo_per_nome[nome] = mercato

    noti = sorted(set(mercati) | set(MERCATI_NUOVA_PROCEDURA))
    piano: list[Spostamento] = []
    for relativo in file_da_sistemare:
        if not relativo.lower().endswith(ESTENSIONI_MODELLO):
            continue
        nome = relativo.rpartition("/")[2]
        mercato = mercato_nuovo_per_nome.get(relativo) or mercato_nuovo_per_nome.get(nome)
        if mercato:
            sotto = destination_subdir(mercato).replace(os.sep, "/")
            destinazione, motivo = f"{sotto}/{nome}", f"procedura nuova, {mercato}"
        else:
            mercato = mercato_di(relativo, noti)
            destinazione = f"{ARCHIVIO_DIRNAME}/{mercato}/{nome}"
            motivo = f"vecchia procedura, {mercato}"
        if destinazione != relativo:
            piano.append(Spostamento(sorgente=relativo, destinazione=destinazione, motivo=motivo))
    return piano


def riscrivi_righe(righe: list[dict], piano: Iterable[Spostamento]) -> int:
    """Aggiorna i percorsi delle righe di registry secondo il piano.

    Modifica `righe` sul posto e ritorna quanti campi ha toccato. La chiave e'
    il percorso RELATIVO completo, non il nome file: due file omonimi in
    cartelle diverse sono due file diversi, e una riga che punta a un file che
    il piano non muove resta com'e'. Sono compresi i tre campi che contengono un percorso
    (`model_path`, `metadata_path`, `extra.calibration.calibrator_path`)
    perche' un calibratore lasciato indietro romperebbe il rollback tanto
    quanto un modello.
    """
    nuova_posizione = {s.sorgente: s.destinazione for s in piano}

    def riscritto(valore: Optional[str]) -> Optional[str]:
        relativo = relative_to_best_models(valore or "")
        if relativo is None:
            return None
        destinazione = nuova_posizione.get(relativo)
        if destinazione is None:
            return None
        return f"{CONTAINER_BEST_MODELS}/{destinazione}"

    toccati = 0
    for riga in righe:
        for campo in ("model_path", "metadata_path"):
            nuovo = riscritto(riga.get(campo))
            if nuovo and nuovo != riga.get(campo):
                riga[campo] = nuovo
                toccati += 1
        calibrazione = (riga.get("extra") or {}).get("calibration") or {}
        nuovo = riscritto(calibrazione.get("calibrator_path"))
        if nuovo and nuovo != calibrazione.get("calibrator_path"):
            calibrazione["calibrator_path"] = nuovo
            toccati += 1
    return toccati


def normalizza_in_forma_container(righe: list[dict]) -> int:
    """Porta in forma container i percorsi rimasti assoluti di una macchina.

    Il 2026-09-14 il bridge aveva corretto a mano solo `model_path`: i campi
    `calibrator_path` e `metadata_path` delle tre righe promosse erano rimasti
    con il percorso Windows del worktree di lavoro. Non li usa la serving
    logic, ma un percorso che esiste solo su una macchina che non e' quella
    dove gira l'app e' una bugia scritta nel registry.
    """
    toccati = 0
    for riga in righe:
        for campo in ("model_path", "metadata_path"):
            valore = riga.get(campo) or ""
            nuovo = to_container_path(valore)
            if valore and nuovo != valore:
                riga[campo] = nuovo
                toccati += 1
        calibrazione = (riga.get("extra") or {}).get("calibration") or {}
        valore = calibrazione.get("calibrator_path") or ""
        nuovo = to_container_path(valore)
        if valore and nuovo != valore:
            calibrazione["calibrator_path"] = nuovo
            toccati += 1
    return toccati


# ----------------------------------------------------------------------
# Lettura dello stato reale
# ----------------------------------------------------------------------
def file_da_sistemare(radice: str) -> list[str]:
    """Percorsi relativi dei modelli non ancora al loro posto.

    Tre origini: la radice di `best_models`, i file sfusi in `archivio/` (che
    la prima versione aveva lasciato piatta) e le cartelle `<mercato>_models/`.
    Chi e' gia' sotto `under_over/<mercato>/` o `archivio/<mercato>/` non
    compare: il piano non produce spostamenti a vuoto.
    """
    if not os.path.isdir(radice):
        return []

    def modelli_in(cartella: str) -> list[str]:
        pieno = os.path.join(radice, *cartella.split("/")) if cartella else radice
        if not os.path.isdir(pieno):
            return []
        return sorted(
            f"{cartella}/{nome}" if cartella else nome
            for nome in os.listdir(pieno)
            if os.path.isfile(os.path.join(pieno, nome)) and nome.lower().endswith(ESTENSIONI_MODELLO)
        )

    sottocartelle = sorted(
        nome
        for nome in os.listdir(radice)
        if os.path.isdir(os.path.join(radice, nome)) and nome.endswith("_models")
    )
    elenco = modelli_in("")
    elenco += modelli_in(ARCHIVIO_DIRNAME)
    for cartella in sottocartelle:
        elenco += modelli_in(cartella)
    return elenco


def modelli_nuovi(registry) -> dict[str, list[str]]:
    """Nomi dei file in produzione per i mercati della procedura nuova.

    Un mercato che NON e' ancora stato rifatto (production a 69 feature, o
    nessuna production) non compare: il suo modello e' della vecchia procedura
    e finisce in archivio come tutti gli altri, invece di prendersi una
    cartella dedicata che non gli spetta.
    """
    esito: dict[str, list[str]] = {}
    for mercato in MERCATI_NUOVA_PROCEDURA:
        produzione = registry.get_production(market=mercato)
        if not produzione:
            print(f"   {mercato}: nessuna production, lo tratto come vecchia procedura")
            continue
        n_feature = len(produzione.get("feature_names") or [])
        if n_feature != FEATURE_NUOVA_PROCEDURA:
            print(f"   {mercato}: production a {n_feature} feature, non e' la procedura nuova")
            continue

        nomi = []
        for percorso in (
            produzione.get("model_path"),
            ((produzione.get("extra") or {}).get("calibration") or {}).get("calibrator_path"),
        ):
            relativo = relative_to_best_models(percorso or "")
            if relativo and "/" not in relativo:
                nomi.append(relativo)
        esito[mercato] = nomi
        print(f"   {mercato}: {produzione['run_id'][-22:]}, {n_feature} feature -> {', '.join(nomi) or '(gia a posto)'}")
    return esito


def leggi_righe(index: str) -> list[dict]:
    with open(index, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f if riga.strip()]


def scrivi_righe(index: str, righe: list[dict]) -> None:
    with open(index, "w", encoding="utf-8") as f:
        for riga in righe:
            f.write(json.dumps(riga, ensure_ascii=False) + "\n")


def backup_registry(registry_dir: str, applica: bool) -> str:
    dest = os.path.join(registry_dir, f"_backup_{date.today():%Y%m%d}_riorganizzazione")
    print(f"\nbackup del registry in {dest}")
    if not applica:
        print("   (simulazione: non copiato)")
        return dest
    os.makedirs(dest, exist_ok=True)
    for nome in ("index.jsonl", "promotion_history.jsonl"):
        src = os.path.join(registry_dir, nome)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(dest, nome))
        n_src = sum(1 for _ in open(src, encoding="utf-8"))
        n_dst = sum(1 for _ in open(os.path.join(dest, nome), encoding="utf-8"))
        if n_src != n_dst:
            raise RuntimeError(f"backup incompleto di {nome}: {n_src} righe contro {n_dst}")
        print(f"   {nome}: {n_src} righe")
    return dest


def collisioni(radice: str, piano: Iterable[Spostamento]) -> list[Spostamento]:
    """Spostamenti la cui destinazione e' gia' occupata.

    Sovrascrivere sarebbe la fine del rollback: il file vecchio sparirebbe
    senza che nessuna riga di registry se ne accorga.
    """
    return [
        s
        for s in piano
        if os.path.exists(os.path.join(radice, *s.destinazione.split("/")))
    ]


def _impronta(percorso: str) -> str:
    digest = hashlib.sha256()
    with open(percorso, "rb") as f:
        for blocco in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(blocco)
    return digest.hexdigest()


def descrivi_collisione(radice: str, s: Spostamento) -> str:
    """Le due righe che servono per decidere: sono lo stesso file o no.

    Due file con lo stesso nome possono essere lo stesso modello copiato due
    volte (e allora la copia nella radice e' un residuo) oppure due modelli
    diversi che si contendono un nome, com'era prima dei nomi con suffisso
    data. Le due cose si risolvono in modo opposto, quindi lo script non
    indovina: mostra il confronto e lascia decidere.
    """
    sorgente = os.path.join(radice, *s.sorgente.split("/"))
    destinazione = os.path.join(radice, *s.destinazione.split("/"))

    def scheda(percorso: str) -> str:
        info = os.stat(percorso)
        quando = dt.datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"{info.st_size:>12,} byte   modificato {quando}"

    uguali = (
        os.path.getsize(sorgente) == os.path.getsize(destinazione)
        and _impronta(sorgente) == _impronta(destinazione)
    )
    verdetto = (
        "IDENTICI: quello nella radice e' una copia di quello gia' in archivio"
        if uguali
        else "DIVERSI: sono due modelli distinti che condividono il nome"
    )
    return (
        f"   {s.sorgente}\n"
        f"      nella radice : {scheda(sorgente)}\n"
        f"      in archivio  : {scheda(destinazione)}\n"
        f"      -> {verdetto}"
    )


def disambigua(radice: str, piano: Iterable[Spostamento]) -> list[Spostamento]:
    """Piano in cui le destinazioni occupate prendono un nome distinto.

    Al nome si aggiunge la data di modifica del file, che e' l'unica cosa che
    davvero lo distingue dall'omonimo gia' archiviato. Cosi' nessuno dei due
    file sparisce e le righe di registry restano separabili: quella che
    puntava alla radice segue il file rinominato, quella che puntava ad
    archivio non si muove.
    """
    risolto: list[Spostamento] = []
    for s in piano:
        destinazione = s.destinazione
        if not os.path.exists(os.path.join(radice, *destinazione.split("/"))):
            risolto.append(s)
            continue

        sorgente = os.path.join(radice, *s.sorgente.split("/"))
        quando = dt.datetime.fromtimestamp(os.stat(sorgente).st_mtime).strftime("%Y%m%dT%H%M%S")
        cartella, nome = destinazione.rsplit("/", 1) if "/" in destinazione else ("", destinazione)
        radice_nome, estensione = os.path.splitext(nome)
        candidato = f"{radice_nome}__{quando}{estensione}"
        contatore = 2
        while os.path.exists(os.path.join(radice, *(f"{cartella}/{candidato}").split("/"))):
            candidato = f"{radice_nome}__{quando}_{contatore}{estensione}"
            contatore += 1
        risolto.append(
            Spostamento(
                sorgente=s.sorgente,
                destinazione=f"{cartella}/{candidato}" if cartella else candidato,
                motivo=f"{s.motivo} (nome gia' occupato, rinominato)",
            )
        )
    return risolto


def sposta(radice: str, piano: Iterable[Spostamento], applica: bool) -> int:
    spostati = 0
    for s in piano:
        sorgente = os.path.join(radice, *s.sorgente.split("/"))
        destinazione = os.path.join(radice, *s.destinazione.split("/"))
        if not os.path.exists(sorgente):
            print(f"   {s.sorgente}: gia' assente, salto")
            continue
        if applica:
            os.makedirs(os.path.dirname(destinazione), exist_ok=True)
            shutil.move(sorgente, destinazione)
        spostati += 1
    return spostati


def togli_cartelle_vuote(radice: str, applica: bool) -> list[str]:
    """Le `<mercato>_models/` svuotate dallo spostamento non servono piu'."""
    tolte = []
    for nome in sorted(os.listdir(radice)) if os.path.isdir(radice) else []:
        pieno = os.path.join(radice, nome)
        if nome.endswith("_models") and os.path.isdir(pieno) and not os.listdir(pieno):
            if applica:
                os.rmdir(pieno)
            tolte.append(nome)
    return tolte


def verifica(radice: str, index: str) -> int:
    print("\nverifica finale")
    rotte = []
    for riga in leggi_righe(index):
        percorso = riga.get("model_path") or ""
        locale = percorso.replace(CONTAINER_BEST_MODELS, radice) if percorso.startswith(CONTAINER_BEST_MODELS) else percorso
        if locale and not os.path.exists(locale):
            rotte.append((riga.get("run_id", "?"), percorso))
    print(f"   righe che puntano a un file inesistente: {len(rotte)}")
    for run_id, percorso in rotte[:10]:
        print(f"      {run_id[-22:]}  ->  {percorso}")
    return len(rotte)


def carica_i_nuovi(radice: str, registry) -> int:
    """Carica davvero i modelli in produzione e ritorna quanti non si aprono.

    Una riga che punta a un file esistente non basta a dire che e' andata
    bene: il file potrebbe essere stato spostato a meta', o non essere il
    modello che dice di essere. Un modello che non si carica e' un mercato
    senza predizioni, quindi il conteggio decide l'esito dello script.
    """
    import joblib

    print("\n   caricamento di prova dei modelli in produzione della procedura nuova")
    falliti = 0
    for mercato in MERCATI_NUOVA_PROCEDURA:
        produzione = registry.get_production(market=mercato)
        if not produzione:
            print(f"      {mercato}: nessuna production")
            continue
        percorso = (produzione.get("model_path") or "").replace(CONTAINER_BEST_MODELS, radice)
        try:
            modello = joblib.load(percorso)
            print(f"      {mercato}: {os.path.relpath(percorso, radice)}  OK ({type(modello).__name__})")
        except Exception as exc:  # noqa: BLE001 - qualunque errore qui va mostrato, non nascosto
            print(f"      {mercato}: NON CARICABILE -> {exc}")
            falliti += 1
    return falliti


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="esegue davvero (default: simulazione)")
    parser.add_argument("--radice", default=BEST, help=f"cartella dei modelli (default: {BEST})")
    parser.add_argument(
        "--risolvi-collisioni",
        action="store_true",
        help="quando in archivio esiste gia' un file con lo stesso nome, archivia quello "
             "della radice con un nome distinto invece di fermarsi (non cancella mai niente)",
    )
    args = parser.parse_args()

    radice = os.path.abspath(args.radice)
    registry_dir = os.path.join(radice, "registry")
    index = os.path.join(registry_dir, "index.jsonl")
    if not os.path.exists(index):
        print(f"Registry non trovato: {index}")
        print("Questo script va eseguito sulla macchina dove sta il registry vero.")
        return 1

    from src.service_ia.training.model_registry import ModelRegistry

    registry = ModelRegistry(registry_dir=registry_dir)

    print("modelli in produzione della procedura nuova")
    nuovi = modelli_nuovi(registry)

    piano = pianifica(file_da_sistemare(radice), nuovi, mercati=registry.list_markets())
    if not piano:
        print("\nNessun modello fuori posto: best_models e' gia' riorganizzata.")
        return 0

    print(f"\npiano: {len(piano)} file da spostare")
    print(f"  {'file':52} {'destinazione':62} motivo")
    print("  " + "-" * 134)
    for s in piano:
        print(f"  {s.sorgente[:52]:52} {s.destinazione[:62]:62} {s.motivo}")

    occupate = collisioni(radice, piano)
    if occupate:
        print(f"\n{len(occupate)} destinazioni sono gia' occupate. Non sovrascrivo mai: ecco cosa sono.")
        for s in occupate:
            print(descrivi_collisione(radice, s))
        if not args.risolvi_collisioni:
            print("\nMi fermo. Rilancia con --risolvi-collisioni per archiviarli con un nome")
            print("distinto (nome__<data di modifica>.pkl): nessuno dei due file va perso e le")
            print("righe di registry restano separate. Se invece il file nella radice e' un")
            print("residuo da buttare, cancellalo tu e rilancia senza il flag.")
            return 1
        piano = disambigua(radice, piano)
        print("\npiano aggiornato per le collisioni:")
        for s in piano:
            if "rinominato" in s.motivo:
                print(f"  {s.sorgente[:52]:52} -> {s.destinazione}")

    righe = leggi_righe(index)
    # Copia PROFONDA: `extra.calibration` e' annidato, e una copia
    # superficiale lo condividerebbe con le righe vere, riscrivendole gia'
    # durante l'anteprima della simulazione.
    da_riscrivere = riscrivi_righe(copy.deepcopy(righe), piano)
    print(f"\nrighe di registry da aggiornare: {da_riscrivere} campi su {len(righe)} righe")

    if not args.apply:
        print("\nSimulazione conclusa. Rilancia con --apply per eseguire.")
        return 0

    backup_registry(registry_dir, applica=True)
    spostati = sposta(radice, piano, applica=True)
    print(f"\nfile spostati: {spostati}")

    toccati = riscrivi_righe(righe, piano)
    toccati += normalizza_in_forma_container(righe)
    scrivi_righe(index, righe)
    print(f"campi di registry riscritti: {toccati}   ({index} aggiornato)")

    vuote = togli_cartelle_vuote(radice, applica=True)
    if vuote:
        print(f"cartelle vuote rimosse: {', '.join(vuote)}")

    rotte = verifica(radice, index)
    falliti = carica_i_nuovi(radice, ModelRegistry(registry_dir=registry_dir))
    if rotte or falliti:
        print("\nATTENZIONE: la verifica non e' pulita, controlla prima di riavviare i container.")
        print(f"   righe rotte: {rotte}   modelli che non si caricano: {falliti}")
        print(f"   il registry di partenza e' in {os.path.join(registry_dir, f'_backup_{date.today():%Y%m%d}_riorganizzazione')}")
        return 1

    print("\nFatto. Riavvia i container api e scheduler perche' rileggano il registry:")
    print("   docker restart soccer_api soccer_scheduler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
