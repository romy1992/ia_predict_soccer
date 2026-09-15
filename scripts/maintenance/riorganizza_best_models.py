"""Da' una forma a `best_models/`: i modelli nuovi per mercato, il resto in archivio.

DA ESEGUIRE SULLA MACCHINA DELL'OPERATORE, dove sta il registry vero (quello
montato dai container con `./best_models:/app/best_models`). La sessione cloud
ha un `best_models/` inesistente: la cartella e' in `.gitignore` e non viaggia
per git, quindi qui non ci sarebbe niente da spostare.

DA COSA A COSA

    prima                              dopo
    best_models/                       best_models/
        under_over_1_5_champion_...        under_over/under_over_1_5/under_over_1_5_champion_...
        under_over_1_5_champion_cal...     under_over/under_over_1_5/under_over_1_5_champion_cal...
        under_over_2_5_champion_...        under_over/under_over_2_5/...
        under_over_3_5_champion_...        under_over/under_over_3_5/...
        under_over_4_5_champion.pkl        archivio/under_over_4_5_champion.pkl
        goal_no_goal_champion.pkl          archivio/goal_no_goal_champion.pkl
        corners_*_champion.pkl             archivio/corners_*_champion.pkl
        cards_*_champion.pkl               archivio/cards_*_champion.pkl
        ...                                archivio/...
        archivio/                          archivio/         (gia' li', invariato)
        registry/                          registry/         (mai toccata)

Sotto `under_over/<mercato>/` va SOLO il modello in produzione della procedura
nuova (quote separate per esito, 33 feature) con il suo calibratore: sono
quelli che l'app usa davvero. Tutto il resto e' della vecchia procedura e va
in `archivio/`, su richiesta esplicita dell'operatore: Under/Over 4.5,
goal_no_goal, corners, cards, gli ensemble, i campioni a 69 feature rimasti.

ARCHIVIARE NON E' RITIRARE
Under/Over 4.5, goal_no_goal, corners e cards restano in PRODUZIONE nel
registry: qui si sposta il file e si riscrive la riga che lo referenzia, non
si cambia lo stage. Continuano a servire predizioni esattamente come prima,
solo da un percorso diverso. Quando verranno rifatti con la procedura nuova
prenderanno anche loro una cartella dedicata.

SICUREZZA
- parte in SIMULAZIONE: senza --apply stampa il piano e non scrive niente;
- prima di toccare il registry ne fa una copia in `registry/_backup_<data>_riorganizzazione/`;
- non sovrascrive mai un file gia' presente a destinazione: se ne trova uno,
  si ferma prima di spostare qualunque cosa;
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
"""

from __future__ import annotations

import argparse
import copy
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
def pianifica(file_radice: Iterable[str], nuovi: Mapping[str, Iterable[str]]) -> list[Spostamento]:
    """Piano di spostamento a partire dai file nella radice di `best_models`.

    `nuovi` mappa mercato -> nomi dei file della procedura nuova (modello e
    calibratore in produzione). Quei file vanno nella cartella del mercato,
    tutti gli altri in `archivio/`. L'ordine dell'elenco in ingresso viene
    preservato, cosi' il piano stampato e' stabile fra una simulazione e
    l'esecuzione.
    """
    destinazione_per_nome: dict[str, str] = {}
    for mercato, nomi in nuovi.items():
        for nome in nomi:
            destinazione_per_nome[nome] = mercato

    piano: list[Spostamento] = []
    for nome in file_radice:
        if not nome.lower().endswith(ESTENSIONI_MODELLO):
            continue
        mercato = destinazione_per_nome.get(nome)
        if mercato:
            sotto = destination_subdir(mercato) or os.path.join("under_over", mercato)
            piano.append(
                Spostamento(
                    sorgente=nome,
                    destinazione=f"{sotto.replace(os.sep, '/')}/{nome}",
                    motivo=f"procedura nuova, {mercato}",
                )
            )
        else:
            piano.append(
                Spostamento(
                    sorgente=nome,
                    destinazione=f"{ARCHIVIO_DIRNAME}/{nome}",
                    motivo="vecchia procedura",
                )
            )
    return piano


def riscrivi_righe(righe: list[dict], piano: Iterable[Spostamento]) -> int:
    """Aggiorna i percorsi delle righe di registry secondo il piano.

    Modifica `righe` sul posto e ritorna quanti campi ha toccato. Tocca solo
    le righe che puntano a un file spostato e che oggi lo cercano nella radice
    di `best_models`: una riga che punta gia' a `archivio/` o a una cartella di
    mercato resta com'e'. Sono compresi i tre campi che contengono un percorso
    (`model_path`, `metadata_path`, `extra.calibration.calibrator_path`)
    perche' un calibratore lasciato indietro romperebbe il rollback tanto
    quanto un modello.
    """
    nuova_posizione = {s.sorgente: s.destinazione for s in piano}

    def riscritto(valore: Optional[str]) -> Optional[str]:
        relativo = relative_to_best_models(valore or "")
        if relativo is None or "/" in relativo:
            # Fuori da best_models, oppure gia' in una sottocartella: in
            # entrambi i casi non e' un file che questo piano sta spostando.
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
def file_nella_radice(radice: str) -> list[str]:
    if not os.path.isdir(radice):
        return []
    return sorted(
        nome
        for nome in os.listdir(radice)
        if os.path.isfile(os.path.join(radice, nome)) and nome.lower().endswith(ESTENSIONI_MODELLO)
    )


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


def collisioni(radice: str, piano: Iterable[Spostamento]) -> list[str]:
    """Destinazioni gia' occupate.

    Sovrascrivere sarebbe la fine del rollback: il file vecchio sparirebbe
    senza che nessuna riga di registry se ne accorga.
    """
    return [
        s.destinazione
        for s in piano
        if os.path.exists(os.path.join(radice, *s.destinazione.split("/")))
    ]


def sposta(radice: str, piano: Iterable[Spostamento], applica: bool) -> int:
    spostati = 0
    for s in piano:
        sorgente = os.path.join(radice, s.sorgente)
        destinazione = os.path.join(radice, *s.destinazione.split("/"))
        if not os.path.exists(sorgente):
            print(f"   {s.sorgente}: gia' assente, salto")
            continue
        if applica:
            os.makedirs(os.path.dirname(destinazione), exist_ok=True)
            shutil.move(sorgente, destinazione)
        spostati += 1
    return spostati


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

    radice_file = file_nella_radice(radice)
    piano = pianifica(radice_file, nuovi)
    if not piano:
        print("\nLa radice di best_models non contiene modelli da spostare: gia' riorganizzata.")
        return 0

    print(f"\npiano: {len(piano)} file da spostare")
    print(f"  {'file':52} {'destinazione':62} motivo")
    print("  " + "-" * 134)
    for s in piano:
        print(f"  {s.sorgente[:52]:52} {s.destinazione[:62]:62} {s.motivo}")

    occupate = collisioni(radice, piano)
    if occupate:
        print(f"\nMi fermo: {len(occupate)} destinazioni sono gia' occupate e non sovrascrivo mai.")
        for destinazione in occupate[:10]:
            print(f"   {destinazione}")
        return 1

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
