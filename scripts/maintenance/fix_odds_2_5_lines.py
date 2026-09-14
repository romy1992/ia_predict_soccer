"""Riscrive a DB le quote Under/Over 2.5 di odds-api, prendendo la linea giusta.

IL PROBLEMA
`df_odds_service.py` costruiva le quote storiche da odds-api leggendo il
mercato `totals` con `search_price_name(out_totals, 'Over')`, che ritorna il
PRIMO esito di nome 'Over' qualunque sia la sua linea, e lo salvava sotto una
chiave `over_2.5_<bookmaker>` scritta a mano. Ma nel mercato `totals` ogni
bookmaker espone la PROPRIA linea principale, che non e' sempre 2.5.

Misurato sul payload grezzo (`dataset/odds/id_odds_h2h_totals.csv`):

    quote 'Over' salvate come "Over 2.5" : 77.316
      con point == 2.5  (corrette)       : 52.436   67,82%
      con point != 2.5  (SBAGLIATE)      : 24.880   32,18%
    fixture con almeno una linea errata  : 10.146   65,46%

Le linee sbagliate sono quasi tutte vicine (3.5, 2.75, 3.0, 2.25), quindi
nessun controllo sui valori anomali poteva accorgersene: un Over 3.5 a 2.40
sembra un Over 2.5 normalissimo. E l'overround restava coerente, perche' over
e under venivano dalla STESSA linea sbagliata. I casi eclatanti gia' notati
(un "Over 2.5" a 21.00) erano solo la coda: quelli sono linee tipo 6.5/7.5.

LA CORREZIONE
Il payload originale di odds-api e' ancora nel progetto, col campo `point`.
Si rilegge quello e si riscrive la chiave usando solo gli esiti a point 2.5.
Nessun dato viene inventato: se un bookmaker nel mercato `totals` non quotava
il 2.5, la sua quota sparisce invece di restare sbagliata.

COSTO MISURATO
    fixture con almeno una quota 2.5 vera : 14.046 / 15.499  (90,6%)
    fixture che restano senza quote       :  1.453           ( 9,4%)
    bookmaker per fixture: da 5,20 (sporchi) a 3,73 (corretti)

COSA TOCCA E COSA NO
Tocca SOLO le chiavi `over_2.5_*` / `under_2.5_*` (formato underscore-punto)
dentro la colonna JSON `under_over_2_5` delle righe Odds con
`odds_from='odds-api'`. Restano intatte:
  - `alternate_over_2_5_*` / `alternate_under_2_5_*`, che arrivano da
    `get_alternate_totals()` e il filtro sulla linea ce l'hanno gia';
  - `over 2.5_*` / `under 2.5_*` (con lo SPAZIO), che sono api-sports e
    passano da `map_odds()`, dove il confronto e' sulla stringa 'Over 2.5';
  - ogni altro mercato: h2h, goal/no goal, 1.5, 3.5, 4.5, corner, cartellini.
    Il blocco buggato scriveva solo quelle due chiavi.

Uso:
    python scripts/maintenance/fix_odds_2_5_lines.py              # simulazione
    python scripts/maintenance/fix_odds_2_5_lines.py --apply      # scrive
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CSV_GREZZO = os.path.join("src", "service_ia", "dataset", "odds", "id_odds_h2h_totals.csv")
BACKUP = os.path.join("scripts", "maintenance", "_backup_under_over_2_5.jsonl")
LINEA = 2.5
PREFISSI_DA_RIFARE = ("over_2.5_", "under_2.5_")
BATCH_COMMIT = 500  # una transazione sola con ~13k UPDATE fa cadere la connessione al Postgres remoto (Railway)


def carica_quote_corrette(percorso: str) -> tuple[dict[str, dict[str, float]], Counter]:
    """Rilegge il payload grezzo e tiene solo gli esiti a point 2.5.

    Ritorna {id_evento_odds_api: {chiave: quota}} piu' un contatore diagnostico.
    """
    csv.field_size_limit(10**9)
    corrette: dict[str, dict[str, float]] = {}
    diag = Counter()

    with open(percorso, newline="", encoding="utf-8") as handle:
        for riga in csv.DictReader(handle):
            id_evento = (riga.get("id") or "").strip()
            if not id_evento:
                continue
            try:
                bookmakers = ast.literal_eval(riga["bookmakers"])
            except (ValueError, SyntaxError):
                diag["righe_illeggibili"] += 1
                continue

            quote: dict[str, float] = {}
            for book in bookmakers:
                titolo = book.get("title")
                if not titolo:
                    continue
                for mercato in book.get("markets", []):
                    if mercato.get("key") != "totals":
                        continue
                    esiti = mercato.get("outcomes") or []
                    over = next(
                        (e for e in esiti if e.get("name") == "Over" and e.get("point") == LINEA), None
                    )
                    under = next(
                        (e for e in esiti if e.get("name") == "Under" and e.get("point") == LINEA), None
                    )
                    # Contabilita' di cosa si stava salvando prima, per il report.
                    primo_over = next((e for e in esiti if e.get("name") == "Over"), None)
                    if primo_over is not None:
                        diag["quote_lette_prima"] += 1
                        if primo_over.get("point") != LINEA:
                            diag["quote_di_linea_sbagliata"] += 1
                            diag[f"linea_{primo_over.get('point')}"] += 1
                    if over and over.get("price"):
                        quote[f"over_2.5_{titolo}"] = over["price"]
                    if under and under.get("price"):
                        quote[f"under_2.5_{titolo}"] = under["price"]

            corrette[id_evento] = quote
            diag["eventi_nel_csv"] += 1
            if not quote:
                diag["eventi_senza_nessuna_quota_2_5"] += 1

    return corrette, diag


def ricostruisci(bucket: dict, quote_corrette: dict[str, float]) -> tuple[dict, int, int]:
    """Toglie le chiavi buggate dal bucket e ci rimette quelle giuste.

    Non tocca nient'altro: `alternate_*` e le chiavi con lo spazio (api-sports)
    passano indenni.
    """
    tenute = {k: v for k, v in (bucket or {}).items() if not k.startswith(PREFISSI_DA_RIFARE)}
    rimosse = len(bucket or {}) - len(tenute)
    tenute.update(quote_corrette)
    return tenute, rimosse, len(quote_corrette)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="scrive davvero a DB (default: simulazione)")
    parser.add_argument("--limite", type=int, default=None, help="ferma dopo N partite, per provare")
    args = parser.parse_args()

    if not os.path.exists(CSV_GREZZO):
        print(f"Manca il payload grezzo: {CSV_GREZZO}")
        return 1

    print("Rilettura del payload grezzo di odds-api...")
    corrette, diag = carica_quote_corrette(CSV_GREZZO)
    lette = diag["quote_lette_prima"]
    sbagliate = diag["quote_di_linea_sbagliata"]
    print(f"  eventi nel CSV          : {diag['eventi_nel_csv']:,}")
    print(f"  quote Over lette prima  : {lette:,}")
    print(f"  di linea SBAGLIATA      : {sbagliate:,}  ({sbagliate/lette:.2%})" if lette else "")
    print(f"  eventi senza 2.5 vero   : {diag['eventi_senza_nessuna_quota_2_5']:,}")

    from sqlalchemy.orm import selectinload  # noqa: E402
    from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

    from src.repository.base.repository_db import SessionLocal  # noqa: E402
    from src.service_ia.model.match import Match  # noqa: E402

    sessione = SessionLocal()
    sessione.expire_on_commit = False  # i commit a lotti non devono invalidare gli oggetti gia' caricati
    try:
        partite = (
            sessione.query(Match)
            .options(selectinload(Match.odds))
            .filter(Match.id_events.isnot(None))
            .all()
        )
        print(f"\npartite con id_events a DB: {len(partite):,}")

        conta = Counter()
        backup_righe = []
        toccate = 0
        righe_da_commit = 0
        backup_handle = None
        if args.apply:
            os.makedirs(os.path.dirname(BACKUP), exist_ok=True)
            backup_handle = open(BACKUP, "w", encoding="utf-8")

        try:
            for partita in partite:
                if args.limite and toccate >= args.limite:
                    break
                quote = corrette.get(partita.id_events)
                if quote is None and partita.id_alternate_events:
                    quote = corrette.get(partita.id_alternate_events)
                if quote is None:
                    conta["partite_senza_payload"] += 1
                    continue

                for riga_odds in partita.odds or []:
                    if riga_odds.odds_from != "odds-api":
                        conta["righe_non_odds_api_saltate"] += 1
                        continue
                    bucket = riga_odds.under_over_2_5
                    if not isinstance(bucket, dict) or not bucket:
                        conta["righe_senza_bucket"] += 1
                        continue

                    nuovo, rimosse, riscritte = ricostruisci(bucket, quote)
                    if nuovo == bucket:
                        conta["righe_gia_corrette"] += 1
                        continue

                    riga_backup = {
                        "id_odds_fk": riga_odds.id_odds_fk,
                        "id_match": riga_odds.id_match,
                        "id_events": partita.id_events,
                        "prima": bucket,
                    }
                    backup_righe.append(riga_backup)
                    if backup_handle:
                        backup_handle.write(json.dumps(riga_backup, ensure_ascii=False) + "\n")
                    conta["chiavi_rimosse"] += rimosse
                    conta["chiavi_riscritte"] += riscritte
                    conta["righe_modificate"] += 1
                    if not nuovo:
                        conta["righe_rimaste_vuote"] += 1

                    if args.apply:
                        riga_odds.under_over_2_5 = nuovo
                        flag_modified(riga_odds, "under_over_2_5")
                        righe_da_commit += 1
                        if righe_da_commit % BATCH_COMMIT == 0:
                            sessione.commit()
                            print(
                                f"  commit parziale: {righe_da_commit:,} righe scritte "
                                f"({datetime.now(timezone.utc).isoformat()})",
                                flush=True,
                            )
                toccate += 1

            if args.apply and righe_da_commit % BATCH_COMMIT != 0:
                sessione.commit()
        finally:
            if backup_handle:
                backup_handle.close()

        print("\n" + "=" * 62)
        print("SIMULAZIONE" if not args.apply else "APPLICATO")
        print("=" * 62)
        for chiave in (
            "righe_modificate",
            "chiavi_rimosse",
            "chiavi_riscritte",
            "righe_rimaste_vuote",
            "righe_gia_corrette",
            "righe_senza_bucket",
            "partite_senza_payload",
            "righe_non_odds_api_saltate",
        ):
            print(f"  {chiave:28} {conta[chiave]:8,}")

        if args.apply:
            print(f"\nbackup del PRIMA salvato in {BACKUP} ({len(backup_righe):,} righe)")
            print(f"commit (a lotti da {BATCH_COMMIT}) completato alle {datetime.now(timezone.utc).isoformat()}")
        else:
            sessione.rollback()
            print("\nNiente scritto. Rilancia con --apply per applicare.")
        return 0
    finally:
        sessione.close()


if __name__ == "__main__":
    sys.exit(main())
