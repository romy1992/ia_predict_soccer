"""Fonde le righe `match` duplicate sullo stesso `id_fixture`, tenendo i dati.

IL PROBLEMA
`match.id_fixture` non ha vincolo di unicita' (`Match.id_fixture` in
`src/service_ia/model/match.py`: `unique=True` era commentato), e la chiave
primaria e' un `uuid4()` generato lato Python. `download_import_matches`
cerca la riga esistente con `filter_by({'id_fixture': ...}).first()` e, se
non la trova, crea un `Match` nuovo con un uuid nuovo - ma le righe nuove
restano bufferizzate in `list_matches` e vengono scritte con `save_all`
SOLO alla fine del job. Due esecuzioni sovrapposte sulla stessa finestra di
date (misurato 2026-09-15: `daily_refresh` schedulato 06:21:20->06:30:02 e
una seconda catena 06:23:01->06:31:27, processi `scheduler` e `api`
distinti, nessun lock condiviso) non vedono quindi mai gli insert l'una
dell'altra: entrambe concludono "questa fixture non c'e'" e inseriscono.

Il risultato sono due righe per la stessa partita. Gli aggiornamenti
successivi passano da `.first()`, che senza `order_by` ne prende una
arbitraria: l'altra resta congelata allo stato del momento in cui e' nata,
tipicamente `NS` senza punteggio. E una riga `NS` con calcio d'inizio nel
passato viene classificata "live" da `DashboardService._classify_phase`,
per cui la partita risulta "In diretta" per sempre (caso reale: fixture
1550118 Como-Parma del 2026-09-14, una riga FT 2-1 e una riga NS vuota).

MISURATO SUL DB (2026-09-15)
    id_fixture duplicati                       : 116  (232 righe, 116 di troppo)
    gruppi con entrambe le righe che hanno figli:  63
    gruppi con una sola riga con figli         :  33
    gruppi con nessuna riga con figli          :  20
    finestra delle date coinvolte              : 2026-09-14 .. 2026-09-19
                                                 (la finestra di `future_sync`)

COSA FA
Per ogni `id_fixture` duplicato elegge una riga VINCENTE e ci fa convergere
i dati delle altre, invece di buttarle:

  1. vincente = la riga con piu' dati, in quest'ordine di preferenza:
     stato finale (FT/AET/PEN/ABD/CANC/PST/WO) > punteggio valorizzato >
     piu' `odds_snapshot` > piu' `statistics` > piu' `odds` >
     `mean_statistics` presente > `id_match_fk` piu' basso (solo per
     rendere la scelta deterministica a parita' di tutto).
  2. `odds_snapshot`: le righe delle perdenti vengono SEMPRE ri-agganciate
     alla vincente. E' un log append-only e la PK `id_snapshot` e' un hash
     del contenuto (fixture+bookmaker+mercato+esito+captured_at), quindi
     ri-agganciare non puo' generare conflitti di PK e non duplica nulla -
     recupera invece lo storico quote che era finito spezzato tra le due
     righe (es. fixture 1550120: 1039 + 1178 snapshot da riunire).
  3. `statistics` / `odds`: se la vincente NON ne ha e una perdente si',
     vengono ri-agganciate; altrimenti le righe della perdente vengono
     cancellate (ri-agganciarle darebbe 4 righe `statistics` per una
     partita da 2 squadre, che romperebbe `_resolve_team_stats`).
  4. la riga `match` perdente viene cancellata.

COSA NON TOCCA
Nessuna riga `match` non duplicata. Nessuna colonna della riga vincente:
non riscrive stato/punteggio/medie, sposta solo i figli. Nessuna tabella
che non sia `match`/`statistics`/`odds`/`odds_snapshot` (sono le uniche tre
FK verso `match.id_match_fk`, verificato su information_schema).
`prediction_ledger`/`betting_slips`/`match_prediction_snapshot` referenziano
`fixture_id` (int) e non `id_match_fk`, quindi la fusione li rende anzi
coerenti: prima puntavano a una fixture con due righe.

Va eseguito PRIMA della migration `f1a2b3c4d5e6` che aggiunge
`uq_match_id_fixture`: con i duplicati ancora a DB quella migration
fallisce (ed e' giusto che fallisca, invece di scartare righe in silenzio).

Uso:
    python scripts/maintenance/dedup_match_id_fixture.py            # simulazione
    python scripts/maintenance/dedup_match_id_fixture.py --apply    # scrive
    python scripts/maintenance/dedup_match_id_fixture.py --verbose  # riga per riga
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from src.repository.base.repository_db import SessionLocal  # noqa: E402
from src.service_ia.config.app_config import load_app_config  # noqa: E402

BACKUP = os.path.join("scripts", "maintenance", "_backup_dedup_match_id_fixture.jsonl")

# Stessi stati considerati "finali" da `DashboardService.FINAL_STATUSES`: una
# riga in uno di questi stati e' per definizione piu' avanzata di una NS.
STATI_FINALI = ("FT", "AET", "PEN", "ABD", "CANC", "PST", "WO")


def carica_gruppi(session) -> dict[int, list[dict]]:
    """Legge tutte le righe `match` con un `id_fixture` duplicato, con il
    conteggio dei figli per ciascuna (una query sola, non una per riga)."""
    righe = session.execute(
        text(
            """
            with dup as (
                select id_fixture from match
                where id_fixture is not null
                group by id_fixture having count(*) > 1
            )
            select
                m.id_fixture,
                m.id_match_fk,
                m.status,
                m.score_home,
                m.score_away,
                m.date_match,
                m.name_home,
                m.name_away,
                m.mean_statistics is not null as ha_medie,
                (select count(*) from odds_snapshot o where o.id_match = m.id_match_fk) as n_snap,
                (select count(*) from statistics st where st.id_match = m.id_match_fk) as n_stat,
                (select count(*) from odds od where od.id_match = m.id_match_fk) as n_odds
            from match m
            join dup on dup.id_fixture = m.id_fixture
            order by m.id_fixture, m.id_match_fk
            """
        )
    ).mappings()

    gruppi: dict[int, list[dict]] = {}
    for riga in righe:
        gruppi.setdefault(riga["id_fixture"], []).append(dict(riga))
    return gruppi


def punteggio_completezza(riga: dict) -> tuple:
    """Chiave di ordinamento: piu' alta = riga piu' completa, quindi
    vincente. L'ultimo elemento e' negato perche' a parita' di tutto il
    resto vince l'`id_match_fk` lessicograficamente piu' basso (scelta
    arbitraria ma DETERMINISTICA: rilanciare lo script sullo stesso DB
    deve eleggere sempre la stessa riga)."""
    return (
        1 if (riga["status"] or "").upper() in STATI_FINALI else 0,
        1 if riga["score_home"] is not None else 0,
        riga["n_snap"],
        riga["n_stat"],
        riga["n_odds"],
        1 if riga["ha_medie"] else 0,
        # str() perche' l'id e' una stringa uuid: confronto lessicografico
        _negativo_lessicografico(riga["id_match_fk"]),
    )


def _negativo_lessicografico(valore: str) -> tuple:
    """Inverte l'ordine lessicografico di una stringa usandola come tupla di
    codepoint negati - serve solo per far vincere l'id piu' BASSO dentro una
    chiave di `max()`, senza dover scrivere un comparatore custom."""
    return tuple(-ord(c) for c in str(valore or ""))


def pianifica_gruppo(righe: list[dict]) -> dict:
    """Decide vincente/perdenti e cosa fare dei figli di ciascuna perdente.
    Funzione pura: non tocca il DB, cosi' la simulazione e l'applicazione
    percorrono ESATTAMENTE la stessa logica."""
    vincente = max(righe, key=punteggio_completezza)
    perdenti = [r for r in righe if r["id_match_fk"] != vincente["id_match_fk"]]

    azioni = []
    for perdente in perdenti:
        azione = {
            "id_match_fk": perdente["id_match_fk"],
            "status": perdente["status"],
            # gli snapshot si ri-agganciano sempre: log append-only, PK a hash
            "snapshot_da_riagganciare": perdente["n_snap"],
            # statistics/odds si ri-agganciano SOLO se la vincente non ne ha
            "statistics_da_riagganciare": perdente["n_stat"] if vincente["n_stat"] == 0 else 0,
            "statistics_da_cancellare": 0 if vincente["n_stat"] == 0 else perdente["n_stat"],
            "odds_da_riagganciare": perdente["n_odds"] if vincente["n_odds"] == 0 else 0,
            "odds_da_cancellare": 0 if vincente["n_odds"] == 0 else perdente["n_odds"],
        }
        azioni.append(azione)

    return {"vincente": vincente, "azioni": azioni}


def applica_gruppo(session, piano: dict) -> None:
    """Esegue il piano di UN gruppo. Il commit e' del chiamante (una
    transazione per gruppo: se qualcosa va storto a metA', non resta un
    gruppo mezzo fuso)."""
    id_vincente = piano["vincente"]["id_match_fk"]

    for azione in piano["azioni"]:
        id_perdente = azione["id_match_fk"]

        if azione["snapshot_da_riagganciare"]:
            session.execute(
                text("update odds_snapshot set id_match = :vinc where id_match = :perd"),
                {"vinc": id_vincente, "perd": id_perdente},
            )

        for tabella, chiave_riaggancio, chiave_cancella in (
            ("statistics", "statistics_da_riagganciare", "statistics_da_cancellare"),
            ("odds", "odds_da_riagganciare", "odds_da_cancellare"),
        ):
            if azione[chiave_riaggancio]:
                session.execute(
                    text(f"update {tabella} set id_match = :vinc where id_match = :perd"),
                    {"vinc": id_vincente, "perd": id_perdente},
                )
            elif azione[chiave_cancella]:
                session.execute(
                    text(f"delete from {tabella} where id_match = :perd"),
                    {"perd": id_perdente},
                )

        session.execute(
            text("delete from match where id_match_fk = :perd"),
            {"perd": id_perdente},
        )


def scrivi_backup(session, gruppi: dict[int, list[dict]], piani: list, percorso: str) -> None:
    """Salva lo stato PRE-fusione di ogni riga coinvolta.

    Gli `odds_snapshot` NON vengono dumpati (sono 311k righe e comunque
    vengono solo ri-agganciati, mai cancellati: restano tutti a DB). Le
    righe `statistics`/`odds` che invece vengono CANCELLATE sono dumpate per
    intero, colonna per colonna: sono poche decine e senza il contenuto il
    backup non permetterebbe di rimetterle come erano."""
    da_cancellare: dict[str, dict[str, list]] = {}
    for _id_fixture, piano in piani:
        for azione in piano["azioni"]:
            id_perdente = azione["id_match_fk"]
            righe_figlie: dict[str, list] = {}
            for tabella, chiave in (("statistics", "statistics_da_cancellare"), ("odds", "odds_da_cancellare")):
                if not azione[chiave]:
                    continue
                righe_figlie[tabella] = [
                    dict(r)
                    for r in session.execute(
                        text(f"select * from {tabella} where id_match = :perd"),
                        {"perd": id_perdente},
                    ).mappings()
                ]
            if righe_figlie:
                da_cancellare[id_perdente] = righe_figlie

    os.makedirs(os.path.dirname(percorso) or ".", exist_ok=True)
    with open(percorso, "w", encoding="utf-8") as f:
        for id_fixture, righe in sorted(gruppi.items()):
            f.write(
                json.dumps(
                    {
                        "id_fixture": id_fixture,
                        "salvato_il": datetime.now(timezone.utc).isoformat(),
                        "righe": righe,
                        "figli_cancellati": {
                            id_match: figli
                            for id_match, figli in da_cancellare.items()
                            if any(r["id_match_fk"] == id_match for r in righe)
                        },
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="scrive davvero a DB (default: simulazione)")
    parser.add_argument("--verbose", action="store_true", help="stampa il dettaglio di ogni gruppo")
    args = parser.parse_args()

    load_app_config()
    session = SessionLocal()

    gruppi = carica_gruppi(session)
    if not gruppi:
        print("Nessun id_fixture duplicato: niente da fare.")
        return 0

    totale = Counter()
    piani = []
    for id_fixture, righe in sorted(gruppi.items()):
        piano = pianifica_gruppo(righe)
        piani.append((id_fixture, piano))

        totale["gruppi"] += 1
        totale["righe_da_cancellare"] += len(piano["azioni"])
        for azione in piano["azioni"]:
            totale["snapshot_riagganciati"] += azione["snapshot_da_riagganciare"]
            totale["statistics_riagganciate"] += azione["statistics_da_riagganciare"]
            totale["statistics_cancellate"] += azione["statistics_da_cancellare"]
            totale["odds_riagganciate"] += azione["odds_da_riagganciare"]
            totale["odds_cancellate"] += azione["odds_da_cancellare"]

        if args.verbose:
            v = piano["vincente"]
            print(
                f"\nfixture {id_fixture}  {v['name_home']} - {v['name_away']}  {v['date_match']}"
            )
            print(
                f"  TIENE    {v['id_match_fk']}  status={v['status']:4} "
                f"score={v['score_home']}-{v['score_away']} "
                f"snap={v['n_snap']} stat={v['n_stat']} odds={v['n_odds']}"
            )
            for azione in piano["azioni"]:
                print(
                    f"  CANCELLA {azione['id_match_fk']}  status={str(azione['status']):4} "
                    f"-> snapshot riagganciati={azione['snapshot_da_riagganciare']} "
                    f"statistics(riag/canc)={azione['statistics_da_riagganciare']}/{azione['statistics_da_cancellare']} "
                    f"odds(riag/canc)={azione['odds_da_riagganciare']}/{azione['odds_da_cancellare']}"
                )

    print("\n=== RIEPILOGO ===")
    print(f"  gruppi duplicati              : {totale['gruppi']}")
    print(f"  righe match da cancellare     : {totale['righe_da_cancellare']}")
    print(f"  odds_snapshot ri-agganciati   : {totale['snapshot_riagganciati']}")
    print(f"  statistics ri-agganciate      : {totale['statistics_riagganciate']}")
    print(f"  statistics cancellate         : {totale['statistics_cancellate']}")
    print(f"  odds ri-agganciate            : {totale['odds_riagganciate']}")
    print(f"  odds cancellate               : {totale['odds_cancellate']}")

    if not args.apply:
        print("\nSimulazione: niente scritto. Rilancia con --apply per applicare.")
        return 0

    scrivi_backup(session, gruppi, piani, BACKUP)
    print(f"\nStato pre-fusione salvato in {BACKUP}")

    for id_fixture, piano in piani:
        try:
            applica_gruppo(session, piano)
            session.commit()
        except Exception as exc:
            session.rollback()
            print(f"ERRORE sul gruppo {id_fixture}: {exc}")
            raise

    residui = session.execute(
        text(
            "select count(*) from (select id_fixture from match "
            "where id_fixture is not null group by id_fixture having count(*) > 1) t"
        )
    ).scalar()
    print(f"\nFatto. id_fixture ancora duplicati: {residui}")
    return 0 if residui == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
