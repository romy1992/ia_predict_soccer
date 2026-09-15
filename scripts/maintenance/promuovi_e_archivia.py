"""Allinea il registry dei modelli allo stato deciso il 2026-09-14.

DA ESEGUIRE SULLA MACCHINA DELL'OPERATORE, dove sta il registry vero. La
sessione cloud ha un registry effimero e scollegato: promuovere li' non
cambierebbe nulla per l'app.

COSA FA
Per Under/Over 1.5, 2.5 e 3.5 ricostruisce il modello nuovo dai `best_params`
gia' scelti (i report sono su git), lo calibra, lo registra e lo promuove.
I .pkl non viaggiano per git perche' `best_models/` e' in .gitignore, ma non
serve trasferirli: con lo stesso dataset e lo stesso seed il modello si
riottiene identico qui.

I modelli vecchi vengono SPOSTATI in `best_models/archivio/`, non cancellati:
il rollback deve restare possibile. Su richiesta esplicita dell'operatore.

Under/Over 4.5, corners, cards, goal_no_goal e h2h non vengono toccati.

IL CASO 3.5
Su 3.5 la policy rifiuta la promozione: il gate di qualita' passa su tutte e
cinque le soglie, ma il confronto con la production fallisce (selection_score
0,6961 contro 0,7008). L'operatore ha deciso di forzarla dopo aver visto le
prove da entrambe le parti:

  a favore   la production ha le quote LEGACY mescolate (una media sola su
             Over e Under insieme); il candidato le separa per esito, vince
             sull'AUC in tutte e tre le configurazioni provate (0,5974 contro
             0,5867) e dopo calibrazione ha ECE 0,0146 contro 0,1070
  contro     perde su log-loss (0,6363 contro 0,6287) e Brier (0,2225 contro
             0,2191), che sono le metriche su cui il selection_score pesa

La forzatura lascia traccia di entrambe nel registry: un bypass senza memoria
di cosa ha scavalcato e' peggio del bypass stesso.

SICUREZZA
- parte in SIMULAZIONE: senza --apply non scrive niente;
- prima di toccare il registry ne fa una copia in `_backup_<data>/`;
- salta i mercati gia' a posto (production gia' a 33 feature);
- alla fine verifica che ogni riga di registry punti a un file esistente e
  che i modelli promossi si carichino davvero.

Uso:
    python scripts/maintenance/promuovi_e_archivia.py            # simulazione
    python scripts/maintenance/promuovi_e_archivia.py --apply    # esegue
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

BEST = "best_models"
REGISTRY = os.path.join(BEST, "registry")
ARCHIVIO = os.path.join(BEST, "archivio")
CONTAINER = "/app/best_models"

# mercato -> (linea, candidato da imporre o None, va forzata la promozione?)
DA_FARE = {
    "under_over_1_5": ("1_5", None, False),
    "under_over_2_5": ("2_5", "logistic", False),
    "under_over_3_5": ("3_5", None, True),
}
INTOCCABILI = ("under_over_4_5", "corners", "cards", "goal_no_goal", "h2h")

MOTIVO_FORZATURA = (
    "PROMOZIONE FORZATA su decisione esplicita dell'operatore, contro il rifiuto della "
    "policy. Il gate di qualita' passa su tutte e cinque le soglie; blocca il confronto "
    "con la production (selection_score 0,6961 contro 0,7008, delta -0,0047). Motivo: la "
    "production ha le quote LEGACY mescolate (una sola media su Over e Under insieme, "
    "nessuna feature separata per esito) e il candidato le separa; il candidato vince "
    "sull'AUC in tutte e tre le configurazioni provate (0,5974 contro 0,5867) e per una "
    "strategia a soglia l'ordinamento conta piu' della taratura assoluta, che viene "
    "comunque sistemata dalla calibrazione (ECE 0,1301 -> 0,0146). Resta vero che il "
    "candidato perde su log-loss (0,6363 contro 0,6287) e Brier (0,2225 contro 0,2191), "
    "che sono le metriche su cui il selection_score pesa di piu': e' una scelta "
    "consapevole, non la correzione di un errore del gate."
)
METADATI_FORZATURA = {
    "forced": True,
    "policy_outcome": "rejected",
    "policy_comparison": {"candidate_score": 0.6961, "production_score": 0.7008, "delta": -0.0047},
    "gate_checks": "tutte superate (log_loss, brier, ece, auc, sample_size)",
    "evidenza_a_favore": {
        "auc_candidato": 0.5974, "auc_production": 0.5867,
        "ece_dopo_calibrazione": 0.0146, "ece_production": 0.1070,
        "feature": "33 separate per esito contro 69 con quote mescolate",
    },
    "evidenza_contraria": {"log_loss": "0,6363 contro 0,6287", "brier": "0,2225 contro 0,2191"},
}


def stato(registry) -> dict:
    fuori = {}
    for mercato in list(DA_FARE) + list(INTOCCABILI):
        p = registry.get_production(market=mercato)
        fuori[mercato] = None if p is None else {
            "run_id": p["run_id"],
            "feature": len(p.get("feature_names") or []),
            "model_path": p.get("model_path"),
        }
    return fuori


def stampa_stato(titolo: str, s: dict) -> None:
    print(f"\n{titolo}")
    print(f"  {'mercato':18} {'feature':>7}  {'run_id':24} file")
    print("  " + "-" * 82)
    for mercato, p in s.items():
        if p is None:
            print(f"  {mercato:18} {'-':>7}  {'(nessuna production)':24}")
            continue
        print(f"  {mercato:18} {p['feature']:7}  {p['run_id'][-22:]:24} {os.path.basename(p['model_path'] or '')}")


def backup_registry(applica: bool) -> str:
    dest = os.path.join(REGISTRY, f"_backup_{date.today():%Y%m%d}")
    print(f"\nbackup del registry in {dest}")
    if not applica:
        print("   (simulazione: non copiato)")
        return dest
    os.makedirs(dest, exist_ok=True)
    for nome in ("index.jsonl", "promotion_history.jsonl"):
        src = os.path.join(REGISTRY, nome)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(dest, nome))
        n_src = sum(1 for _ in open(src, encoding="utf-8"))
        n_dst = sum(1 for _ in open(os.path.join(dest, nome), encoding="utf-8"))
        if n_src != n_dst:
            raise RuntimeError(f"backup incompleto di {nome}: {n_src} righe contro {n_dst}")
        print(f"   {nome}: {n_src} righe")
    return dest


def archivia(registry, applica: bool) -> None:
    """Sposta i .pkl non piu' in produzione e riscrive i percorsi nel registry.

    Un file spostato senza aggiornare la riga che lo referenzia darebbe "File
    modello non trovato" al primo utilizzo, quindi le due cose vanno insieme.
    """
    print("\narchiviazione dei modelli non piu' in produzione")
    index = os.path.join(REGISTRY, "index.jsonl")
    righe = [json.loads(l) for l in open(index, encoding="utf-8")]
    in_produzione = {p["model_path"] for p in (registry.get_production(market=m) for m in DA_FARE) if p}

    spostati, aggiornate = [], 0
    for r in righe:
        if r.get("market") not in DA_FARE:
            continue
        percorso = r.get("model_path") or ""
        if not percorso or percorso in in_produzione or "/archivio/" in percorso:
            continue
        nome = os.path.basename(percorso)
        locale = os.path.join(BEST, nome)
        if not os.path.exists(locale):
            print(f"   {nome}: file gia' assente, aggiorno solo la riga")
        else:
            if applica:
                os.makedirs(ARCHIVIO, exist_ok=True)
                shutil.move(locale, os.path.join(ARCHIVIO, nome))
            spostati.append(nome)
        r["model_path"] = f"{CONTAINER}/archivio/{nome}"
        cal = (r.get("extra") or {}).get("calibration") or {}
        if cal.get("calibrator_path"):
            ncal = os.path.basename(cal["calibrator_path"])
            lcal = os.path.join(BEST, ncal)
            if os.path.exists(lcal) and applica:
                os.makedirs(ARCHIVIO, exist_ok=True)
                shutil.move(lcal, os.path.join(ARCHIVIO, ncal))
            cal["calibrator_path"] = f"{CONTAINER}/archivio/{ncal}"
        aggiornate += 1

    # Fino all'introduzione dei nomi con suffisso data, ogni riaddestramento
    # sovrascriveva `<mercato>_champion.pkl`: piu' run diversi condividono
    # quindi lo stesso nome file. Spostandolo, le altre righe resterebbero
    # rotte. Il bridge ne ha trovate 4 cosi' il 2026-09-14.
    for r in righe:
        percorso = r.get("model_path") or ""
        nome = os.path.basename(percorso)
        if nome in spostati and "/archivio/" not in percorso:
            r["model_path"] = f"{CONTAINER}/archivio/{nome}"
            aggiornate += 1

    print(f"   file spostati: {len(spostati)}   righe di registry aggiornate: {aggiornate}")
    for n in spostati:
        print(f"      {n}")
    if applica and aggiornate:
        with open(index, "w", encoding="utf-8") as f:
            for r in righe:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   {index} riscritto")
    elif not applica:
        print("   (simulazione: niente spostato, niente riscritto)")


def promuovi(mercato: str, linea: str, candidato: str | None, applica: bool) -> None:
    cmd = [sys.executable, os.path.join("scripts", "analysis", "promuovi_mercato.py"), mercato, linea]
    if candidato:
        cmd.append(candidato)
    print(f"\n>>> {' '.join(cmd)}")
    if not applica:
        print("   (simulazione: non eseguito)")
        return
    esito = subprocess.run(cmd, capture_output=True, text=True)
    for riga in esito.stdout.splitlines():
        if "reliability" not in riga:
            print("   " + riga)
    if esito.returncode != 0:
        print(esito.stderr[-2000:])
        raise RuntimeError(f"promozione fallita per {mercato}")


def forza(registry, mercato: str, applica: bool) -> None:
    p = registry.get_production(market=mercato)
    if p and "20260914" in (p.get("model_path") or ""):
        print(f"\n{mercato}: la policy ha gia' promosso il modello nuovo, nessuna forzatura necessaria")
        return
    ultimo = registry.get_latest(market=mercato)
    if not ultimo or "20260914" not in (ultimo.get("model_path") or ""):
        raise RuntimeError(f"{mercato}: non trovo il run appena registrato, mi fermo")
    print(f"\n{mercato}: la policy ha rifiutato. Forzatura su decisione dell'operatore.")
    print(f"   run: {ultimo['run_id']}")
    if not applica:
        print("   (simulazione: non forzato)")
        return
    registry.promote(run_id=ultimo["run_id"], to_stage="production", actor="operator_request",
                     reason=MOTIVO_FORZATURA, metadata=METADATI_FORZATURA)
    print("   forzata")


def verifica(registry) -> None:
    print("\nverifica finale")
    index = os.path.join(REGISTRY, "index.jsonl")
    rotte = []
    for l in open(index, encoding="utf-8"):
        r = json.loads(l)
        p = r.get("model_path") or ""
        locale = p.replace(CONTAINER, BEST) if p.startswith(CONTAINER) else p
        if locale and not os.path.exists(locale):
            rotte.append((r["run_id"], p))
    print(f"   righe che puntano a un file inesistente: {len(rotte)}")
    for run_id, p in rotte[:10]:
        print(f"      {run_id[-22:]}  ->  {p}")

    import joblib
    import pandas as pd
    for mercato, (linea, _, _) in DA_FARE.items():
        p = registry.get_production(market=mercato)
        if not p:
            print(f"   {mercato}: nessuna production")
            continue
        locale = (p["model_path"] or "").replace(CONTAINER, BEST)
        try:
            modello = joblib.load(locale)
            csv = os.path.join("scripts", "analysis", "_export", f"{mercato}_raw.csv")
            df = pd.read_csv(csv).head(50)
            prob = modello.predict_proba(df[p["feature_names"]])[:, 1]
            print(f"   {mercato}: caricato, {len(prob)} predizioni fra {prob.min():.3f} e {prob.max():.3f}")
        except Exception as exc:
            print(f"   {mercato}: NON CARICABILE -> {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="esegue davvero (default: simulazione)")
    args = parser.parse_args()

    from src.service_ia.training.model_registry import ModelRegistry
    registry = ModelRegistry()

    prima = stato(registry)
    stampa_stato("STATO ATTUALE", prima)

    da_rifare = {m: v for m, v in DA_FARE.items()
                 if not (prima.get(m) and prima[m]["feature"] == 33)}
    gia_ok = [m for m in DA_FARE if m not in da_rifare]
    if gia_ok:
        print(f"\ngia' a posto (production gia' a 33 feature), non li tocco: {', '.join(gia_ok)}")
    if da_rifare:
        print(f"da rifare: {', '.join(da_rifare)}")

    # L'archiviazione va fatta ANCHE quando non c'e' niente da promuovere: e'
    # proprio allora che i vecchi modelli sono superati e vanno spostati. Il
    # return anticipato stava prima di questo blocco, quindi rilanciando lo
    # script a promozioni gia' fatte usciva su "Niente da fare" e i .pkl a 69
    # feature non si sarebbero mai mossi. Il bridge se n'e' accorto il
    # 2026-09-14 e ha dovuto archiviarli a mano.
    backup_registry(args.apply)
    archivia(registry, args.apply)
    if not da_rifare:
        print("\nNiente da promuovere: fatta la sola archiviazione.")
        if args.apply:
            verifica(ModelRegistry())
        return 0

    for mercato, (linea, candidato, forzare) in da_rifare.items():
        promuovi(mercato, linea, candidato, args.apply)
        if forzare and args.apply:
            forza(ModelRegistry(), mercato, args.apply)

    if args.apply:
        stampa_stato("STATO FINALE", stato(ModelRegistry()))
        verifica(ModelRegistry())
        print("\nRicorda di riavviare i container api e scheduler.")
    else:
        print("\nSimulazione conclusa. Rilancia con --apply per eseguire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
