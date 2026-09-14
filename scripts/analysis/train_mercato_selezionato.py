"""Addestra un mercato con la pipeline COMPLETA sul set di feature scelto.

Generalizza `train_over_1_5_selected.py`, che aveva mercato, linea e colonne
scritti dentro. Qui sono argomenti, cosi' il passo 7 della procedura
(`docs/soccer_oracle_v2_detailed/PROMPT_rifacimento_mercato.md`) si ripete
identico su ogni linea.

Qui non si misura piu' niente a mano: gira `train_market()` invariata - grid
search sui 3 candidati, ensemble voting/stacking sui 2 migliori, champion per
selection_score, calibrazione isotonica - limitata al set scelto via
`feature_columns`. La sorgente dati e' il CSV gia' esportato invece del DB,
perche' la sessione cloud il Postgres non lo vede.

Uso:
    python scripts/analysis/train_mercato_selezionato.py under_over_2_5 2_5
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.rifacimento_mercato import quote_per_linea, righe_sospette  # noqa: E402
from src.service_ia.training.market_service.filter_market_service import FilterMarketService  # noqa: E402
from src.service_ia.training.train_multi_market import train_market  # noqa: E402

EXPORT = os.path.join("scripts", "analysis", "_export")

# Il set vincente del passo 5, misurato: tiri + disciplina battono sia i soli
# tiri sia tutte e 54 le statistiche. Stesso esito su Under/Over 1.5.
GRANDEZZE = [
    "shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal",
    "shots_outsidebox", "blocked_shots", "yellow_cards", "red_cards", "fouls",
]


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python scripts/analysis/train_mercato_selezionato.py <mercato> <linea>")
        return 1
    mercato, linea = sys.argv[1], sys.argv[2]
    percorso = os.path.join(EXPORT, f"{mercato}_raw.csv")
    if not os.path.exists(percorso):
        print(f"Manca il dataset grezzo: {percorso}")
        return 1

    df = pd.read_csv(percorso)
    quote = quote_per_linea(linea)
    statistiche = [
        f"mean_{g}_{lato}_stat"
        for g in GRANDEZZE
        for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]
    feature = quote + statistiche

    mancanti = [c for c in feature if c not in df.columns]
    if mancanti:
        print(f"Colonne assenti dal CSV: {mancanti}")
        return 1

    prima = len(df)
    df = df[df[quote].notna().all(axis=1)].copy()
    senza_quote = prima - len(df)

    # Quote impossibili da un bookmaker reale: overround sotto 1 significa che
    # la somma delle probabilita' implicite sta sotto il 100%. Dopo la
    # correzione della linea ne restano pochissime, quasi tutte di BetMGM sul
    # percorso `alternate_*`, di cui non abbiamo il payload grezzo per
    # ricostruirle. Misurato: scartarle migliora (AUC 0.6026 contro 0.6015).
    sospette = righe_sospette(df, linea)
    df = df[~sospette].copy()

    # Righe senza statistiche: scartate invece che riempite. Sono l'1,2% e
    # manca sempre il blocco intero, mai una grandezza sola. Riempirle con la
    # mediana di tutto il dataset introdurrebbe leakage temporale, e lo
    # StackingClassifier della pipeline ha passthrough=True - gira le feature
    # grezze al modello finale scavalcando gli imputer, quindi con NaN va in
    # errore.
    incomplete = df[statistiche].isna().any(axis=1)
    df = df[~incomplete].copy()

    print(f"mercato   : {mercato}   linea: {linea}")
    print(f"righe     : {len(df):,}  (scartate {senza_quote} senza quote, "
          f"{int(sospette.sum())} sospette, {int(incomplete.sum())} senza statistiche)")
    print(f"feature   : {len(feature)}  ({len(quote)} quote + {len(statistiche)} statistiche)")
    print(f"base rate : {df['y'].mean():.4f}\n")

    inizio = time.time()
    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        risultato = train_market(
            market=mercato,
            selection_method="kbest",
            save_model=True,
            feature_columns=feature,
        )
    durata = time.time() - inizio

    print(f"\nstato     : {risultato.status}")
    print(f"champion  : {risultato.champion}")
    print(f"durata    : {durata/60:.1f} minuti")
    print(f"feature selezionate dal modello: {len(risultato.selected_features)}")
    print(f"   {risultato.selected_features}")

    dettagli = risultato.details or {}
    print("\nCONFRONTO FRA I CANDIDATI:")
    for nome, payload in (dettagli.get("model_results") or {}).items():
        metriche = payload.get("probability_metrics") or {}
        print(
            f"   {nome:26} score {payload.get('selection_score')} | "
            f"auc {metriche.get('auc')} | logloss {metriche.get('log_loss')} | brier {metriche.get('brier')}"
        )

    calibrazione = dettagli.get("calibration") or {}
    if calibrazione.get("enabled"):
        print(f"\nCALIBRAZIONE ({calibrazione.get('method')}):")
        print(f"   prima : {calibrazione.get('pre_metrics')}")
        print(f"   dopo  : {calibrazione.get('post_metrics')}")

    destinazione = os.path.join(EXPORT, f"train_{mercato}_selected.json")
    with open(destinazione, "w", encoding="utf-8") as f:
        json.dump(
            {
                "market": mercato,
                "rows": len(df),
                "feature_columns": feature,
                "status": risultato.status,
                "champion": risultato.champion,
                "selected_features": risultato.selected_features,
                "details": dettagli,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\nreport salvato in {destinazione}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
