"""Addestra Under/Over 1.5 con la pipeline COMPLETA sul set di feature scelto.

Chiude il percorso fatto su questo mercato: EDA sul dato grezzo, unificazione
degli alias nelle quote, misura del contributo di ogni blocco di statistiche
e confronto dei set. Il vincitore e' `quote + tiri + disciplina` (33 colonne),
che batte le 60 complete su AUC, log-loss e Brier.

Qui non si misura piu' niente a mano: gira `train_market()` invariata - grid
search sui 3 candidati, ensemble voting/stacking sui 2 migliori, champion per
selection_score, calibrazione - limitata alle 33 colonne via `feature_columns`.
La sorgente dati e' il CSV gia' esportato invece del DB (stesso pattern di
`train_from_export.py`), perche' la sessione cloud il Postgres non lo vede.

Uso:
    python scripts/analysis/train_over_1_5_selected.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.market_service.filter_market_service import FilterMarketService  # noqa: E402
from src.service_ia.training.train_multi_market import train_market  # noqa: E402

MERCATO = "under_over_1_5"
CSV = os.path.join("scripts", "analysis", "_export", f"{MERCATO}_raw.csv")

QUOTE = [
    "prob_norm_over_1_5",
    "odds_mean_over_1_5",
    "odds_mean_under_1_5",
    "odds_count",
    "odds_std_over_1_5",
    "overround",
]
GRANDEZZE = [
    "shots_on_goal",
    "total_shots",
    "shots_insidebox",
    "shots_off_goal",
    "shots_outsidebox",
    "blocked_shots",
    "yellow_cards",
    "red_cards",
    "fouls",
]


def main() -> int:
    df = pd.read_csv(CSV)
    statistiche = [
        f"mean_{g}_{lato}_stat"
        for g in GRANDEZZE
        for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]
    feature = QUOTE + statistiche

    mancanti = [c for c in feature if c not in df.columns]
    if mancanti:
        print(f"Colonne assenti dal CSV: {mancanti}")
        return 1

    # Le righe senza quote non sono utilizzabili: senza prezzo non esiste ne'
    # la feature principale ne' una scommessa da valutare.
    prima = len(df)
    df = df[df[QUOTE].notna().all(axis=1)].copy()

    # Righe senza statistiche: scartate invece che riempite. Sono l'1.3% (il
    # blocco `mean_statistics` manca in blocco, mai una grandezza sola), e
    # scartarle e' l'unica opzione senza controindicazioni: riempirle con la
    # mediana calcolata su tutto il dataset introdurrebbe un filo di leakage
    # temporale, e lo `StackingClassifier` della pipeline ha `passthrough=True`
    # - gira le feature grezze al modello finale scavalcando gli imputer dei
    # modelli base, quindi con NaN va proprio in errore.
    incomplete = df[statistiche].isna().any(axis=1)
    df = df[~incomplete].copy()
    print(f"righe scartate: {prima - len(df)} (senza quote o senza statistiche)")

    print(f"mercato   : {MERCATO}")
    print(f"righe     : {len(df):,}")
    print(f"feature   : {len(feature)}  ({len(QUOTE)} quote + {len(statistiche)} statistiche)")
    print(f"base rate : {df['y'].mean():.4f}\n")

    inizio = time.time()
    with patch.object(FilterMarketService, "build_dataset", return_value=df):
        risultato = train_market(
            market=MERCATO,
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

    destinazione = os.path.join("scripts", "analysis", "_export", f"train_{MERCATO}_selected.json")
    with open(destinazione, "w", encoding="utf-8") as f:
        json.dump(
            {
                "market": MERCATO,
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
