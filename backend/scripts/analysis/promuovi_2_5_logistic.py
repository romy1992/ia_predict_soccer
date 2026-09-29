"""Addestra e promuove Under/Over 2.5 con la LOGISTICA (scelta dell'operatore).

La pipeline sceglie il champion per `selection_score`, e su questo mercato ha
scelto `voting` (0.6802) contro `logistic` (0.6801). Ma la logistica batte
l'ensemble su TUTTE e quattro le metriche - AUC 0,6105 contro 0,6098,
log-loss, Brier ed ECE - e perde solo il punteggio composito, per un
decimillesimo. A parita' sostanziale di risultato una regressione logistica
su 33 colonne e' piu' semplice da servire, piu' stabile nel tempo e si puo'
leggere: l'operatore ha scelto quella.

Qui il champion e' quindi imposto, non selezionato. Tutto il resto - stessa
pipeline, stessi best_params della grid, stessa CV walk-forward, stessa
calibrazione - resta quello del training.

Il nome del file e' distinto da quello in produzione: sovrascrivendo
`under_over_2_5_champion.pkl` si perderebbe il rollback, e la vecchia riga di
registry punterebbe in silenzio al modello nuovo.

Uso:
    python scripts/analysis/promuovi_2_5_logistic.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.passo6_soglie_e_roi import costruisci, prepara  # noqa: E402
from src.ml.calibration.calibration_service import CalibrationService  # noqa: E402
from src.service_ia.training.model_registry import ModelRegistry  # noqa: E402
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402
from src.service_ia.training.utility_training.save_load import SaveLoad  # noqa: E402

MERCATO = "under_over_2_5"
LINEA = "2_5"
CANDIDATO = "logistic"
SUFFISSO = "20260914"
EXPORT = os.path.join("scripts", "analysis", "_export")


def main() -> int:
    report = json.load(open(os.path.join(EXPORT, f"train_{MERCATO}_selected.json")))
    params = report["details"]["models"][CANDIDATO]["best_params"]
    metriche_grid = report["details"]["models"][CANDIDATO]["probability_metrics"]

    df, feature = prepara(MERCATO, LINEA)
    X, y = df[feature], df["y"].astype(int)
    splits = _filter_valid_splits(y, _build_temporal_cv(df) or [])
    modello = costruisci(CANDIDATO, params, n_feature=len(feature))

    print(f"mercato : {MERCATO}   candidato imposto: {CANDIDATO}")
    print(f"righe   : {len(df):,}   feature: {len(feature)}   parametri: {params}")
    print(f"fold    : {len(splits)}   base rate: {y.mean():.4f}\n")

    ris = CalibrationService.calibrate_estimator(estimator=modello, X=X, y=y, cv_splits=splits)
    print(f"calibrazione: {ris.method}")
    print(f"   prima : ece {ris.pre_metrics.get('ece'):.4f}  logloss {ris.pre_metrics.get('log_loss'):.4f}  "
          f"brier {ris.pre_metrics.get('brier'):.4f}  auc {ris.pre_metrics.get('auc'):.4f}")
    print(f"   dopo  : ece {ris.post_metrics.get('ece'):.4f}  logloss {ris.post_metrics.get('log_loss'):.4f}  "
          f"brier {ris.post_metrics.get('brier'):.4f}  auc {ris.post_metrics.get('auc'):.4f}")

    calibrato = ris.calibrator
    percorso_cal = os.path.abspath(os.path.join("best_models", f"{MERCATO}_champion_calibrator_{SUFFISSO}.pkl"))
    os.makedirs(os.path.dirname(percorso_cal), exist_ok=True)
    joblib.dump(calibrato, percorso_cal)
    print(f"\ncalibratore salvato in {percorso_cal}")

    saver = SaveLoad(
        save_pkl=True,
        filename=f"{MERCATO}_champion_{SUFFISSO}",
        market_name=MERCATO,
        feature_names=feature,
        metrics={
            "selection_metric": "composite_probability_score",
            "selection_score": report["details"]["models"][CANDIDATO].get("selection_score"),
            "best_cv_f1": report["details"]["models"][CANDIDATO].get("best_cv_f1"),
            "log_loss": metriche_grid.get("log_loss"),
            "brier": metriche_grid.get("brier"),
            "ece": metriche_grid.get("ece"),
            "auc": metriche_grid.get("auc"),
            "calibration_enabled": True,
            "calibration_method": ris.method,
            "pre_calibration_log_loss": ris.pre_metrics.get("log_loss"),
            "post_calibration_log_loss": ris.post_metrics.get("log_loss"),
            "pre_calibration_brier": ris.pre_metrics.get("brier"),
            "post_calibration_brier": ris.post_metrics.get("brier"),
            "model_family": CANDIDATO,
            "rows": len(df),
            "selected_features_count": params.get("selector__k"),
        },
        registry_enabled=True,
    )
    saver.save_model(
        estimator=calibrato,
        model_name=CANDIDATO,
        params=params,
        extra={
            "selection_method": "kbest",
            "selection_in_pipeline": True,
            "champion_forced": True,
            "champion_forced_reason": (
                "logistic batte voting su AUC, log-loss, Brier ed ECE ma perde il "
                "selection_score per 0,0001; scelta esplicita dell'operatore per "
                "semplicita' e stabilita'"
            ),
            "calibration": {
                "enabled": True,
                "method": ris.method,
                "pre_metrics": ris.pre_metrics,
                "post_metrics": ris.post_metrics,
                "calibrator_path": percorso_cal,
                "sample_size": ris.sample_size,
            },
        },
    )

    registry = ModelRegistry()
    ultimo = registry.get_latest(market=MERCATO)
    # Controllo che sia davvero la riga appena scritta e non una precedente:
    # promuovere il run sbagliato manderebbe in produzione un altro modello.
    if not ultimo or SUFFISSO not in str(ultimo.get("model_path", "")):
        print(f"\nL'ultimo run registrato non e' quello appena salvato: {ultimo}")
        return 1
    run_id = ultimo["run_id"]
    print(f"\nrun registrato: {run_id}")
    print(f"   percorso: {ultimo.get('model_path')}")

    esito = registry.promote_with_policy(
        run_id=run_id,
        to_stage="production",
        reason=(
            "Rifacimento completo del mercato dopo la correzione delle quote: il 32% "
            "di quelle salvate come Over 2.5 era di un'altra linea. Champion imposto a "
            "logistic su scelta dell'operatore."
        ),
        actor="operator_request",
    )
    print(f"\nesito promozione: {esito}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
