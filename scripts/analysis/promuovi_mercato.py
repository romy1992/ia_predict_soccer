"""Addestra e promuove un mercato con il candidato scelto.

Senza indicare il candidato si usa il champion del report di training. Si
puo' imporne un altro quando il gate ha scelto per un margine che i dati non
sostengono: su Under/Over 2.5 aveva preso `voting` (0.6802) contro `logistic`
(0.6801), ma la logistica batteva l'ensemble su AUC, log-loss, Brier ed ECE e
perdeva solo il punteggio composito, per un decimillesimo.

Tutto il resto - stessa pipeline, stessi best_params della grid, stessa CV
walk-forward, stessa calibrazione - resta quello del training.

Il nome del file e' distinto da quello in produzione: sovrascrivendo
`under_over_2_5_champion.pkl` si perderebbe il rollback, e la vecchia riga di
registry punterebbe in silenzio al modello nuovo.

Il file viene salvato nella cartella del mercato, `best_models/under_over/
<mercato>/`, insieme al suo calibratore.

Uso:
    python scripts/analysis/promuovi_mercato.py under_over_3_5 3_5
    python scripts/analysis/promuovi_mercato.py under_over_2_5 2_5 logistic
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
from src.service_ia.training.model_paths import destination_subdir, to_container_path  # noqa: E402
from src.service_ia.training.model_registry import ModelRegistry  # noqa: E402
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402
from src.service_ia.training.utility_training.save_load import SaveLoad  # noqa: E402

SUFFISSO = "20260914"
EXPORT = os.path.join("scripts", "analysis", "_export")


def riscrivi_percorsi_container() -> None:
    """Porta in forma container i percorsi delle righe appena registrate.

    Tocca solo le righe che hanno un percorso non ancora in forma container:
    quelle gia' corrette restano come sono. La conversione passa da
    `to_container_path`, che PRESERVA la sottocartella: da quando i modelli
    nuovi stanno in `best_models/under_over/<mercato>/`, ricostruire il
    percorso dal solo nome file lo appiattirebbe nella radice e il file non si
    troverebbe piu'.
    """
    index = os.path.join("best_models", "registry", "index.jsonl")
    if not os.path.exists(index):
        return
    righe = [json.loads(l) for l in open(index, encoding="utf-8")]
    corrette = 0
    for r in righe:
        for campo in ("model_path", "metadata_path"):
            valore = r.get(campo) or ""
            nuovo = to_container_path(valore)
            if valore and nuovo != valore:
                r[campo] = nuovo
                corrette += 1
        cal = (r.get("extra") or {}).get("calibration") or {}
        valore = cal.get("calibrator_path") or ""
        nuovo = to_container_path(valore)
        if valore and nuovo != valore:
            cal["calibrator_path"] = nuovo
            corrette += 1
    if corrette:
        with open(index, "w", encoding="utf-8") as f:
            for r in righe:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"percorsi riscritti in forma container: {corrette}")


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python scripts/analysis/promuovi_mercato.py <mercato> <linea> [candidato]")
        return 1
    MERCATO, LINEA = sys.argv[1], sys.argv[2]
    report = json.load(open(os.path.join(EXPORT, f"train_{MERCATO}_selected.json")))
    CANDIDATO = sys.argv[3] if len(sys.argv) > 3 else report["champion"]
    imposto = CANDIDATO != report["champion"]
    params = report["details"]["models"][CANDIDATO]["best_params"]
    metriche_grid = report["details"]["models"][CANDIDATO]["probability_metrics"]

    df, feature = prepara(MERCATO, LINEA)
    X, y = df[feature], df["y"].astype(int)
    splits = _filter_valid_splits(y, _build_temporal_cv(df) or [])
    modello = costruisci(CANDIDATO, params, n_feature=len(feature))

    print(f"mercato : {MERCATO}   candidato: {CANDIDATO}"
          f"{'  (IMPOSTO, non e il champion)' if imposto else '  (champion del gate)'}")
    print(f"righe   : {len(df):,}   feature: {len(feature)}   parametri: {params}")
    print(f"fold    : {len(splits)}   base rate: {y.mean():.4f}\n")

    ris = CalibrationService.calibrate_estimator(estimator=modello, X=X, y=y, cv_splits=splits)
    print(f"calibrazione: {ris.method}")
    print(f"   prima : ece {ris.pre_metrics.get('ece'):.4f}  logloss {ris.pre_metrics.get('log_loss'):.4f}  "
          f"brier {ris.pre_metrics.get('brier'):.4f}  auc {ris.pre_metrics.get('auc'):.4f}")
    print(f"   dopo  : ece {ris.post_metrics.get('ece'):.4f}  logloss {ris.post_metrics.get('log_loss'):.4f}  "
          f"brier {ris.post_metrics.get('brier'):.4f}  auc {ris.post_metrics.get('auc'):.4f}")

    calibrato = ris.calibrator
    # Modello e calibratore vanno nella cartella del mercato
    # (`best_models/under_over/<mercato>/`), non piu' nella radice: la radice
    # era una cartella sola con dentro i modelli di tutti i mercati, dove
    # riconoscere quali fossero quelli in uso richiedeva di leggere il
    # registry riga per riga.
    sotto = destination_subdir(MERCATO)
    percorso_cal = os.path.abspath(
        os.path.join("best_models", sotto, f"{MERCATO}_champion_calibrator_{SUFFISSO}.pkl")
    )
    os.makedirs(os.path.dirname(percorso_cal), exist_ok=True)
    joblib.dump(calibrato, percorso_cal)
    print(f"\ncalibratore salvato in {percorso_cal}")

    saver = SaveLoad(
        save_pkl=True,
        filename=os.path.join(sotto, f"{MERCATO}_champion_{SUFFISSO}"),
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
            "champion_forced": imposto,
            "champion_forced_reason": (
                "scelta esplicita dell'operatore contro il gate" if imposto else None
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

    # `SaveLoad` registra il percorso ASSOLUTO della macchina su cui gira. Ma
    # l'app dell'operatore gira in Docker con `./best_models:/app/best_models`,
    # quindi al primo utilizzo darebbe "File modello non trovato". Va riscritto
    # in forma container. Il bridge se n'e' accorto il 2026-09-14 e ha dovuto
    # correggere a mano le tre righe appena promosse.
    riscrivi_percorsi_container()

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
            f"Rifacimento del mercato con la procedura completa: quote separate per "
            f"esito invece delle legacy mescolate, set di feature scelto sui numeri "
            f"(quote + tiri + disciplina), candidato {CANDIDATO}."
        ),
        actor="operator_request",
    )
    print(f"\nesito promozione: {esito}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
