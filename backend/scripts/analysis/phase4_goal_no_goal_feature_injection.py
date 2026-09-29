"""FASE 4 (idea richiesta dall'utente): usare la probabilita' OOF del
classificatore "goal_no_goal" (Both Teams To Score) come feature aggiuntiva
per allenare Under/Over 1.5/2.5/3.5/4.5.

Razionale di dominio: GG=Si (entrambe le squadre segnano) implica
DETERMINISTICAMENTE totale gol >= 2, quindi Over 1.5 - ma SOLO per quella
soglia (una partita 1-1 e' GG=Si ma Under 2.5). Per le soglie successive la
relazione e' solo correlativa, non certa. Inoltre la probabilita' *predetta*
dal modello resta comunque un valore incerto (non un fatto), quindi va
trattata come feature probabilistica, mai come override deterministico -
esattamente come la cascata sequenziale tra soglie di
`phase3b_sequential_stacking_thresholds.py`, di cui questo script riusa
l'infrastruttura (`_cascade_oof`/`_assert_cascade_no_future_leakage`), qui
con "goal_no_goal" come UNICA sorgente iniettata in OGNUNA delle 4 soglie
indipendentemente (non incatenata soglia-su-soglia).

DA NON CONFONDERE con:
- la cascata soglia-su-soglia di phase3b (1.5->2.5->3.5->4.5): qui la fonte
  e' sempre lo stesso modello goal_no_goal, non la soglia precedente.
- la feature deterministica `gd_over_1_5` di Fase 3 (Poisson, nessun
  training): qui la feature e' la probabilita' PREDETTA da un classificatore
  vero, allenato con la pipeline di produzione (`train_market("goal_no_goal")`).

Dati: richiede `scripts/analysis/_export/goal_no_goal.csv` (esportato da
`export_goal_no_goal_for_cloud_training.py`) e
`scripts/analysis/_export/totals_evaluation_frame.csv` +
`totals_feature_columns.json` (esportati da
`export_datasets_for_cloud_training.py`, recuperati dalla history git per
questo run - stesso identico schema, nessuna nuova query DB necessaria).

Anti-leakage: la OOF di goal_no_goal e' calcolata con lo stesso principio
walk-forward espandente gia' verificato in questa sessione (ogni valore usa
solo fixture di goal_no_goal cronologicamente precedenti alla propria). Il
join con il frame 'totals' e' per `id_fixture` (nessun rischio temporale
aggiuntivo). Il training di ciascuna soglia usa poi le proprie cv_splits
(gia' leak-safe per costruzione, chronological expanding window) ristrette
alle sole righe con OOF goal_no_goal definita - verificato esplicitamente
con `_assert_cascade_no_future_leakage` (riusata identica da phase3b), oltre
che dalla garanzia strutturale gia' citata.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from scripts.analysis.phase3b_sequential_stacking_thresholds import (
    _assert_cascade_no_future_leakage,
    _cascade_oof,
    _score,
)
from src.ml.markets.totals.totals_market import _RF_KWARGS, THRESHOLD_LABELS, _class1_probability
from src.ml.validation.temporal_split import expanding_window_splits
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

EXPORT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase4_goal_no_goal_feature_injection_result.json"))
ADOPTION_DELTA_THRESHOLD = 0.01


def _compute_goal_no_goal_oof(gg_df: pd.DataFrame) -> pd.Series:
    """OOF walk-forward per goal_no_goal, stessa CV di produzione
    (`_build_temporal_cv`/`_filter_valid_splits`, train_multi_market.py) -
    RandomForest semplice (`_RF_KWARGS`) per generare la feature, non il
    GridSearch completo (non serve: qui vogliamo solo un segnale OOF onesto,
    non il campione finale - il campione reale resta il .pkl gia' registrato)."""
    gg_df = gg_df.sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    raw_splits = _build_temporal_cv(gg_df)
    y = gg_df["y"].astype(int)
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)
    if not cv_splits:
        raise ValueError("CV goal_no_goal insufficiente per generare OOF")

    feature_cols = [c for c in gg_df.columns if c not in {"id_fixture", "season", "league", "market", "prediction_at", "y"}]
    X = gg_df[feature_cols]

    n = len(gg_df)
    oof = np.full(n, np.nan)
    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
            continue
        model = RandomForestClassifier(**_RF_KWARGS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = _class1_probability(proba, model.classes_)

    result = pd.Series(oof, index=gg_df["id_fixture"].to_numpy(), name="gg_oof")
    n_defined = int(np.isfinite(oof).sum())
    print(f"goal_no_goal OOF: {n_defined}/{n} righe definite (cv_splits={len(cv_splits)})")
    return result


def run_injection(totals_frame: pd.DataFrame, feature_columns: list[str], gg_oof_by_fixture: pd.Series, cv_splits: list[tuple[list[int], list[int]]]) -> dict[str, Any]:
    frame = totals_frame.copy()
    frame["gg_oof"] = frame["id_fixture"].map(gg_oof_by_fixture)

    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})
    if not oof_index_all:
        raise ValueError("Nessun indice OOF disponibile: walk-forward totals non valido")

    eligible_train = {int(i) for i in oof_index_all if pd.notna(frame["gg_oof"].iloc[i])}
    if not eligible_train:
        raise ValueError("Nessuna riga con gg_oof definita tra gli indici OOF: join id_fixture fallito o vuoto")

    _assert_cascade_no_future_leakage(frame=frame, cv_splits=cv_splits, oof_index_all=eligible_train)

    per_threshold: dict[str, dict[str, Any]] = {}
    for label in THRESHOLD_LABELS:
        y_col = f"y_{label}"

        baseline_probs = _cascade_oof(frame, feature_columns, cv_splits, y_col, restrict_train_to=eligible_train)
        with_gg_columns = feature_columns + ["gg_oof"]
        with_gg_probs = _cascade_oof(frame, with_gg_columns, cv_splits, y_col, restrict_train_to=eligible_train)

        eval_index = sorted(
            i for i in eligible_train if not np.isnan(baseline_probs[i]) and not np.isnan(with_gg_probs[i])
        )
        if not eval_index:
            print(f"  {label}: nessuna riga valutabile, salto")
            continue

        baseline_metrics = _score(frame, y_col, baseline_probs, eval_index)
        with_gg_metrics = _score(frame, y_col, with_gg_probs, eval_index)
        delta = with_gg_metrics["selection_score"] - baseline_metrics["selection_score"]

        per_threshold[label] = {
            "baseline_no_gg": baseline_metrics,
            "with_gg_feature": with_gg_metrics,
            "delta_selection_score": delta,
            "n_eval_rows": len(eval_index),
        }

    deltas = [payload["delta_selection_score"] for payload in per_threshold.values()]
    adopted = [label for label, payload in per_threshold.items() if payload["delta_selection_score"] > ADOPTION_DELTA_THRESHOLD]

    return {
        "n_eligible_train_rows": len(eligible_train),
        "per_threshold": per_threshold,
        "adopted_thresholds": adopted,
        "verdict": (
            f"ADOTTATO su: {', '.join(adopted)} (delta selection_score > {ADOPTION_DELTA_THRESHOLD})"
            if adopted
            else f"SCARTATO: nessuna soglia migliora selection_score di oltre {ADOPTION_DELTA_THRESHOLD} con la feature gg_oof"
        ),
    }


def main() -> None:
    t0 = time.time()
    gg_df = pd.read_csv(os.path.join(EXPORT_DIR, "goal_no_goal.csv"))
    gg_df["prediction_at"] = pd.to_datetime(gg_df["prediction_at"], utc=True, errors="coerce")
    gg_df = gg_df.dropna(subset=["prediction_at"])
    print(f"goal_no_goal: {len(gg_df)} righe")

    totals_frame = pd.read_csv(os.path.join(EXPORT_DIR, "totals_evaluation_frame.csv"))
    totals_frame["prediction_at"] = pd.to_datetime(totals_frame["prediction_at"], utc=True, errors="coerce")
    totals_frame = totals_frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    with open(os.path.join(EXPORT_DIR, "totals_feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    print(f"totals: {len(totals_frame)} righe, {len(feature_columns)} feature")

    gg_oof_by_fixture = _compute_goal_no_goal_oof(gg_df)

    min_train = max(30, int(len(totals_frame) * 0.45))
    min_valid = max(10, int(len(totals_frame) * 0.1))
    cv_splits = expanding_window_splits(frame=totals_frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid)
    if not cv_splits:
        print("CV insufficiente, esperimento non eseguibile.")
        return

    result = run_injection(totals_frame=totals_frame, feature_columns=feature_columns, gg_oof_by_fixture=gg_oof_by_fixture, cv_splits=cv_splits)

    print(f"\n=== Feature injection goal_no_goal -> Under/Over (n_eligible_train_rows={result['n_eligible_train_rows']}) ===")
    for label in THRESHOLD_LABELS:
        payload = result["per_threshold"].get(label)
        if payload is None:
            continue
        b = payload["baseline_no_gg"]
        w = payload["with_gg_feature"]
        print(
            f"{label:>10s}: n_eval_rows={payload['n_eval_rows']} "
            f"baseline_selection_score={b['selection_score']:.4f} "
            f"with_gg_selection_score={w['selection_score']:.4f} "
            f"delta={payload['delta_selection_score']:+.4f}"
        )

    print(f"\nVERDETTO: {result['verdict']}")

    payload = {**result, "elapsed_seconds": round(time.time() - t0, 2)}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
