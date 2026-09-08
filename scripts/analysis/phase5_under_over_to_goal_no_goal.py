"""FASE 5 (idea proposta dall'operatore, 2026-09-08): "se il modello predice
Over su tutte le soglie fino a 4.5, c'e' una probabilita' alta che esca
anche Goal (BTTS)" - usare le 4 probabilita' OOF Under/Over (1.5/2.5/3.5/4.5)
INSIEME come feature aggiuntive per allenare goal_no_goal.

Direzione OPPOSTA rispetto a `phase4_goal_no_goal_feature_injection.py`
(che iniettava la probabilita' di goal_no_goal NEI 4 Under/Over, scartato:
delta max +0.0089). Qui l'idea e' concettualmente diversa, non solo
invertita: le 4 probabilita' Under/Over insieme sono un riassunto
compresso dell'INTERA distribuzione dei gol (quanti gol totali ci si
aspetta), non una singola probabilita' ridondante - piu' vicino a uno
stacking classico (feature = output di piu' base model) che a una feature
singola derivata dallo stesso spazio di input.

Anti-leakage: stessa infrastruttura di `phase3b_sequential_stacking_thresholds.py`
(`_cascade_oof`/`_assert_cascade_no_future_leakage`/`_score`, qui riusate
identiche) - le 4 OOF Under/Over sono calcolate walk-forward (RandomForest,
`_RF_KWARGS`, stesso identico approccio "segnale" gia' usato in fase 3b/4,
non i champion reali con GridSearch completo: qui serve solo un segnale
onesto per decidere se l'idea merita un retraining vero, non il modello
finale) sul frame condiviso 'totals' (`totals_evaluation_frame.csv`), poi
unite per `id_fixture` al dataset di goal_no_goal. Il confronto
baseline-vs-con-feature usa le STESSE righe (`eligible_train`/`eval_index`,
identico principio di fase 3b/4) per isolare l'effetto della sola feature.

Nota di scope: se il risultato fosse ADOTTATO, servirebbe poi un secondo
passo di ingegnerizzazione (fuori da questo script) per rendere le 4
probabilita' Under/Over disponibili anche a goal_no_goal in produzione
(dipendenza in piu' nella pipeline di training/serving, i 4 modelli
Under/Over andrebbero eseguiti prima di goal_no_goal) - qui si valuta SOLO
se il segnale statistico esiste, non si implementa la pipeline finale.
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
OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase5_under_over_to_goal_no_goal_result.json"))
ADOPTION_DELTA_THRESHOLD = 0.01


def _compute_under_over_oof(totals_frame: pd.DataFrame, feature_columns: list[str], cv_splits: list[tuple[list[int], list[int]]]) -> dict[str, np.ndarray]:
    """OOF walk-forward INDIPENDENTE per ciascuna delle 4 soglie (nessuna
    catena soglia-su-soglia, a differenza di phase3b: qui ogni soglia usa
    SOLO le proprie feature originali, esattamente come il livello radice
    1.5 di phase3b)."""
    oof_by_threshold: dict[str, np.ndarray] = {}
    for label in THRESHOLD_LABELS:
        y_col = f"y_{label}"
        oof_by_threshold[label] = _cascade_oof(totals_frame, feature_columns, cv_splits, y_col, restrict_train_to=None)
        n_defined = int(np.isfinite(oof_by_threshold[label]).sum())
        print(f"  {label}: {n_defined}/{len(totals_frame)} OOF definite")
    return oof_by_threshold


def run_injection(gg_df: pd.DataFrame, gg_feature_cols: list[str], uo_oof_by_fixture: dict[str, pd.Series], cv_splits: list[tuple[list[int], list[int]]]) -> dict[str, Any]:
    frame = gg_df.copy()
    uo_feature_cols = []
    for label, series in uo_oof_by_fixture.items():
        col = f"uo_{label}_oof"
        frame[col] = frame["id_fixture"].map(series)
        uo_feature_cols.append(col)

    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})
    if not oof_index_all:
        raise ValueError("Nessun indice OOF disponibile: walk-forward goal_no_goal non valido")

    eligible_train = {
        int(i) for i in oof_index_all
        if all(pd.notna(frame[col].iloc[i]) for col in uo_feature_cols)
    }
    if not eligible_train:
        raise ValueError("Nessuna riga con tutte le 4 OOF Under/Over definite: join id_fixture fallito o vuoto")

    _assert_cascade_no_future_leakage(frame=frame, cv_splits=cv_splits, oof_index_all=eligible_train)

    baseline_probs = _cascade_oof(frame, gg_feature_cols, cv_splits, "y", restrict_train_to=eligible_train)
    with_uo_columns = gg_feature_cols + uo_feature_cols
    with_uo_probs = _cascade_oof(frame, with_uo_columns, cv_splits, "y", restrict_train_to=eligible_train)

    eval_index = sorted(
        i for i in eligible_train if not np.isnan(baseline_probs[i]) and not np.isnan(with_uo_probs[i])
    )
    if not eval_index:
        raise ValueError("Nessuna riga valutabile dopo il join")

    baseline_metrics = _score(frame, "y", baseline_probs, eval_index)
    with_uo_metrics = _score(frame, "y", with_uo_probs, eval_index)
    delta = with_uo_metrics["selection_score"] - baseline_metrics["selection_score"]

    return {
        "n_eligible_train_rows": len(eligible_train),
        "n_eval_rows": len(eval_index),
        "baseline_no_uo_features": baseline_metrics,
        "with_uo_features": with_uo_metrics,
        "delta_selection_score": delta,
        "verdict": (
            f"ADOTTATO: delta selection_score {delta:+.4f} supera la soglia {ADOPTION_DELTA_THRESHOLD}"
            if delta > ADOPTION_DELTA_THRESHOLD
            else f"SCARTATO: delta selection_score {delta:+.4f} non supera la soglia {ADOPTION_DELTA_THRESHOLD}"
        ),
    }


def main() -> None:
    t0 = time.time()
    gg_df = pd.read_csv(os.path.join(EXPORT_DIR, "goal_no_goal.csv"))
    gg_df["prediction_at"] = pd.to_datetime(gg_df["prediction_at"], utc=True, errors="coerce")
    gg_df = gg_df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    gg_feature_cols = [c for c in gg_df.columns if c not in {"id_fixture", "season", "league", "market", "prediction_at", "y"}]
    print(f"goal_no_goal: {len(gg_df)} righe, {len(gg_feature_cols)} feature")

    totals_frame = pd.read_csv(os.path.join(EXPORT_DIR, "totals_evaluation_frame.csv"))
    totals_frame["prediction_at"] = pd.to_datetime(totals_frame["prediction_at"], utc=True, errors="coerce")
    totals_frame = totals_frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    with open(os.path.join(EXPORT_DIR, "totals_feature_columns.json"), "r", encoding="utf-8") as f:
        totals_feature_columns = json.load(f)
    print(f"totals: {len(totals_frame)} righe, {len(totals_feature_columns)} feature")

    min_train_totals = max(30, int(len(totals_frame) * 0.45))
    min_valid_totals = max(10, int(len(totals_frame) * 0.1))
    totals_cv_splits = expanding_window_splits(
        frame=totals_frame, time_col="prediction_at", n_splits=5,
        min_train_size=min_train_totals, min_valid_size=min_valid_totals,
    )
    if not totals_cv_splits:
        print("CV insufficiente sul frame totals, esperimento non eseguibile.")
        return

    print("\nCalcolo OOF Under/Over (indipendenti, RandomForest segnale):")
    uo_oof_arrays = _compute_under_over_oof(totals_frame, totals_feature_columns, totals_cv_splits)
    uo_oof_by_fixture = {
        label: pd.Series(arr, index=totals_frame["id_fixture"].to_numpy())
        for label, arr in uo_oof_arrays.items()
    }

    raw_splits = _build_temporal_cv(gg_df)
    gg_y = gg_df["y"].astype(int)
    gg_cv_splits = _filter_valid_splits(y=gg_y, splits=raw_splits)
    if not gg_cv_splits:
        print("CV insufficiente su goal_no_goal, esperimento non eseguibile.")
        return

    result = run_injection(gg_df=gg_df, gg_feature_cols=gg_feature_cols, uo_oof_by_fixture=uo_oof_by_fixture, cv_splits=gg_cv_splits)

    print(f"\n=== goal_no_goal: baseline vs +4 feature Under/Over (n_eval_rows={result['n_eval_rows']}) ===")
    print(f"baseline (solo feature originali)   : selection_score={result['baseline_no_uo_features']['selection_score']:.4f}")
    print(f"con 4 probabilita' Under/Over aggiunte: selection_score={result['with_uo_features']['selection_score']:.4f}")
    print(f"delta = {result['delta_selection_score']:+.4f}")
    print(f"\nVERDETTO: {result['verdict']}")

    payload = {**result, "elapsed_seconds": round(time.time() - t0, 2)}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
