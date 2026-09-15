"""FASE 6: riaddestro h2h e dc sul dataset INTERO con quote per ESITO,
poi A/B della cascata dc + P_OOF(h2h).

Contesto (gia' diagnosticato, non rifatto qui): i champion serviti oggi
sono `stage: candidate`, addestrati su 400 righe, 69 feature IDENTICHE
tra i due mercati (15 quote aggregate `odds_mean/slot_*` + 54 mean_stats).
Nessuna quota 1X2 per esito. Entrambi degeneri sulla classe maggioritaria.

Questo script:

1. Verifica a DB le chiavi reali di `Odds.h2h` / `Odds.dc` (non assume
   i nomi degli esiti).
2. Costruisce i dataset con `FilterMarketService.build_dataset` (target `y`)
   e tiene SOLO le feature per esito + mean_stats, escludendo
   `LEGACY_ODDS_FEATURES` (slot/mean mescolati).
3. Walk-forward OOF (`_build_temporal_cv` di produzione, stesso seed) con
   RandomForest `class_weight='balanced'` - segnale onesto, non la grid
   search di `train_market()`: qui si decide SE l'idea merita un retraining
   di produzione, non si promuove niente.
4. A/B obbligatorio a parita' di split e seed: `dc` base vs `dc` +
   P_OOF(h2h). Le probabilita' h2h sono out-of-fold (`_cascade_oof` /
   walk-forward), mai in-sample. Pattern gia' validato in
   `phase5_under_over_to_goal_no_goal.py` / `phase3b_sequential_stacking_thresholds.py`.
   Se la cascata non vince, si documenta e si ferma (precedente: phase3).

Nessun `save_model` / nessuna promozione: se una variante batte DAVVERO il
baseline (non solo +1pp di accuracy sulla classe maggioritaria), lo script
PROPONE la promozione e si ferma.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from scripts.analysis.phase3b_sequential_stacking_thresholds import (
    _assert_cascade_no_future_leakage,
)
from src.ml.evaluation.probability_metrics import (
    champion_probability_score,
    compute_probability_metrics,
)
from src.repository.base.repository_db import SessionLocal
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase6_h2h_dc_retrain_cascade_result.json"))
SEED = 42
ADOPTION_DELTA_THRESHOLD = 0.01
# Sotto queste soglie il modello e' ancora degenere (macro-F1 < coin-flip
# per-classe, recall della minoranza vicino a zero come i champion da 400 righe).
MIN_MACRO_F1 = 0.50
MIN_MINORITY_RECALL = 0.25
# +1.2 pp su dc e' il "sembra buono ed e' il nulla" gia' misurato sui
# champion degeneri: per proporre una promozione serve uno scarto piu' largo.
MIN_ACCURACY_PP_FOR_PROMOTION = 3.0

_META_COLUMNS = {"y", "market", "id_fixture", "season", "league", "prediction_at"}
_RF_KWARGS = dict(
    n_estimators=300,
    min_samples_leaf=20,
    random_state=SEED,
    n_jobs=1,
    class_weight="balanced",
)

H2H_CANONICAL = ("home", "draw", "away")
DC_CANONICAL = ("1x", "12", "x2")
H2H_REQUIRED_ODDS = ["odds_mean_home", "odds_mean_draw", "odds_mean_away"]
DC_REQUIRED_ODDS = ["odds_mean_1x", "odds_mean_x2"]


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def inspect_odds_keys() -> dict[str, Any]:
    """Chiavi per-esito REALI in `Odds.h2h` / `Odds.dc`, prima di scegliere
    le feature. Nessuna chiamata ad API-Sports: sola lettura DB."""
    session = SessionLocal()
    try:
        counts = session.execute(
            text("SELECT COUNT(*) AS n_odds, COUNT(h2h) AS n_h2h, COUNT(dc) AS n_dc FROM odds")
        ).mappings().one()
        rows = session.execute(text("SELECT h2h, dc FROM odds")).mappings().all()
    finally:
        SessionLocal.remove()

    h2h_slugs = Counter()
    dc_slugs = Counter()
    dc_raw_outcomes = Counter()
    h2h_n = dc_n = 0
    h2h_all3 = 0
    dc_has_1x = dc_has_12 = dc_has_x2_raw = dc_has_draw_away_raw = 0
    dc_x2_after_alias = 0
    dc_contaminated = 0

    for row in rows:
        h2h = row["h2h"]
        dc = row["dc"]
        if isinstance(h2h, dict) and h2h:
            h2h_n += 1
            slugs = set()
            for key in h2h:
                outcome, _bm = FilterMarketService._split_outcome_and_bookmaker(str(key))
                slugs.add(FilterMarketService._normalize_outcome_name(outcome))
            for slug in slugs:
                h2h_slugs[slug] += 1
            if set(H2H_CANONICAL) <= slugs:
                h2h_all3 += 1
        if isinstance(dc, dict) and dc:
            dc_n += 1
            slugs = set()
            raw_outcomes = []
            for key in dc:
                outcome, _bm = FilterMarketService._split_outcome_and_bookmaker(str(key))
                raw_outcomes.append(outcome)
                dc_raw_outcomes[outcome] += 1
                slugs.add(FilterMarketService._normalize_outcome_name(outcome))
            for slug in slugs:
                dc_slugs[slug] += 1
            raw_lower = {str(o).strip().lower() for o in raw_outcomes}
            if "1x" in slugs:
                dc_has_1x += 1
            if "12" in slugs:
                dc_has_12 += 1
            if "x2" in raw_lower:
                dc_has_x2_raw += 1
            if "draw/away" in raw_lower or "draw_away" in raw_lower:
                dc_has_draw_away_raw += 1
            if "x2" in slugs:
                dc_x2_after_alias += 1
            if any(slug.startswith("over_") or slug.startswith("under_") for slug in slugs):
                dc_contaminated += 1

    return {
        "table_counts": dict(counts),
        "h2h": {
            "rows_with_json": h2h_n,
            "rows_with_home_draw_away": h2h_all3,
            "slug_row_coverage": dict(h2h_slugs.most_common()),
            "note": "100% home/draw/away se slug_row_coverage ha solo quelle tre chiavi.",
        },
        "dc": {
            "rows_with_json": dc_n,
            "rows_with_1x": dc_has_1x,
            "rows_with_12": dc_has_12,
            "rows_with_raw_X2": dc_has_x2_raw,
            "rows_with_raw_draw_away": dc_has_draw_away_raw,
            "rows_with_x2_after_alias": dc_x2_after_alias,
            "rows_contaminated_with_over_under": dc_contaminated,
            "slug_row_coverage": dict(dc_slugs.most_common(20)),
            "raw_outcome_prefixes": dict(dc_raw_outcomes.most_common(15)),
            "note": (
                "draw/away viene unificato a x2 da _OUTCOME_ALIASES. "
                "corner Over/Under 12.5 nel bucket dc e' contamination: "
                "CANONICAL_ODDS_OUTCOMES['dc'] lo esclude dall'overround."
            ),
        },
    }


def _mean_stat_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_home_stat") or c.endswith("_away_stat") or c.endswith("_diff_stat")]


def h2h_feature_columns(df: pd.DataFrame) -> list[str]:
    odds = [
        "prob_norm_home",
        "prob_norm_draw",
        "prob_norm_away",
        "odds_mean_home",
        "odds_mean_draw",
        "odds_mean_away",
        "odds_count_home",
        "odds_std_home",
        "overround",
    ]
    return [c for c in odds if c in df.columns] + _mean_stat_columns(df)


def dc_feature_columns(df: pd.DataFrame) -> list[str]:
    # Double Chance NON e' un mercato esclusivo: 1X/12/X2 sommano ~2, non ~1.
    # `implied_prob_1x` = 1/quota media 1X e' la stima grezza di P(1X).
    # `prob_norm_1x` dividerebbe per l'overround a tre chiavi (~2.1) e
    # dimezzerebbe la probabilita': NON usarla come feature principale.
    odds = [
        "implied_prob_1x",
        "implied_prob_x2",
        "implied_prob_12",
        "odds_mean_1x",
        "odds_mean_x2",
        "odds_mean_12",
        "odds_count_1x",
        "odds_std_1x",
        "overround",
    ]
    return [c for c in odds if c in df.columns] + _mean_stat_columns(df)


def _prepare_frame(df: pd.DataFrame, required_odds: list[str]) -> pd.DataFrame:
    frame = df.copy()
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"])
    present = [c for c in required_odds if c in frame.columns]
    if present:
        mask = (frame[present] > 0).all(axis=1)
        frame = frame[mask]
    frame = frame.sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


def _model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**_RF_KWARGS)),
        ]
    )


def _oof_probs(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: list[tuple[list[int], list[int]]],
    y_col: str = "y",
    restrict_train_to: Optional[set[int]] = None,
) -> np.ndarray:
    n = len(frame)
    oof = np.full(n, np.nan)
    y = frame[y_col].astype(int)
    X = frame[feature_columns]
    for train_idx, valid_idx in cv_splits:
        effective_train = train_idx if restrict_train_to is None else [i for i in train_idx if i in restrict_train_to]
        if not effective_train or not valid_idx:
            continue
        if y.iloc[effective_train].nunique() < 2:
            continue
        model = _model()
        model.fit(X.iloc[effective_train], y.iloc[effective_train])
        proba = model.predict_proba(X.iloc[valid_idx])
        classes = list(model.named_steps["rf"].classes_)
        class1_col = classes.index(1) if 1 in classes else -1
        oof[valid_idx] = proba[:, class1_col]
    return oof


def _majority_baseline(y: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(y.astype(int), minlength=2)
    majority = int(np.argmax(counts))
    accuracy = float((y == majority).mean())
    pred = np.full_like(y, majority)
    report = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
    return {
        "majority_class": majority,
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "classification_report": report,
    }


def _bookmaker_baseline(y: np.ndarray, p_market: np.ndarray) -> Optional[dict[str, Any]]:
    if p_market is None or len(p_market) != len(y) or np.any(~np.isfinite(p_market)):
        return None
    pred = (p_market >= 0.5).astype(int)
    metrics = compute_probability_metrics(y_true=y, probabilities=p_market, n_bins=10)
    report = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": metrics.get("auc"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "log_loss": metrics.get("log_loss"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "classification_report": report,
        "pred_positive_rate": float(pred.mean()),
    }


def score_oof(y: np.ndarray, p: np.ndarray, p_market: Optional[np.ndarray] = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    metrics = compute_probability_metrics(y_true=y, probabilities=p, n_bins=10)
    report = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y, pred, average="weighted", zero_division=0))
    majority = _majority_baseline(y)
    accuracy = float(accuracy_score(y, pred))
    accuracy_pp = (accuracy - majority["accuracy"]) * 100.0
    pos_rate = float(y.mean())
    minority_label = 0 if pos_rate >= 0.5 else 1
    minority_recall = float(report[str(minority_label)]["recall"])
    selection_score = champion_probability_score(metrics=metrics, f1_weighted=weighted_f1)
    degenerate = macro_f1 < MIN_MACRO_F1 or minority_recall < MIN_MINORITY_RECALL
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority["accuracy"],
        "accuracy_minus_majority_pp": accuracy_pp,
        "majority_class": majority["majority_class"],
        "class_counts": majority["class_counts"],
        "positive_rate": pos_rate,
        "pred_positive_rate": float(pred.mean()),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "minority_class": minority_label,
        "minority_recall": minority_recall,
        "auc": metrics.get("auc"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "log_loss": metrics.get("log_loss"),
        "selection_score": selection_score,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "labels": "[true 0/1] x [pred 0/1]"},
        "classification_report": report,
        "degenerate": degenerate,
        "bookmaker_implied_baseline": _bookmaker_baseline(y, p_market) if p_market is not None else None,
    }


def evaluate_market(frame: pd.DataFrame, feature_columns: list[str], market_prob_col: Optional[str]) -> dict[str, Any]:
    y = frame["y"].astype(int)
    cv_splits = _filter_valid_splits(y, _build_temporal_cv(frame) or [])
    if len(cv_splits) < 2:
        return {
            "metrics": {"status": "skipped_invalid_temporal_folds", "n_rows": int(len(frame)), "degenerate": True},
            "oof": np.full(len(frame), np.nan),
            "cv_splits": [],
            "frame": frame,
        }

    oof = _oof_probs(frame, feature_columns, cv_splits)
    valid = np.isfinite(oof)
    idx = np.where(valid)[0]
    p_market = frame[market_prob_col].to_numpy()[idx] if market_prob_col and market_prob_col in frame.columns else None
    scored = score_oof(y.to_numpy()[idx], oof[idx], p_market=p_market)
    scored.update(
        {
            "status": "ok",
            "n_rows": int(len(frame)),
            "n_oof": int(len(idx)),
            "n_features": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "cv_folds": int(len(cv_splits)),
        }
    )
    return {"metrics": scored, "oof": oof, "cv_splits": cv_splits, "frame": frame}


def _print_metrics(title: str, metrics: dict[str, Any]) -> None:
    cm = metrics.get("confusion_matrix") or {}
    print(f"\n=== {title} ===")
    print(
        f"n={metrics.get('n')} n_oof={metrics.get('n_oof', metrics.get('n'))} "
        f"feature={metrics.get('n_features', '?')}"
    )
    print(
        f"accuracy={metrics['accuracy']:.4f}  majority={metrics['majority_baseline_accuracy']:.4f}  "
        f"delta={metrics['accuracy_minus_majority_pp']:+.2f} pp"
    )
    print(
        f"auc={metrics['auc']:.4f}  brier={metrics['brier']:.4f}  ece={metrics['ece']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}  minority_recall(class {metrics['minority_class']})="
        f"{metrics['minority_recall']:.4f}"
    )
    print(
        f"pred_positive_rate={metrics['pred_positive_rate']:.3f}  "
        f"true_positive_rate={metrics['positive_rate']:.3f}  degenerate={metrics['degenerate']}"
    )
    print(f"confusion_matrix TN={cm.get('tn')} FP={cm.get('fp')} FN={cm.get('fn')} TP={cm.get('tp')}")
    report = metrics.get("classification_report") or {}
    for label in ("0", "1"):
        row = report.get(label) or {}
        print(
            f"  class {label}: precision={row.get('precision', 0):.4f} "
            f"recall={row.get('recall', 0):.4f} f1={row.get('f1-score', 0):.4f} "
            f"support={row.get('support', 0)}"
        )
    book = metrics.get("bookmaker_implied_baseline")
    if book:
        print(
            f"bookmaker implied: accuracy={book['accuracy']:.4f} auc={book['auc']:.4f} "
            f"brier={book['brier']:.4f} macro-F1={book['macro_f1']:.4f}"
        )


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    skip = {"oof", "cv_splits", "frame"}
    return json.loads(json.dumps({k: v for k, v in metrics.items() if k not in skip}, default=_native))


def run_cascade_ab(
    h2h_frame: pd.DataFrame,
    h2h_features: list[str],
    h2h_oof: np.ndarray,
    dc_frame: pd.DataFrame,
    dc_features: list[str],
    dc_cv_splits: list[tuple[list[int], list[int]]],
) -> dict[str, Any]:
    """A/B a parita' di split e seed: dc base vs dc + P_OOF(h2h)."""
    h2h_oof_by_fixture = pd.Series(h2h_oof, index=h2h_frame["id_fixture"].to_numpy())
    frame = dc_frame.copy()
    frame["h2h_home_win_oof"] = frame["id_fixture"].map(h2h_oof_by_fixture)

    oof_index_all = sorted({idx for _, valid_idx in dc_cv_splits for idx in valid_idx})
    eligible_train = {int(i) for i in oof_index_all if pd.notna(frame["h2h_home_win_oof"].iloc[i])}
    if not eligible_train:
        return {"status": "skipped_no_h2h_oof_join", "n_eligible": 0}

    _assert_cascade_no_future_leakage(frame=frame, cv_splits=dc_cv_splits, oof_index_all=eligible_train)

    baseline_probs = _oof_probs(frame, dc_features, dc_cv_splits, restrict_train_to=eligible_train)
    cascade_cols = dc_features + ["h2h_home_win_oof"]
    cascade_probs = _oof_probs(frame, cascade_cols, dc_cv_splits, restrict_train_to=eligible_train)

    eval_index = [
        i
        for i in sorted(eligible_train)
        if np.isfinite(baseline_probs[i]) and np.isfinite(cascade_probs[i])
    ]
    if not eval_index:
        return {"status": "skipped_no_eval_rows", "n_eligible": len(eligible_train)}

    y = frame["y"].astype(int).to_numpy()
    p_market = frame["implied_prob_1x"].to_numpy() if "implied_prob_1x" in frame.columns else None
    idx = np.array(eval_index)
    market_slice = p_market[idx] if p_market is not None else None
    baseline_metrics = score_oof(y[idx], baseline_probs[idx], p_market=market_slice)
    cascade_metrics = score_oof(y[idx], cascade_probs[idx], p_market=market_slice)
    delta_selection = cascade_metrics["selection_score"] - baseline_metrics["selection_score"]
    delta_auc = float(cascade_metrics["auc"] - baseline_metrics["auc"])
    delta_macro_f1 = cascade_metrics["macro_f1"] - baseline_metrics["macro_f1"]

    adopted = (
        delta_selection > ADOPTION_DELTA_THRESHOLD
        and not cascade_metrics["degenerate"]
        and cascade_metrics["macro_f1"] >= baseline_metrics["macro_f1"]
    )
    verdict = (
        f"ADOTTATO: delta selection_score {delta_selection:+.4f} supera {ADOPTION_DELTA_THRESHOLD} "
        f"e il modello a cascata non e' degenere."
        if adopted
        else (
            f"SCARTATO: delta selection_score {delta_selection:+.4f} "
            f"(auc {delta_auc:+.4f}, macro-F1 {delta_macro_f1:+.4f}) non batte il dc indipendente. "
            "Stesso esito di phase3_cascading_ab_test.py: la cascata non si adotta."
        )
    )
    return {
        "status": "ok",
        "n_eligible_train_rows": len(eligible_train),
        "n_eval_rows": len(eval_index),
        "join_coverage": float(len(eligible_train) / max(1, len(oof_index_all))),
        "baseline_dc": baseline_metrics,
        "cascade_dc_plus_h2h_oof": cascade_metrics,
        "delta_selection_score": delta_selection,
        "delta_auc": delta_auc,
        "delta_macro_f1": delta_macro_f1,
        "adopted": adopted,
        "verdict": verdict,
    }


def _promotion_proposal(h2h_metrics: dict[str, Any], dc_metrics: dict[str, Any], cascade: dict[str, Any]) -> dict[str, Any]:
    """Propone la promozione SOLO se il modello non e' degenere e batte il
    baseline classe maggioritaria in modo non banale. Non registra niente."""

    def consider(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
        beats_majority = (metrics.get("accuracy_minus_majority_pp") or 0) >= MIN_ACCURACY_PP_FOR_PROMOTION
        ok = (not metrics.get("degenerate", True)) and beats_majority
        book = metrics.get("bookmaker_implied_baseline") or {}
        beats_book = (
            metrics.get("auc") is not None
            and book.get("auc") is not None
            and float(metrics["auc"]) > float(book["auc"])
        )
        return {
            "market": name,
            "propose_promotion": bool(ok),
            "beats_majority": bool(beats_majority),
            "beats_bookmaker_auc": bool(beats_book),
            "reason": (
                f"accuracy {metrics.get('accuracy_minus_majority_pp', 0):+.2f} pp sul majority, "
                f"macro-F1={metrics.get('macro_f1')}, minority_recall={metrics.get('minority_recall')}, "
                f"degenere={metrics.get('degenerate')}, "
                f"auc_modello={metrics.get('auc')} auc_bookmaker={book.get('auc')}"
            ),
        }

    proposals = [consider("h2h", h2h_metrics), consider("dc", dc_metrics)]
    if cascade.get("status") == "ok" and cascade.get("adopted"):
        proposals.append(consider("dc_cascade_h2h_oof", cascade["cascade_dc_plus_h2h_oof"]))
    return {
        "auto_promoted": False,
        "candidates": proposals,
        "next_step": (
            "Se propose_promotion=true, lanciare train_market(market, feature_columns=...) "
            "con save_model=True (resta candidate) e promuovere a mano. "
            "Questo script non ha salvato nessun pkl."
            if any(p["propose_promotion"] for p in proposals)
            else "Nessuna variante merita la promozione: fermarsi qui."
        ),
    }


def main() -> None:
    t0 = time.time()
    print("1) Audit chiavi Odds.h2h / Odds.dc a DB...")
    key_audit = inspect_odds_keys()
    print(f"   odds rows={key_audit['table_counts']}")
    print(f"   h2h slugs={key_audit['h2h']['slug_row_coverage']}")
    print(f"   dc slugs={key_audit['dc']['slug_row_coverage']}")

    service = FilterMarketService()
    print("\n2) build_dataset(h2h) sul dataset intero...")
    h2h_raw = service.build_dataset(market="h2h")
    print(f"   h2h grezzo: {len(h2h_raw)} righe, {h2h_raw.shape[1]} colonne")
    h2h = _prepare_frame(h2h_raw, H2H_REQUIRED_ODDS)
    h2h_features = h2h_feature_columns(h2h)
    print(f"   h2h dopo filtro quote canoniche: {len(h2h)} righe, {len(h2h_features)} feature")
    print(f"   y mean (P casa)={h2h['y'].mean():.4f}")

    print("\n3) build_dataset(dc) sul dataset intero...")
    dc_raw = service.build_dataset(market="dc")
    print(f"   dc grezzo: {len(dc_raw)} righe, {dc_raw.shape[1]} colonne")
    dc = _prepare_frame(dc_raw, DC_REQUIRED_ODDS)
    dc_features = dc_feature_columns(dc)
    print(f"   dc dopo filtro quote canoniche 1X+X2: {len(dc)} righe, {len(dc_features)} feature")
    print(f"   y mean (P 1X)={dc['y'].mean():.4f}")

    print("\n4) Walk-forward OOF h2h (quote per esito + mean_stats)...")
    h2h_eval = evaluate_market(h2h, h2h_features, market_prob_col="prob_norm_home")
    h2h_metrics = h2h_eval["metrics"]
    _print_metrics("h2h  quote per esito, dataset intero", h2h_metrics)

    print("\n5) Walk-forward OOF dc (quote per esito + mean_stats)...")
    dc_eval = evaluate_market(dc, dc_features, market_prob_col="implied_prob_1x")
    dc_metrics = dc_eval["metrics"]
    _print_metrics("dc  quote per esito, dataset intero", dc_metrics)

    print("\n6) A/B cascata dc vs dc + P_OOF(h2h), stesso split/seed...")
    cascade = run_cascade_ab(
        h2h_frame=h2h,
        h2h_features=h2h_features,
        h2h_oof=h2h_eval["oof"],
        dc_frame=dc,
        dc_features=dc_features,
        dc_cv_splits=dc_eval["cv_splits"],
    )
    if cascade.get("status") == "ok":
        _print_metrics("dc BASE (stesse righe della cascata)", cascade["baseline_dc"])
        _print_metrics("dc + P_OOF(h2h)", cascade["cascade_dc_plus_h2h_oof"])
        print(f"\nVERDETTO CASCATA: {cascade['verdict']}")
    else:
        print(f"Cascata non eseguibile: {cascade}")

    proposal = _promotion_proposal(h2h_metrics, dc_metrics, cascade)
    print("\n7) Promozione: NESSUNA (auto_promoted=false)")
    for cand in proposal["candidates"]:
        print(f"   {cand['market']}: propose={cand['propose_promotion']}  {cand['reason']}")
    print(f"   {proposal['next_step']}")

    payload = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "odds_key_audit": key_audit,
        "h2h": _serializable_metrics(h2h_metrics),
        "dc": _serializable_metrics(dc_metrics),
        "cascade_ab": {
            k: _serializable_metrics(v) if isinstance(v, dict) else _native(v)
            for k, v in cascade.items()
            if k not in {"baseline_dc", "cascade_dc_plus_h2h_oof"}
        }
        | (
            {
                "baseline_dc": _serializable_metrics(cascade["baseline_dc"]),
                "cascade_dc_plus_h2h_oof": _serializable_metrics(cascade["cascade_dc_plus_h2h_oof"]),
            }
            if cascade.get("status") == "ok"
            else {}
        ),
        "promotion": proposal,
        "notes": {
            "h2h_target": "y=1 vittoria casa, y=0 pareggio+trasferta (NON e' un 1X2 a 3 classi)",
            "dc_target": "y=1 esito 1X (casa o pareggio), y=0 vittoria trasferta",
            "oof": "walk-forward expanding window, mai in-sample",
            "promotion": "nessun pkl salvato, nessuno stage cambiato",
        },
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_native)
    print(f"\nSalvato in {OUTPUT_PATH} ({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
