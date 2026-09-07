"""Under/Over multi-linea consolidato (MARKET-04): 1.5, 2.5, 3.5, 4.5.

Confronta 3 modi di ottenere P(Over 1.5/2.5/3.5/4.5) COERENTI tra loro
(stesso dataset, stesso walk-forward temporale), senza duplicare logica
gia' validata altrove:

- "binary_independent": 4 classificatori binari scorrelati (stessa logica
  gia' addestrabile via `train_multi_market.py`/`DirectMarketExpert` per i
  mercati 'under_over_1_5/2_5/3_5/4_5', EXP-05), uno per soglia. NESSUNA
  garanzia di monotonicita' tra le soglie (sono modelli indipendenti: il
  rumore statistico puo' violare l'ordine).
- "hierarchical": UN SOLO classificatore multiclasse sui bin di gol totali
  (0-1, 2, 3, 4, 5+, i cui confini coincidono esattamente con le 4 soglie);
  P(Over t) e' la somma cumulata (dall'alto) delle probabilita' dei bin.
  Monotono per costruzione, perche' le 4 soglie condividono la STESSA
  distribuzione sottostante (da cui il nome "hierarchical").
- "goal_distribution": Poisson con lambda dai rating point-in-time
  (`TeamStrengthExpert` EXP-01 + `GoalDistributionExpert.estimate_lambdas_from_ratings`,
  EXP-02) via `poisson_over_probabilities`. Monotono per costruzione (gia'
  garantito nel modulo EXP-02).

I 3 approcci vengono valutati sullo STESSO walk-forward temporale
(`expanding_window_splits`) e sulle STESSE fixture (out-of-fold), per un
confronto onesto. La monotonicita' e' comunque un acceptance criteria
OBBLIGATORIO sull'output finale: indipendentemente da quale approccio vince
il confronto, l'output esposto viene sempre passato attraverso
`enforce_monotonic_over_probabilities` prima di essere restituito.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.ml.evaluation.multiclass_probability_metrics import reorder_probabilities_to_labels
from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.experts.goal_distribution.goal_distribution_expert import (
    LEAGUE_AVG_GOALS_DEFAULT,
    GoalDistributionExpert,
    poisson_over_probabilities,
)
from src.ml.experts.team_strength.team_strength_expert import TeamStrengthExpert
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.utils import convert_orm_match_to_dict

MARKET_NAME = "totals"
THRESHOLDS: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)
REFERENCE_ODDS_MARKET = "under_over_2_5"
APPROACHES: tuple[str, ...] = ("binary_independent", "hierarchical", "goal_distribution")
GOAL_BIN_LABELS: tuple[int, ...] = (0, 1, 2, 3, 4)

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]
_NON_FEATURE_TARGET_COLUMNS = {"total_goals", "goal_bin"}

_RF_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced", n_jobs=-1)


def _threshold_label(threshold: float) -> str:
    return f"over_{str(float(threshold)).replace('.', '_')}"


THRESHOLD_LABELS: tuple[str, ...] = tuple(_threshold_label(t) for t in THRESHOLDS)


# ---------------------------------------------------------------------------
# Funzioni pure: bin ordinali, monotonicita', derivazione Poisson.
# ---------------------------------------------------------------------------


def total_goals_to_bin(total_goals: Any, thresholds: tuple[float, ...] = THRESHOLDS) -> np.ndarray:
    """Bin ordinali i cui confini coincidono ESATTAMENTE con le soglie O/U.

    bin(totale) = numero di soglie che il totale supera (0..len(thresholds)).
    Con thresholds=(1.5,2.5,3.5,4.5): totale 0/1 -> bin 0; 2 -> bin 1; 3 -> bin 2;
    4 -> bin 3; >=5 -> bin 4. Nessuna approssimazione: ogni soglia resta un
    confine esatto tra due bin contigui.
    """
    goals = np.asarray(total_goals, dtype=float).reshape(-1)
    sorted_thresholds = sorted(set(float(t) for t in thresholds))
    bins = np.zeros(goals.shape[0], dtype=int)
    for th in sorted_thresholds:
        bins += (goals > th).astype(int)
    return bins


def enforce_monotonic_over_probabilities(
    probabilities_by_threshold: dict[str, Any],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, np.ndarray]:
    """Garantisce P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5) riga per riga.

    Acceptance criteria OBBLIGATORIO indipendente dall'approccio che ha
    generato le probabilita' grezze: proiezione per cumulative minimum,
    stesso principio gia' usato in `poisson_over_probabilities` (EXP-02).
    """
    sorted_thresholds = sorted(set(float(t) for t in thresholds))
    result: dict[str, np.ndarray] = {}
    previous: Optional[np.ndarray] = None
    for th in sorted_thresholds:
        label = _threshold_label(th)
        current = np.asarray(probabilities_by_threshold[label], dtype=float)
        if previous is not None:
            current = np.minimum(current, previous)
        previous = current
        result[label] = current
    return result


def over_probabilities_from_bin_probabilities(
    bin_probabilities: Any,
    bin_classes: Sequence[Any],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, np.ndarray]:
    """Deriva P(Over t) per ogni soglia dalla distribuzione multiclasse sui
    bin di gol totali: somma cumulata dall'alto. Monotono per costruzione
    (stesso vettore riga che somma a 1)."""
    ordered = reorder_probabilities_to_labels(
        estimator_classes=list(bin_classes),
        raw_probabilities=np.asarray(bin_probabilities, dtype=float),
        class_labels=list(GOAL_BIN_LABELS),
    )
    reverse_cumsum = np.cumsum(ordered[:, ::-1], axis=1)[:, ::-1]  # reverse_cumsum[:, k] = P(bin >= k)

    result: dict[str, np.ndarray] = {}
    for position, th in enumerate(sorted(set(float(t) for t in thresholds))):
        bin_min = position + 1
        result[_threshold_label(th)] = reverse_cumsum[:, bin_min]
    return result


def score_distribution_over_probabilities(
    home_lambda: float,
    away_lambda: float,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, float]:
    """P(Over t) per ogni soglia via Poisson (EXP-02), lambda combinato
    home+away (somma di due Poisson indipendenti e' ancora Poisson)."""
    combined_lambda = float(home_lambda) + float(away_lambda)
    frame = poisson_over_probabilities(lam=[combined_lambda], thresholds=thresholds)
    return {col: float(frame.iloc[0][col]) for col in frame.columns}


def _class1_probability(raw_proba: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    ordered = reorder_probabilities_to_labels(estimator_classes=list(classes), raw_probabilities=raw_proba, class_labels=[0, 1])
    return ordered[:, 1]


# ---------------------------------------------------------------------------
# Dataset builder: riusa SOLO mattoni gia' validati (nessuna nuova logica di
# feature extraction) - FilterMarketService (EXP-05) + TeamStrengthExpert
# (EXP-01) + GoalDistributionExpert (EXP-02).
# ---------------------------------------------------------------------------


def build_totals_frame_from_records(
    matches: list[dict[str, Any]],
    reference_market: str = REFERENCE_ODDS_MARKET,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats + target reale) allineato sulle 4
    soglie O/U. Le feature (odds+mean_stats) usano il mercato di riferimento
    'under_over_2_5' (il piu' liquido/standard): il TARGET reale invece e'
    SEMPRE calcolato dal numero di gol effettivi (mai dalle quote), quindi e'
    corretto per costruzione indipendentemente dalla soglia di riferimento
    scelta per le feature odds.
    """
    service = FilterMarketService()
    rows: list[dict[str, Any]] = []
    for match in matches:
        row = service._build_row(match=match, market=reference_market, with_target=False)
        if not row:
            continue

        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            continue
        home_ft = stat_home.get("score_ft")
        away_ft = stat_away.get("score_ft")
        if home_ft is None or away_ft is None:
            continue

        total_goals = int(home_ft) + int(away_ft)
        row["total_goals"] = total_goals
        row["goal_bin"] = int(total_goals_to_bin(np.array([total_goals]), thresholds=thresholds)[0])
        for th in thresholds:
            row[f"y_{_threshold_label(th)}"] = int(total_goals > th)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


def _goal_distribution_probabilities_from_ratings(
    ratings_frame: pd.DataFrame,
    league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """P(Over t) via score distribution Poisson (EXP-02) da rating
    point-in-time gia' calcolati (EXP-01): nessun training, nessun leakage
    aggiuntivo (i rating usano gia' solo lo storico anteriore alla fixture)."""
    columns = ["id_fixture"] + [_threshold_label(t) for t in thresholds]
    if ratings_frame.empty:
        return pd.DataFrame(columns=columns)

    home_lambdas: list[float] = []
    away_lambdas: list[float] = []
    for _, row in ratings_frame.iterrows():
        home_lambda, away_lambda = GoalDistributionExpert.estimate_lambdas_from_ratings(
            home_attack_rating=row["home_team_attack_rating"],
            away_defense_rating=row["away_team_defense_rating"],
            away_attack_rating=row["away_team_attack_rating"],
            home_defense_rating=row["home_team_defense_rating"],
            league_avg_goals=league_avg_goals,
        )
        home_lambdas.append(home_lambda)
        away_lambdas.append(away_lambda)

    combined_lambda = np.asarray(home_lambdas, dtype=float) + np.asarray(away_lambdas, dtype=float)
    over_probs = poisson_over_probabilities(lam=combined_lambda, thresholds=thresholds)
    over_probs.insert(0, "id_fixture", ratings_frame["id_fixture"].astype(int).to_numpy())
    return over_probs


def build_totals_evaluation_frame(
    matches: list[dict[str, Any]],
    reference_market: str = REFERENCE_ODDS_MARKET,
    thresholds: tuple[float, ...] = THRESHOLDS,
    league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
) -> tuple[pd.DataFrame, list[str]]:
    """Ritorna (frame, feature_columns): STESSE fixture per i 3 approcci,
    ordinate temporalmente, con feature (odds+mean_stats), target (goal_bin,
    y_over_*) e probabilita' goal_distribution (prefisso 'gd_') gia' allineate
    per id_fixture."""
    reference_frame = build_totals_frame_from_records(matches, reference_market=reference_market, thresholds=thresholds)
    if reference_frame.empty:
        return pd.DataFrame(), []

    ratings_frame = TeamStrengthExpert().build_ratings_dataset(matches)
    gd_frame = _goal_distribution_probabilities_from_ratings(
        ratings_frame=ratings_frame, league_avg_goals=league_avg_goals, thresholds=thresholds
    )
    if gd_frame.empty:
        return pd.DataFrame(), []
    gd_frame = gd_frame.rename(columns={col: f"gd_{col}" for col in gd_frame.columns if col != "id_fixture"})

    merged = reference_frame.merge(gd_frame, on="id_fixture", how="inner")
    if merged.empty:
        return pd.DataFrame(), []

    merged = merged.sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    excluded = set(_META_COLUMNS) | _NON_FEATURE_TARGET_COLUMNS
    excluded |= {f"y_{label}" for label in THRESHOLD_LABELS}
    excluded |= {f"gd_{label}" for label in THRESHOLD_LABELS}
    feature_columns = [col for col in reference_frame.columns if col not in excluded]
    return merged, feature_columns


# ---------------------------------------------------------------------------
# Walk-forward OOF per i 3 approcci (stessi `cv_splits` per tutti).
# ---------------------------------------------------------------------------


def _binary_independent_oof(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: list[tuple[list[int], list[int]]],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, np.ndarray]:
    """OOF P(Over t) con un classificatore INDIPENDENTE per ciascuna soglia."""
    n = len(frame)
    result = {_threshold_label(t): np.full(n, np.nan) for t in thresholds}
    X = frame[feature_columns]

    for th in thresholds:
        label = _threshold_label(th)
        y = frame[f"y_{label}"].astype(int)
        for train_idx, valid_idx in cv_splits:
            if not train_idx or not valid_idx:
                continue
            if y.iloc[train_idx].nunique() < 2:
                continue
            model = RandomForestClassifier(**_RF_KWARGS)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[valid_idx])
            result[label][valid_idx] = _class1_probability(proba, model.classes_)

    return result


def _hierarchical_oof(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: list[tuple[list[int], list[int]]],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, np.ndarray]:
    """OOF P(Over t) per TUTTE le soglie da UN SOLO modello multiclasse sui
    bin di gol totali (condivide la stessa distribuzione tra le 4 soglie)."""
    n = len(frame)
    result = {_threshold_label(t): np.full(n, np.nan) for t in thresholds}
    X = frame[feature_columns]
    y = frame["goal_bin"].astype(int)

    for train_idx, valid_idx in cv_splits:
        if not train_idx or not valid_idx:
            continue
        if y.iloc[train_idx].nunique() < 2:
            continue
        model = RandomForestClassifier(**_RF_KWARGS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        per_threshold = over_probabilities_from_bin_probabilities(proba, model.classes_, thresholds=thresholds)
        for label, values in per_threshold.items():
            result[label][valid_idx] = values

    return result


# ---------------------------------------------------------------------------
# Report benchmark (acceptance criteria) + selezione + monotonicita' finale.
# ---------------------------------------------------------------------------


@dataclass
class TotalsBenchmarkReport:
    """Report comparativo (acceptance criteria): metriche per ciascuna
    soglia/approccio, punteggio aggregato per approccio, il migliore
    selezionato e le probabilita' finali (SEMPRE proiettate monotone).

    `best_approach_by_threshold` (campo ADDITIVO, non tocca la selezione
    del vincitore GLOBALE ne' la proiezione monotona sotto): oltre al
    vincitore aggregato sulle 4 soglie (`best_approach`, INVARIATO - resta
    l'unico usato per decidere cosa salvare nel registry), espone anche
    quale approccio avrebbe il `selection_score` piu' alto PER CIASCUNA
    soglia singolarmente. Serve a rispondere alla domanda "hierarchical
    perde sistematicamente contro binary_independent su qualche soglia
    specifica?" senza dover rieseguire il benchmark isolando le soglie."""

    threshold_metrics: dict[str, dict[str, dict[str, Any]]]
    approach_aggregate_scores: dict[str, float]
    best_approach: str
    monotonicity_violations_before_projection: dict[str, int]
    final_probabilities: dict[str, np.ndarray] = field(default_factory=dict)
    evaluated_index: list[int] = field(default_factory=list)
    best_approach_by_threshold: dict[str, str] = field(default_factory=dict)


def _count_monotonicity_violations(probs_by_threshold: dict[str, np.ndarray], thresholds: tuple[float, ...]) -> int:
    ordered_arrays = [np.asarray(probs_by_threshold[_threshold_label(t)], dtype=float) for t in sorted(set(float(t) for t in thresholds))]
    stacked = np.vstack(ordered_arrays)
    diffs = np.diff(stacked, axis=0)  # deve essere <= 0 (non crescente al crescere della soglia)
    return int(np.sum(diffs > 1e-9))


def benchmark_totals_approaches(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: list[tuple[list[int], list[int]]],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> TotalsBenchmarkReport:
    """Confronta binary_independent vs hierarchical vs goal_distribution
    sullo STESSO walk-forward; sceglie il migliore per metriche
    probabilistiche + betting (`champion_probability_score`) e proietta
    SEMPRE l'output vincente su probabilita' monotone."""
    oof_index = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})
    if not oof_index:
        raise ValueError("Nessun indice OOF disponibile: walk-forward non valido")

    binary_probs = _binary_independent_oof(frame, feature_columns, cv_splits, thresholds=thresholds)
    hierarchical_probs = _hierarchical_oof(frame, feature_columns, cv_splits, thresholds=thresholds)
    goal_distribution_probs = {label: frame[f"gd_{label}"].to_numpy(dtype=float) for label in THRESHOLD_LABELS}

    raw_by_approach: dict[str, dict[str, np.ndarray]] = {
        "binary_independent": binary_probs,
        "hierarchical": hierarchical_probs,
        "goal_distribution": goal_distribution_probs,
    }

    # Righe con OOF valido per TUTTI gli approcci/soglie (confronto onesto sullo stesso sample).
    valid_mask = np.ones(len(oof_index), dtype=bool)
    for probs_by_threshold in raw_by_approach.values():
        for values in probs_by_threshold.values():
            arr = np.asarray(values, dtype=float)[oof_index]
            valid_mask &= ~np.isnan(arr)
    oof_index = [idx for idx, keep in zip(oof_index, valid_mask) if keep]
    if not oof_index:
        raise ValueError("Nessuna riga OOF valida per tutti gli approcci: dataset insufficiente per il confronto")

    monotonicity_violations: dict[str, int] = {}
    threshold_metrics: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in THRESHOLD_LABELS}
    approach_scores_by_threshold: dict[str, list[float]] = {a: [] for a in APPROACHES}

    for approach, probs_by_threshold in raw_by_approach.items():
        restricted = {label: np.asarray(values, dtype=float)[oof_index] for label, values in probs_by_threshold.items()}
        monotonicity_violations[approach] = _count_monotonicity_violations(restricted, thresholds=thresholds)

        for th in sorted(set(float(t) for t in thresholds)):
            label = _threshold_label(th)
            y_true = frame[f"y_{label}"].astype(int).to_numpy()[oof_index]
            probs = restricted[label]
            metrics = compute_probability_metrics(y_true=y_true, probabilities=probs, n_bins=10)
            predicted = (probs >= 0.5).astype(int)
            f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
            selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
            threshold_metrics[label][approach] = {**metrics, "f1_weighted": f1_weighted, "selection_score": selection_score}
            approach_scores_by_threshold[approach].append(selection_score)

    approach_aggregate_scores = {
        approach: float(np.mean(scores)) for approach, scores in approach_scores_by_threshold.items() if scores
    }
    best_approach = max(approach_aggregate_scores.items(), key=lambda kv: kv[1])[0]

    # Vincitore per-soglia (ADDITIVO, cfr. docstring `TotalsBenchmarkReport`):
    # NON influenza `best_approach`/`final_probabilities` sotto, che restano
    # calcolati SOLO sull'aggregato delle 4 soglie come gia' validato.
    best_approach_by_threshold = {
        label: max(approaches.items(), key=lambda kv: kv[1]["selection_score"])[0]
        for label, approaches in threshold_metrics.items()
        if approaches
    }

    winning_raw = {label: np.asarray(raw_by_approach[best_approach][label], dtype=float)[oof_index] for label in THRESHOLD_LABELS}
    final_probabilities = enforce_monotonic_over_probabilities(winning_raw, thresholds=thresholds)

    return TotalsBenchmarkReport(
        threshold_metrics=threshold_metrics,
        approach_aggregate_scores=approach_aggregate_scores,
        best_approach=best_approach,
        monotonicity_violations_before_projection=monotonicity_violations,
        final_probabilities=final_probabilities,
        evaluated_index=oof_index,
        best_approach_by_threshold=best_approach_by_threshold,
    )


# ---------------------------------------------------------------------------
# Orchestrazione end-to-end (dataset reale -> report -> salvataggio 'candidate').
# ---------------------------------------------------------------------------


@dataclass
class TotalsBenchmarkRunResult:
    market: str
    rows: int
    status: str
    best_approach: Optional[str]
    details: dict[str, Any]


def _fit_final_binary_independent(frame: pd.DataFrame, feature_columns: list[str], thresholds: tuple[float, ...]) -> dict[str, Any]:
    X = frame[feature_columns]
    bundle: dict[str, Any] = {}
    for th in thresholds:
        label = _threshold_label(th)
        y = frame[f"y_{label}"].astype(int)
        model = RandomForestClassifier(**_RF_KWARGS)
        model.fit(X, y)
        bundle[label] = model
    return bundle


def _fit_final_hierarchical(frame: pd.DataFrame, feature_columns: list[str]) -> Any:
    X = frame[feature_columns]
    y = frame["goal_bin"].astype(int)
    model = RandomForestClassifier(**_RF_KWARGS)
    model.fit(X, y)
    return model


def _aggregate_metrics_for_winning_approach(report: TotalsBenchmarkReport, rows: int) -> dict[str, Any]:
    """Metriche aggregate (media sulle 4 soglie) del solo approccio vincente,
    con le chiavi STANDARD gia' cercate da `promotion_policy._resolve_metric`
    (`selection_score`/`log_loss`/`brier`/`ece`/`auc`/`sample_size`) - PRIMA
    l'unica metrica salvata era `approach_aggregate_score` (chiave non
    riconosciuta dal gate, che quindi falliva SEMPRE per assenza di
    `sample_size`): nessuna logica di selezione/monotonicita' cambiata qui,
    solo le metriche del vincitore gia' calcolato vengono ESPOSTE in modo
    piu' completo (Fase 4 del task Under/Over: "selection_score, log_loss,
    brier, ece, auc, f1_weighted... rows")."""
    per_threshold = [report.threshold_metrics[label][report.best_approach] for label in THRESHOLD_LABELS]
    aucs = [m["auc"] for m in per_threshold if m.get("auc") is not None]
    return {
        "approach_aggregate_score": report.approach_aggregate_scores[report.best_approach],
        "selection_score": report.approach_aggregate_scores[report.best_approach],
        "log_loss": float(np.mean([m["log_loss"] for m in per_threshold])),
        "brier": float(np.mean([m["brier"] for m in per_threshold])),
        "ece": float(np.mean([m["ece"] for m in per_threshold])),
        "auc": float(np.mean(aucs)) if aucs else None,
        "f1_weighted": float(np.mean([m["f1_weighted"] for m in per_threshold])),
        "rows": int(rows),
        "sample_size": int(rows),
    }


def _save_winning_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    report: TotalsBenchmarkReport,
    thresholds: tuple[float, ...],
) -> Optional[dict[str, Any]]:
    """Salva come 'candidate' (mai 'production' automatica, vedi OPS-02) SOLO
    se il vincitore introduce un modello nuovo (hierarchical/binary_independent).
    Per goal_distribution non c'e' nulla di nuovo da salvare: e' un calcolo
    deterministico gia' gestito da EXP-01/EXP-02."""
    if report.best_approach == "goal_distribution":
        return None

    if report.best_approach == "hierarchical":
        artifact: Any = _fit_final_hierarchical(frame, feature_columns)
        filename = f"{MARKET_NAME}_hierarchical_champion.pkl"
    else:
        artifact = _fit_final_binary_independent(frame, feature_columns, thresholds)
        filename = f"{MARKET_NAME}_binary_independent_champion.pkl"

    model_path = os.path.abspath(os.path.join("best_models", filename))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)

    registry = ModelRegistry()
    return registry.register(
        model_path=model_path,
        market=MARKET_NAME,
        model_name=report.best_approach,
        feature_names=feature_columns,
        metrics=_aggregate_metrics_for_winning_approach(report=report, rows=len(frame)),
        extra={
            "thresholds": list(thresholds),
            "threshold_metrics": report.threshold_metrics,
            "best_approach_by_threshold": report.best_approach_by_threshold,
            "monotonicity_violations_before_projection": report.monotonicity_violations_before_projection,
        },
        stage="candidate",
    )


def run_totals_benchmark(
    matches: list[dict[str, Any]],
    thresholds: tuple[float, ...] = THRESHOLDS,
    reference_market: str = REFERENCE_ODDS_MARKET,
    league_avg_goals: float = LEAGUE_AVG_GOALS_DEFAULT,
    save_model: bool = True,
) -> TotalsBenchmarkRunResult:
    """Costruisce il dataset reale, esegue `benchmark_totals_approaches` e,
    se richiesto, registra il modello vincente (quando esiste) come
    'candidate'."""
    frame, feature_columns = build_totals_evaluation_frame(
        matches=matches, reference_market=reference_market, thresholds=thresholds, league_avg_goals=league_avg_goals
    )
    if frame.empty:
        return TotalsBenchmarkRunResult(market=MARKET_NAME, rows=0, status="skipped_no_data", best_approach=None, details={})

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(
        frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
    )
    if not cv_splits:
        return TotalsBenchmarkRunResult(
            market=MARKET_NAME,
            rows=len(frame),
            status="skipped_insufficient_rows_for_temporal_cv",
            best_approach=None,
            details={},
        )

    report = benchmark_totals_approaches(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits, thresholds=thresholds)

    run_metadata = None
    if save_model:
        run_metadata = _save_winning_model(frame=frame, feature_columns=feature_columns, report=report, thresholds=thresholds)

    return TotalsBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        best_approach=report.best_approach,
        details={
            "threshold_metrics": report.threshold_metrics,
            "approach_aggregate_scores": report.approach_aggregate_scores,
            "best_approach_by_threshold": report.best_approach_by_threshold,
            "monotonicity_violations_before_projection": report.monotonicity_violations_before_projection,
            "run": run_metadata,
        },
    )


def run_totals_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    reference_market: str = REFERENCE_ODDS_MARKET,
    save_model: bool = True,
) -> TotalsBenchmarkRunResult:
    """Variante DB reale: filtra le fixture con statistiche/mean_statistics/odds
    disponibili (stesso requisito del dataset diretto)."""
    match_repo = MatchRepository()
    filters: dict[str, Any] = {
        "statistics": "not None",
        "mean_statistics": "not None",
        "odds": "not None",
        "status": ["FT"],
    }
    if seasons:
        filters["season"] = seasons
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))

    return run_totals_benchmark(matches=matches, reference_market=reference_market, save_model=save_model)
