"""Corners O/U specializzato con linea configurabile (MARKET-05).

Generalizza il mercato 'corners' esistente (soglia FISSA a 9.5, cioe'
`total_corners >= 10`, in `FilterMarketService._label_by_market` — non
modificata qui per preservare compatibilita' con quel mercato legacy)
introducendo la linea come PARAMETRO esplicito (es. 8.5, 9.5, 10.5, 11.5),
stesso pattern gia' adottato per U/O gol in
`src/ml/markets/totals/totals_market.py` (MARKET-04): il TARGET reale (dal
risultato, mai dalle quote) e' calcolabile per qualunque linea, mentre la
FONTE feature (odds + mean_statistics) resta quella gia' disponibile nella
colonna `Odds.corners`.

Aggiunge:
- feature DEDICATE ai corner (oltre alle generiche odds/mean_stats gia'
  estratte da `FilterMarketService`): media storica corner fatti/concessi
  per squadra (`mean_statistics['Corner Kicks']`), totale medio atteso e
  differenziale — nessuna nuova query DB, solo un secondo utilizzo mirato
  di un campo gia' disponibile. NESSUNA feature arbitro (a differenza di
  Cards, MARKET-06): discusso e confermato esplicitamente con l'operatore
  il 2026-09-12 - l'arbitro non ha un'influenza diretta/nota sul numero di
  calci d'angolo, a differenza dei cartellini;
- calibrazione (Platt/isotonic) via `CalibrationService` (ML-06, gia'
  esistente, non duplicata) per il modello di CIASCUNA linea;
- report con metriche PER LINEA, ora COMPLETO (acceptance criteria +
  richiesta esplicita "tutte le metriche possibili", 2026-09-12): oltre a
  log loss/Brier/ECE/AUC pre/post calibrazione (gia' esistenti), ogni linea
  espone anche un report di classificazione completo (accuracy, confusion
  matrix, precision/recall/F1 per classe e pesati, curva ROC+PR, soglia
  ottimale di Youden) via `classification_report.py` (NUOVO, condiviso e
  identico a quello usato da cards_market.py, mai duplicato).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.ml.calibration.calibration_service import CalibrationResult, CalibrationService
from src.ml.evaluation.classification_report import compute_full_classification_report
from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.markets.totals.totals_market import _count_monotonicity_violations, enforce_monotonic_over_probabilities
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import _select_champion_via_model_search
from src.service_ia.utility.utils import convert_orm_match_to_dict

MARKET_NAME = "corners"
ODDS_MARKET = "corners"  # colonna Odds.corners esistente, riusata come fonte feature quote
DEFAULT_LINES: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5)

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]


def _line_label(line: float) -> str:
    return f"line_{str(float(line)).replace('.', '_')}"


def label_corners_over(total_corners: Any, line: float) -> Any:
    """Target reale (dal risultato, MAI dalle quote): 1 se corner totali > line.

    Accetta sia uno scalare sia un array; la linea e' un PARAMETRO libero,
    non una soglia hardcoded (acceptance criteria 'Linee configurabili')."""
    result = (np.asarray(total_corners, dtype=float) > float(line)).astype(int)
    return result if result.shape else int(result)


def _corner_dedicated_features(match: dict[str, Any]) -> dict[str, float]:
    """Feature DEDICATE ai corner: media storica corner per squadra (gia'
    disponibile in `mean_statistics['Corner Kicks']`, calcolata da
    `download_match_service.py`), totale medio atteso e differenziale.
    Nessuna nuova query DB: riusa `_resolve_mean_stats` gia' esistente."""
    mean_home, mean_away = FilterMarketService._resolve_mean_stats(match)
    if not mean_home or not mean_away:
        return {}

    home_mean = FilterMarketService._safe_float(mean_home.get("Corner Kicks"))
    away_mean = FilterMarketService._safe_float(mean_away.get("Corner Kicks"))

    return {
        "corner_mean_home_dedicated": home_mean,
        "corner_mean_away_dedicated": away_mean,
        "corner_mean_total_dedicated": home_mean + away_mean,
        "corner_mean_diff_dedicated": home_mean - away_mean,
    }


# Quote placeholder trovate a DB (2026-09-16, report_corners_8_5_passo1.md):
# BetMGM, BetRivers, Bovada quotano lo STESSO valore fisso Over/Under 8.5 su
# migliaia di fixture completamente diverse (overround identico su 5.363
# partite distinte per ciascuno) - non compatibile con quote di mercato
# reali, coprono ~60% delle righe e sporcano odds_mean/overround aggregati.
CONTAMINATED_BOOKMAKERS = frozenset({"BetMGM", "BetRivers", "Bovada"})


def _line_specific_odds_features(match: dict[str, Any], odds_market: str, lines: tuple[float, ...]) -> dict[str, Any]:
    """Feature quote SEPARATE per linea (2026-09-12 - vedi
    `FilterMarketService._extract_line_specific_odds_features`): a
    differenza degli Under/Over gol (dove l'ingestion separa gia' le quote
    per soglia), 'Corners Over Under' mette TUTTE le linee in un unico
    bucket piatto - qui si estrae, PER OGNI linea configurata, solo le
    quote di quella linea (nome colonna con suffisso `_{line_label}`,
    es. `odds_mean_line_8_5`), cosi' il modello di ciascuna linea vede
    SOLO le quote di mercato pertinenti a se stesso."""
    odds_list = match.get("odds") or []
    if not odds_list:
        return {}
    market_odds = (odds_list[0] or {}).get(odds_market)
    if not isinstance(market_odds, dict) or not market_odds:
        return {}

    features: dict[str, Any] = {}
    for line in lines:
        exclude = CONTAMINATED_BOOKMAKERS if odds_market == "corners" else frozenset()
        line_features = FilterMarketService._extract_line_specific_odds_features(
            market_odds, line, exclude_bookmakers=exclude
        )
        for key, value in line_features.items():
            features[f"{key}_{_line_label(line)}"] = value
    return features


def build_corners_prediction_row(
    match: dict[str, Any],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
) -> Optional[dict[str, Any]]:
    """Riga di feature per UNA fixture live/futura (mai training - nessun
    target, il risultato non esiste ancora). Stessa identica logica per-
    partita di `build_corners_frame_from_records` (odds pooled generiche +
    mean_stats + feature dedicate corner + quote line-specific per OGNI
    linea configurata), riusata qui senza duplicazione: entrambe le feature
    dedicate/quote sono funzioni PURE del singolo `match` (nessuna query
    storica necessaria, a differenza dell'arbitro per Cards), quindi
    calcolabili identiche sia in batch (training) sia per una fixture sola
    (serving) - stesso principio gia' seguito da
    `FilterMarketService.build_prediction_frames_from_match` per gli altri
    mercati. Ritorna `None` quando mancano le quote 'corners' (stesso
    requisito minimo del training)."""
    service = FilterMarketService()
    row = service._build_row(match=match, market=odds_market, with_target=False)
    if not row:
        return None
    row.update(_corner_dedicated_features(match))
    row.update(_line_specific_odds_features(match, odds_market, lines))
    return row


def build_corners_frame_from_records(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats generiche + feature dedicate corner +
    target reale PER OGNI linea) sulle fixture con quote 'corners' disponibili."""
    rows: list[dict[str, Any]] = []
    for match in matches:
        row = build_corners_prediction_row(match, lines=lines, odds_market=odds_market)
        if not row:
            continue

        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            continue
        home_corners = stat_home.get("corners")
        away_corners = stat_away.get("corners")
        if home_corners is None or away_corners is None:
            continue

        total_corners = int(home_corners) + int(away_corners)
        row["total_corners"] = total_corners
        for line in lines:
            row[f"y_{_line_label(line)}"] = int(label_corners_over(total_corners, line))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if fill_missing:
        frame = frame.fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    return frame


_ODDS_METRIC_KEYS = ["odds_count", "odds_mean", "odds_std", "odds_min", "odds_max"] + [f"odds_slot_{i}" for i in range(1, 11)]


def _line_specific_odds_columns(line: float, columns: Optional[Iterable[str]] = None) -> set[str]:
    """Colonne quote riferite a QUESTA linea.

    Con `columns` (le colonne reali del frame) riconosce anche le feature
    per esito introdotte il 2026-09-13 (`odds_mean_over_8_5`,
    `prob_norm_under_8_5`, `overround_line_8_5`, ...), che non derivano da
    `_ODDS_METRIC_KEYS` e quindi non sarebbero enumerabili a priori. Senza
    `columns` resta il comportamento originario (solo le legacy), usato dai
    test che verificano la convenzione di nomi.

    Senza questo riconoscimento le nuove colonne sfuggirebbero
    all'esclusione in `_feature_columns_for` e il modello di una linea
    vedrebbe le quote delle ALTRE linee - esattamente il leak che le
    quote per-linea erano nate per eliminare.
    """
    label = _line_label(line)
    legacy = {f"{key}_{label}" for key in _ODDS_METRIC_KEYS}
    if columns is None:
        return legacy
    # Due suffissi, perche' le feature per esito arrivano da due percorsi:
    # `_line_specific_odds_features` produce '..._over_8_5_line_8_5' (suffisso
    # completo), mentre `_build_row` sul bucket intero produce
    # '..._over_8_5' (la linea e' gia' dentro il nome dell'esito, senza il
    # prefisso 'line_'). Senza il secondo suffisso queste ultime sfuggirebbero
    # all'esclusione e la linea 8.5 vedrebbe le quote della 9.5.
    raw_label = label.replace("line_", "", 1)
    return legacy | {
        col for col in columns if col.endswith(f"_{label}") or col.endswith(f"_{raw_label}")
    }


def _feature_columns_for(
    frame: pd.DataFrame,
    lines: tuple[float, ...],
    active_line: Optional[float] = None,
    use_line_specific_odds: bool = True,
) -> list[str]:
    """Colonne feature per il training di UNA linea (`active_line`).

    Quote (2026-09-12, vedi `_line_specific_odds_features`): con
    `use_line_specific_odds=True` (default) usa SOLO le quote della linea
    attiva (`odds_*_line_X_Y`), escludendo sia le quote per-linea delle
    ALTRE linee sia le vecchie quote "pooled" (mischiano tutte le linee
    insieme - il problema che ha motivato questa feature). Con `False`
    (comportamento legacy, usato SOLO per il confronto A/B) torna alle
    quote pooled originarie."""
    y_columns = {f"y_{_line_label(line)}" for line in lines}
    excluded = set(_META_COLUMNS) | {"total_corners"} | y_columns

    all_line_odds_columns: set[str] = set()
    for line in lines:
        all_line_odds_columns |= _line_specific_odds_columns(line, frame.columns)

    if use_line_specific_odds and active_line is not None:
        active_columns = _line_specific_odds_columns(active_line, frame.columns)
        excluded |= (all_line_odds_columns - active_columns)
        # Quote pooled legacy (tutte le linee insieme) + `overround`
        # calcolato sull'intero bucket del mercato, che somma le
        # probabilita' implicite di TUTTE le linee: privo di senso per una
        # linea singola, a differenza di `overround_line_X_Y`.
        excluded |= set(_ODDS_METRIC_KEYS) | {"overround"}
    else:
        excluded |= all_line_odds_columns

    return [col for col in frame.columns if col not in excluded]


# ---------------------------------------------------------------------------
# Monotonicita' tra linee (2026-09-12, "le linee seguono lo stesso principio
# di Under/Over gol? Over 2.5 e' sicuramente anche Over 1.5" - stesso
# principio gia' in produzione per i gol, MARKET-04, `totals_market.py`):
# P(Over 8.5) >= P(Over 9.5) >= P(Over 10.5) >= P(Over 11.5) riga per riga
# sono eventi ANNIDATI sullo stesso conteggio totale, non variabili
# indipendenti - modelli indipendenti per linea NON garantiscono questo
# ordine da soli.
# ---------------------------------------------------------------------------

_RF_MONOTONICITY_KWARGS = dict(n_estimators=150, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")


def _line_independent_oof(
    frame: pd.DataFrame,
    lines: tuple[float, ...],
    cv_splits: list[tuple[list[int], list[int]]],
    use_line_specific_odds: bool = True,
) -> dict[str, np.ndarray]:
    """OOF P(Over linea) per OGNI linea con un RandomForest fisso INDIPENDENTE
    (diagnostico - NON il champion per linea scelto da
    `_select_champion_via_model_search`, troppo costoso per essere rifittato
    qui: stesso principio di disaccoppiamento gia' usato da
    `totals_market._binary_independent_oof`/dal confronto quote di questa
    stessa sessione). Array della lunghezza del frame (NaN dove non c'e'
    copertura OOF), cosi' le righe restano allineate tra le diverse linee."""
    n = len(frame)
    result = {_line_label(line): np.full(n, np.nan) for line in lines}
    for line in lines:
        label = _line_label(line)
        y_col = f"y_{label}"
        if y_col not in frame.columns:
            continue
        feature_columns = _feature_columns_for(frame, lines, active_line=line, use_line_specific_odds=use_line_specific_odds)
        X = frame[feature_columns]
        y = frame[y_col].astype(int)
        for train_idx, valid_idx in cv_splits:
            if not train_idx or not valid_idx or y.iloc[train_idx].nunique() < 2:
                continue
            pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(**_RF_MONOTONICITY_KWARGS)),
            ])
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            result[label][valid_idx] = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
    return result


def _monotonicity_line_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Metriche di qualita' probabilistica standard, PRIMA o DOPO la
    proiezione monotona - stesso set gia' usato ovunque nel progetto
    (`compute_probability_metrics`+`selection_score`), cosi' si puo'
    rispondere direttamente a "la proiezione peggiora le metriche?" invece
    di limitarsi a contare le violazioni corrette."""
    metrics = compute_probability_metrics(y_true=y_true, probabilities=probabilities, n_bins=10)
    predicted = (probabilities >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    accuracy = float((y_true == predicted).mean())
    selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "accuracy": accuracy, "f1_weighted": f1_weighted, "selection_score": selection_score}


def compute_monotonicity_report(
    frame: pd.DataFrame,
    lines: tuple[float, ...],
    cv_splits: list[tuple[list[int], list[int]]],
    use_line_specific_odds: bool = True,
) -> dict[str, Any]:
    """Quanto le predizioni indipendenti per linea rispettano l'ordine
    logico Over 8.5 >= Over 9.5 >= Over 10.5 >= Over 11.5, e proiezione che
    lo impone SEMPRE (`enforce_monotonic_over_probabilities`, riusata
    identica da `totals_market.py` - stesso principio, non duplicata).
    `metrics_by_line` (2026-09-12, "quindi le metriche dopo aver applicato
    questo fix quali sono?"): confronto PRIMA/DOPO la proiezione per ogni
    linea - la linea piu' bassa non cambia mai (e' l'ancora della proiezione
    cumulativa), le altre di solito MIGLIORANO (mai peggiorano di molto):
    la proiezione corregge violazioni logicamente impossibili, non aggiunge
    rumore."""
    sorted_lines = tuple(sorted(set(float(line) for line in lines)))
    if len(sorted_lines) < 2:
        return {"status": "not_applicable", "reason": "meno di 2 linee addestrate"}

    probs_by_line = _line_independent_oof(frame, sorted_lines, cv_splits, use_line_specific_odds=use_line_specific_odds)

    valid_mask = np.ones(len(frame), dtype=bool)
    for values in probs_by_line.values():
        valid_mask &= ~np.isnan(values)
    oof_index = np.where(valid_mask)[0]
    if oof_index.size == 0:
        return {"status": "no_oof_coverage"}

    restricted = {label: values[oof_index] for label, values in probs_by_line.items()}
    violations = _count_monotonicity_violations(restricted, thresholds=sorted_lines, label_fn=_line_label)
    n_pairs = int(oof_index.size * (len(sorted_lines) - 1))
    projected = enforce_monotonic_over_probabilities(restricted, thresholds=sorted_lines, label_fn=_line_label)

    per_line: dict[str, Any] = {}
    for line in sorted_lines:
        label = _line_label(line)
        y_col = f"y_{label}"
        if y_col not in frame.columns:
            continue
        y_true = frame[y_col].astype(int).to_numpy()[oof_index]
        per_line[label] = {
            "before_projection": _monotonicity_line_metrics(y_true, restricted[label]),
            "after_projection": _monotonicity_line_metrics(y_true, projected[label]),
        }

    return {
        "status": "ok",
        "method": "random_forest_fisso_per_linea (diagnostico, non il champion selezionato per linea)",
        "lines": list(sorted_lines),
        "n_rows_evaluated": int(oof_index.size),
        "violations_before_projection": violations,
        "violations_pct": float(violations / n_pairs) if n_pairs else 0.0,
        "metrics_by_line": per_line,
    }


@dataclass
class CornersLineTrainResult:
    """Esito ricerca+training+calibrazione per UNA linea specifica."""

    line: float
    sample_size: int
    feature_names: list[str]
    calibration: CalibrationResult
    # NUOVI (2026-09-12, "hai fatto grid search/ensemble/provato piu'
    # opzioni?"): trasparenza su COSA e' stato provato, non solo il
    # risultato finale - stesso principio gia' seguito da train_market().
    champion_name: str
    model_results: dict[str, Any]


def train_corners_line(
    frame: pd.DataFrame,
    line: float,
    cv_splits: list[tuple[list[int], list[int]]],
    lines_in_frame: tuple[float, ...] = DEFAULT_LINES,
    selection_method: str = "kbest",
    use_line_specific_odds: bool = True,
    search_strategy: str = "random",
) -> CornersLineTrainResult:
    """Addestra + calibra il modello per una singola linea, con validazione
    temporale OOF (mai split random).

    **Model search completo (2026-09-12)**: non piu' un singolo
    `RandomForestClassifier` a iperparametri fissi - riusa
    `_select_champion_via_model_search` (la STESSA ricerca gia' usata da
    `train_market()` per h2h/goal_no_goal/under_over_*, mai duplicata):
    grid search su logistic/random_forest/random_forest_smote, ensemble
    voting+stacking sui 2 migliori, selezione del champion per
    `selection_score`. Il champion e' poi calibrato come prima.

    `search_strategy="random"` (default QUI, diverso dal default "grid" di
    `_select_champion_via_model_search" - 2026-09-12, "forse e' meglio usare
    una random search per il momento"): la grid search esaustiva e' risultata
    troppo costosa su Corners/Cards (68 minuti solo per la suite di test su
    dati sintetici) - random search campiona un sottoinsieme delle
    combinazioni, stesso identico spazio di ricerca/criterio di selezione,
    costo molto piu' basso. Non tocca i mercati flagship (h2h/goal_no_goal/
    under_over_*), che continuano a passare da `train_market()` col default
    "grid" invariato."""
    label = f"y_{_line_label(line)}"
    if label not in frame.columns:
        raise ValueError(f"Linea non presente nel dataset: {line}")

    feature_columns = _feature_columns_for(
        frame, lines_in_frame, active_line=line, use_line_specific_odds=use_line_specific_odds
    )
    X = frame[feature_columns]
    y = frame[label].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Target a classe unica per la linea {line}: impossibile addestrare")

    season_series = frame["season"] if "season" in frame.columns else pd.Series([None] * len(frame))
    league_series = frame["league"] if "league" in frame.columns else pd.Series([None] * len(frame))

    search_result = _select_champion_via_model_search(
        X=X,
        y=y,
        cv_splits=cv_splits,
        market=f"{MARKET_NAME}_{_line_label(line)}",
        season_series=season_series,
        league_series=league_series,
        selection_method=selection_method,
        search_strategy=search_strategy,
    )
    calibration = CalibrationService.calibrate_estimator(
        estimator=search_result.champion_estimator, X=X, y=y, cv_splits=cv_splits
    )

    return CornersLineTrainResult(
        line=float(line),
        sample_size=int(len(y)),
        feature_names=feature_columns,
        calibration=calibration,
        champion_name=search_result.champion_name,
        model_results=search_result.model_results,
    )


@dataclass
class CornersBenchmarkReport:
    """Report multi-linea (acceptance criteria 'Metriche per linea')."""

    lines: tuple[float, ...]
    results: dict[str, CornersLineTrainResult]
    monotonicity: dict[str, Any] = field(default_factory=dict)

    def metrics_summary(self) -> dict[str, dict[str, Any]]:
        """Report COMPLETO per linea (2026-09-12, "tutte le metriche
        possibili"): oltre a `pre_metrics`/`post_metrics` (log loss/Brier/
        ECE/AUC/reliability, gia' esistenti), aggiunge un report di
        classificazione completo (accuracy/confusion matrix/precision/
        recall/F1/ROC/PR/soglia ottimale, via `classification_report.py`)
        calcolato sugli STESSI array OOF gia' prodotti dalla calibrazione -
        nessun nuovo giro di walk-forward."""
        summary: dict[str, dict[str, Any]] = {}
        for label, result in self.results.items():
            calibration = result.calibration
            classification_pre = compute_full_classification_report(
                y_true=calibration.pre_y_true, probabilities=calibration.pre_probabilities
            )
            classification_post = compute_full_classification_report(
                y_true=calibration.post_y_true, probabilities=calibration.post_probabilities
            )
            summary[label] = {
                "line": result.line,
                "sample_size": result.sample_size,
                "pre_metrics": calibration.pre_metrics,
                "post_metrics": calibration.post_metrics,
                "calibration_method": calibration.method,
                "classification_report_pre": classification_pre,
                "classification_report_post": classification_post,
                "champion_family": result.champion_name,
                "model_search_results": result.model_results,
            }
        return summary


def train_corners_all_lines(
    frame: pd.DataFrame,
    lines: tuple[float, ...] = DEFAULT_LINES,
    cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
    selection_method: str = "kbest",
    use_line_specific_odds: bool = True,
    search_strategy: str = "random",
) -> CornersBenchmarkReport:
    """Addestra+calibra ciascuna linea sullo STESSO walk-forward. Una linea
    con classe unica (dati insufficienti) viene saltata SENZA bloccare le
    altre linee addestrabili."""
    if cv_splits is None:
        min_train = max(30, int(len(frame) * 0.45))
        min_valid = max(10, int(len(frame) * 0.1))
        cv_splits = expanding_window_splits(
            frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid
        )
    if not cv_splits:
        raise ValueError("Dataset insufficiente per validazione temporale (nessuno split valido)")

    results: dict[str, CornersLineTrainResult] = {}
    for line in lines:
        try:
            results[_line_label(line)] = train_corners_line(
                frame=frame,
                line=line,
                cv_splits=cv_splits,
                lines_in_frame=lines,
                selection_method=selection_method,
                use_line_specific_odds=use_line_specific_odds,
                search_strategy=search_strategy,
            )
        except ValueError:
            continue

    if not results:
        raise ValueError("Nessuna linea addestrabile con questo dataset (classi troppo sbilanciate per tutte le linee)")

    trained_lines = tuple(result.line for result in results.values())
    monotonicity = compute_monotonicity_report(
        frame, lines=trained_lines, cv_splits=cv_splits, use_line_specific_odds=use_line_specific_odds
    )

    return CornersBenchmarkReport(lines=lines, results=results, monotonicity=monotonicity)


@dataclass
class CornersExpert:
    """Interfaccia comune (`predict_proba`) parametrica sulla linea: OGNI
    linea e' un mercato/modello indipendente nel registry (es.
    'corners_line_9_5'), analogamente a come 'under_over_1_5/2_5/...' sono
    gia' mercati separati (EXP-05)."""

    line: float
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("L'estimator caricato non espone predict_proba: interfaccia comune non rispettata")

    def predict_proba(self, X: Any) -> np.ndarray:
        ordered = X[self.feature_names] if self.feature_names else X
        raw = self.estimator.predict_proba(ordered)
        return CalibrationService._class1_probability(raw)

    def predict_proba_dict(self, X: Any) -> dict[str, np.ndarray]:
        p_over = self.predict_proba(X)
        return {"over": p_over, "under": 1.0 - p_over}

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    @staticmethod
    def _market_for_line(line: float) -> str:
        return f"{MARKET_NAME}_{_line_label(line)}"

    @classmethod
    def _from_run(cls, line: float, run: dict[str, Any]) -> "CornersExpert":
        model_path = run.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path non trovato per il run {run.get('run_id')}: {model_path}")

        estimator = joblib.load(model_path)
        return cls(
            line=float(line),
            estimator=estimator,
            feature_names=list(run.get("feature_names") or []),
            run_id=run.get("run_id"),
            stage=run.get("current_stage") or run.get("stage"),
        )

    @classmethod
    def load_production(cls, line: float, registry: Optional[Any] = None) -> "CornersExpert":
        """Carica lo stage 'production' per la linea richiesta (vincolo
        generale: latest != production)."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_production(market=market)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def load_latest(cls, line: float, registry: Optional[Any] = None) -> "CornersExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_latest(market=market)
        if run is None:
            raise LookupError(f"Nessun modello registrato per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def from_estimator(cls, line: float, estimator: Any, feature_names: Optional[list[str]] = None) -> "CornersExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(line=float(line), estimator=estimator, feature_names=list(feature_names or []))


@dataclass
class CornersBenchmarkRunResult:
    market: str
    rows: int
    status: str
    lines_trained: list[float]
    details: dict[str, Any]


def _save_line_model(result: CornersLineTrainResult) -> dict[str, Any]:
    line_label = _line_label(result.line)
    model_path = os.path.abspath(os.path.join("best_models", f"{MARKET_NAME}_{line_label}_champion.pkl"))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(result.calibration.calibrator, model_path)

    calibration = result.calibration
    classification_pre = compute_full_classification_report(
        y_true=calibration.pre_y_true, probabilities=calibration.pre_probabilities
    )
    classification_post = compute_full_classification_report(
        y_true=calibration.post_y_true, probabilities=calibration.post_probabilities
    )
    post_weighted_f1 = classification_post.get("weighted", {}).get("f1", 0.0)
    selection_score = champion_probability_score(metrics=calibration.post_metrics, f1_weighted=post_weighted_f1)

    registry = ModelRegistry()
    return registry.register(
        model_path=model_path,
        market=f"{MARKET_NAME}_{line_label}",
        model_name=f"calibrated_{result.champion_name}",
        feature_names=result.feature_names,
        # Metriche SCALARI (2026-09-12, "tutte le metriche possibili" +
        # compatibilita' col gate di promozione, OPS-02, che legge
        # esplicitamente post_log_loss/post_brier/post_ece/post_auc/
        # sample_size/selection_score - PRIMA mancavano post_ece/post_auc/
        # sample_size, quindi il gate avrebbe sempre bloccato la promozione
        # di questi mercati per "sample_size non disponibile").
        metrics={
            "sample_size": result.sample_size,
            "pre_log_loss": calibration.pre_metrics.get("log_loss"),
            "pre_brier": calibration.pre_metrics.get("brier"),
            "pre_ece": calibration.pre_metrics.get("ece"),
            "pre_auc": calibration.pre_metrics.get("auc"),
            "post_log_loss": calibration.post_metrics.get("log_loss"),
            "post_brier": calibration.post_metrics.get("brier"),
            "post_ece": calibration.post_metrics.get("ece"),
            "post_auc": calibration.post_metrics.get("auc"),
            "post_accuracy": classification_post.get("accuracy"),
            "post_f1_weighted": post_weighted_f1,
            "selection_score": selection_score,
        },
        # Report NESTED completo (confusion matrix/ROC/PR/soglia ottimale +
        # reliability): troppo voluminoso per il dict `metrics` (letto dal
        # gate di promozione e da liste/dashboard sintetiche), ma conservato
        # per intero qui per l'analisi/debug per-linea.
        extra={
            "line": result.line,
            "calibration_method": calibration.method,
            "sample_size": result.sample_size,
            "reliability_pre": calibration.pre_metrics.get("reliability"),
            "reliability_post": calibration.post_metrics.get("reliability"),
            "classification_report_pre": classification_pre,
            "classification_report_post": classification_post,
            # Trasparenza sulla model search (2026-09-12, "hai fatto grid
            # search/voting/stacking? hai provato piu' opzioni?"): TUTTI i
            # candidati confrontati (logistic/random_forest/
            # random_forest_smote/voting/stacking) con i loro selection_score,
            # non solo il vincitore - permette di rivedere il confronto senza
            # dover rilanciare la ricerca.
            "champion_family": result.champion_name,
            "model_search_results": result.model_results,
        },
        stage="candidate",
    )


def run_corners_benchmark(
    matches: Optional[list[dict[str, Any]]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
    frame: Optional[pd.DataFrame] = None,
) -> CornersBenchmarkRunResult:
    """Costruisce il dataset reale, addestra+calibra ciascuna linea e,
    se richiesto, registra ciascun modello come 'candidate' separato.

    `frame` (2026-09-12, additivo): se gia' fornito (es. caricato da un CSV
    esportato da un ambiente con accesso DB reale - vedi
    `export_corners_cards_for_cloud_training.py`/
    `train_corners_cards_from_export.py`, stesso pattern gia' in uso per
    Under/Over/`train_from_export.py`), viene usato DIRETTAMENTE e `matches`
    e' ignorato (mai ricostruito da zero) - permette di eseguire il training
    pesante in un ambiente SENZA accesso diretto al DB."""
    if frame is None:
        frame = build_corners_frame_from_records(matches or [], lines=lines, odds_market=odds_market)
    if frame.empty:
        return CornersBenchmarkRunResult(market=MARKET_NAME, rows=0, status="skipped_no_data", lines_trained=[], details={})

    try:
        report = train_corners_all_lines(frame, lines=lines)
    except ValueError as exc:
        return CornersBenchmarkRunResult(
            market=MARKET_NAME,
            rows=len(frame),
            status="skipped_insufficient_data_for_all_lines",
            lines_trained=[],
            details={"error": str(exc)},
        )

    run_metadata: dict[str, Any] = {}
    if save_model:
        for label, result in report.results.items():
            run_metadata[label] = _save_line_model(result)

    return CornersBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        lines_trained=[result.line for result in report.results.values()],
        details={"metrics_summary": report.metrics_summary(), "runs": run_metadata, "monotonicity": report.monotonicity},
    )


def run_corners_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CornersBenchmarkRunResult:
    """Variante DB reale: filtra le fixture con statistiche/mean_statistics/odds
    disponibili (stesso requisito degli altri dataset builder di mercato)."""
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

    return run_corners_benchmark(matches=matches, lines=lines, odds_market=odds_market, save_model=save_model)
