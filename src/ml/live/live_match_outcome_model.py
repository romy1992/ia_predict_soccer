"""Addestramento/validazione modelli LIVE separati dal pre-match (LIVE-03).

Riusa i mattoni multiclasse gia' validati per il mercato 1X2 pre-match
(MARKET-01/ORACLE-*): stesse metriche (`multiclass_probability_metrics`),
stessa validazione temporale di base (`expanding_window_splits`, ML-02) - MA
con uno split ADATTATO al dataset LIVE, dove esistono PIU' righe per la
stessa fixture (una per ogni snapshot/minuto disponibile, LIVE-02): uno
split "riga per riga" mescolerebbe minuti diversi della STESSA partita tra
train e validation, un leakage evidente (il modello "vedrebbe" gia' nel
training set informazioni fortemente correlate a una partita che dovrebbe
poi validare). `match_level_temporal_splits` risolve raggruppando SEMPRE per
`fixture_id` PRIMA di applicare lo split temporale (acceptance criteria
"Split per match/time" + "No leakage eventi futuri").

Registrato nel `ModelRegistry` con `market='1x2_live'` (mai lo stesso
`market` key del modello pre-match '1x2' di `market_1x2.py`): questo rende
la separazione delle due pipeline VERIFICABILE (stage/lifecycle/promotion
completamente indipendenti), non solo "il codice sta in un modulo diverso"
(acceptance criteria "Pipeline separata da pre-match"). Stesso vincolo
generale del progetto: l'ultimo training registra sempre `stage='candidate'`,
MAI automaticamente 'production' (la promozione resta un'azione esplicita
di `ModelRegistry.promote_with_policy`, OPS-02).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation.multiclass_probability_metrics import (
    compute_multiclass_probability_metrics,
    multiclass_champion_score,
    reorder_probabilities_to_labels,
    temporal_oof_multiclass_probabilities,
)
from src.ml.live.live_feature_store import LiveFeatureStore
from src.ml.live.live_match_outcome_dataset import (
    FINAL_STATUSES,
    META_COLUMNS,
    build_match_outcome_dataset,
)
from src.ml.markets.market_1x2 import OUTCOME_LABELS
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.live_data_repository import LiveDataRepository
from src.service_ia.training.model_registry import ModelRegistry

MARKET_NAME = "1x2_live"

# Finestre temporali per il report "metriche per minuto" (acceptance
# criteria "Metriche live per minuto/finestra"): bordo destro ESCLUSO,
# ultima finestra aperta (90+, tempo di recupero incluso).
_MINUTE_WINDOW_EDGES: tuple[tuple[float, str], ...] = (
    (15.0, "00_15"),
    (30.0, "15_30"),
    (45.0, "30_45"),
    (60.0, "45_60"),
    (75.0, "60_75"),
    (90.0, "75_90"),
)
_WINDOW_ORDER = ["00_15", "15_30", "30_45", "45_60", "60_75", "75_90", "90_plus", "FT", "unknown"]


def minute_window_label(status: Any, minute_total: Any) -> str:
    """Etichetta finestra temporale per il report per-minuto. Uno stato
    FINALE vale sempre 'FT' indipendentemente da `minute_total` (piu'
    robusto di un confronto puramente numerico, per i casi in cui il
    provider non valorizza l'elapsed sull'ultimo poll di una partita
    conclusa)."""
    if str(status or "").upper() in FINAL_STATUSES:
        return "FT"
    if minute_total is None or (isinstance(minute_total, float) and np.isnan(minute_total)):
        return "unknown"
    value = float(minute_total)
    for edge, label in _MINUTE_WINDOW_EDGES:
        if value < edge:
            return label
    return "90_plus"


def match_level_temporal_splits(
    frame: pd.DataFrame,
    fixture_col: str = "fixture_id",
    time_col: str = "match_time",
    n_splits: int = 5,
    min_train_matches: int = 8,
    min_valid_matches: int = 2,
) -> list[tuple[list[int], list[int]]]:
    """Split expanding-window sui MATCH (mai sulle singole righe): tutte le
    righe della stessa fixture finiscono SEMPRE nello stesso lato
    (train/valid) di ogni fold - acceptance criteria "Split per match/time"
    + "No leakage eventi futuri" (una riga di validation non puo' mai
    condividere la fixture con una riga di training, che includerebbe stato
    - passato o futuro rispetto ad essa - della STESSA partita).

    Implementazione: riduce il frame a 1 riga per fixture (il `match_time`
    minimo, proxy del kickoff/primo-snapshot-noto), applica
    `expanding_window_splits` (ML-02, stesso algoritmo gia' validato per il
    pre-match) su quel frame ridotto per ottenere split a livello di MATCH,
    poi espande ogni indice di match a TUTTE le righe (posizioni) del frame
    originale che appartengono a quella fixture.
    """
    if frame.empty or fixture_col not in frame.columns or time_col not in frame.columns:
        return []

    match_level = (
        frame.groupby(fixture_col, as_index=False)[time_col]
        .min()
        .sort_values(by=[time_col, fixture_col])
        .reset_index(drop=True)
    )

    match_splits = expanding_window_splits(
        frame=match_level,
        time_col=time_col,
        n_splits=n_splits,
        min_train_size=min_train_matches,
        min_valid_size=min_valid_matches,
    )
    if not match_splits:
        return []

    positions_by_fixture: dict[Any, list[int]] = {}
    for position, fixture_id in enumerate(frame[fixture_col].tolist()):
        positions_by_fixture.setdefault(fixture_id, []).append(position)

    row_splits: list[tuple[list[int], list[int]]] = []
    for train_match_idx, valid_match_idx in match_splits:
        train_fixtures = match_level.iloc[train_match_idx][fixture_col].tolist()
        valid_fixtures = match_level.iloc[valid_match_idx][fixture_col].tolist()

        train_rows = [pos for fid in train_fixtures for pos in positions_by_fixture.get(fid, [])]
        valid_rows = [pos for fid in valid_fixtures for pos in positions_by_fixture.get(fid, [])]
        if not train_rows or not valid_rows:
            continue
        row_splits.append((sorted(train_rows), sorted(valid_rows)))

    return row_splits


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in META_COLUMNS]


def _model_space() -> dict[str, Pipeline]:
    """Confronto tra due famiglie semplici (nessun `GridSearchCV`: il
    dataset LIVE ha gia' molte righe per fixture - un grid esteso
    moltiplicherebbe i tempi senza un beneficio proporzionato per questo
    task - ma restano comunque PIU' modelli confrontati sullo stesso OOF,
    mai un singolo modello assunto a priori)."""
    return {
        "logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"
                    ),
                ),
            ]
        ),
    }


def minute_window_report(frame: pd.DataFrame, oof: pd.DataFrame) -> list[dict[str, Any]]:
    """Metriche multiclasse raggruppate per finestra temporale (acceptance
    criteria "Metriche live per minuto/finestra") - equivalente multiclasse
    di `grouped_probability_report` (evaluation binario ML), qui perche' non
    esiste ancora una variante "grouped" nativa in
    `multiclass_probability_metrics.py`. `oof["index"]` sono posizioni
    (0-based) nel `frame` originale (garantito da
    `temporal_oof_multiclass_probabilities`/`match_level_temporal_splits`),
    quindi si puo' risalire a `status`/`minute_total` via `frame.iloc`."""
    if oof.empty:
        return []

    prob_cols = [f"prob_{label}" for label in OUTCOME_LABELS]
    windows = [
        minute_window_label(frame.iloc[int(idx)].get("status"), frame.iloc[int(idx)].get("minute_total"))
        for idx in oof["index"].tolist()
    ]
    working = oof.copy()
    working["window"] = windows

    report: list[dict[str, Any]] = []
    for window, chunk in working.groupby("window"):
        metrics = compute_multiclass_probability_metrics(
            y_true=chunk["y_true"].tolist(),
            probabilities=chunk[prob_cols].to_numpy(),
            class_labels=OUTCOME_LABELS,
        )
        report.append({"window": window, **metrics})

    report.sort(key=lambda item: _WINDOW_ORDER.index(item["window"]) if item["window"] in _WINDOW_ORDER else len(_WINDOW_ORDER))
    return report


@dataclass
class LiveMatchOutcomeTrainResult:
    market: str
    rows: int
    matches: int
    status: str
    champion: Optional[str] = None
    best_score: Optional[float] = None
    minute_window_metrics: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def train_live_match_outcome_model(
    frame: Optional[pd.DataFrame] = None,
    feature_store: Optional[LiveFeatureStore] = None,
    live_repository: Optional[LiveDataRepository] = None,
    save_model: bool = True,
    n_splits: int = 5,
    min_train_matches: int = 8,
    min_valid_matches: int = 2,
) -> LiveMatchOutcomeTrainResult:
    """Addestra e valida modelli LIVE (probabilita' 1X2 aggiornate durante
    il match, non solo pre-kickoff). `frame` permette di iniettare un
    dataset gia' costruito (usato nei test); se assente viene costruito da
    `build_match_outcome_dataset` (dataset LIVE reale, mai il dataset
    pre-match di `market_1x2.py`)."""
    df = frame if frame is not None else build_match_outcome_dataset(
        feature_store=feature_store, live_repository=live_repository
    )
    if df.empty:
        return LiveMatchOutcomeTrainResult(market=MARKET_NAME, rows=0, matches=0, status="skipped_no_data")
    df = df.reset_index(drop=True)

    matches_count = int(df["fixture_id"].nunique())
    y = df["y"].astype(str)
    unexpected = set(y.unique()) - set(OUTCOME_LABELS)
    if unexpected:
        # Acceptance criteria (ereditato da MARKET-01): nessun mapping ammesso
        # al di fuori di HOME/DRAW/AWAY, nemmeno per il modello live.
        raise ValueError(f"Etichette 1X2 inattese nel dataset live: {unexpected}")

    feature_names = _feature_columns(df)
    X = df[feature_names]
    if X.empty:
        return LiveMatchOutcomeTrainResult(
            market=MARKET_NAME, rows=len(df), matches=matches_count, status="skipped_no_feature_columns"
        )

    cv_splits = match_level_temporal_splits(
        df, n_splits=n_splits, min_train_matches=min_train_matches, min_valid_matches=min_valid_matches
    )
    cv_splits = [
        (train_idx, valid_idx) for train_idx, valid_idx in cv_splits if y.iloc[train_idx].nunique() >= 2
    ]
    if len(cv_splits) < 2:
        return LiveMatchOutcomeTrainResult(
            market=MARKET_NAME,
            rows=len(df),
            matches=matches_count,
            status="skipped_insufficient_matches_for_temporal_cv",
            details={"classes": y.value_counts().to_dict()},
        )

    model_results: dict[str, dict[str, Any]] = {}
    fitted_estimators: dict[str, Any] = {}
    oof_by_model: dict[str, pd.DataFrame] = {}

    for model_name, pipeline in _model_space().items():
        oof = temporal_oof_multiclass_probabilities(
            estimator=pipeline, X=X, y=y, cv_splits=cv_splits, class_labels=OUTCOME_LABELS
        )
        if oof.empty:
            continue

        prob_cols = [f"prob_{label}" for label in OUTCOME_LABELS]
        prob_matrix = oof[prob_cols].to_numpy()
        metrics = compute_multiclass_probability_metrics(
            y_true=oof["y_true"].tolist(), probabilities=prob_matrix, class_labels=OUTCOME_LABELS
        )
        predicted_labels = [OUTCOME_LABELS[i] for i in prob_matrix.argmax(axis=1)]
        f1_weighted = float(f1_score(oof["y_true"].tolist(), predicted_labels, average="weighted", zero_division=0))
        score = multiclass_champion_score(metrics=metrics, f1_weighted=f1_weighted)

        model_results[model_name] = {
            "probability_metrics": metrics,
            "f1_weighted": f1_weighted,
            "selection_score": score,
        }
        oof_by_model[model_name] = oof

        # Fit finale su TUTTO il dataset disponibile (stesso pattern di
        # `market_1x2.py`: l'OOF serve a scegliere/validare, il modello poi
        # servito viene rifittato sull'intero storico noto).
        pipeline.fit(X, y)
        fitted_estimators[model_name] = pipeline

    if not model_results:
        return LiveMatchOutcomeTrainResult(
            market=MARKET_NAME, rows=len(df), matches=matches_count, status="skipped_no_valid_model"
        )

    champion_name, champion_payload = max(model_results.items(), key=lambda kv: kv[1]["selection_score"])
    champion_estimator = fitted_estimators[champion_name]
    champion_oof = oof_by_model[champion_name]

    minute_report = minute_window_report(df, champion_oof)

    run_metadata = None
    if save_model:
        model_path = os.path.abspath(os.path.join("best_models", f"{MARKET_NAME}_champion.pkl"))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(champion_estimator, model_path)

        registry = ModelRegistry()
        run_metadata = registry.register(
            model_path=model_path,
            market=MARKET_NAME,
            model_name=champion_name,
            feature_names=feature_names,
            metrics={
                "selection_score": champion_payload["selection_score"],
                "f1_weighted": champion_payload["f1_weighted"],
                "log_loss": champion_payload["probability_metrics"].get("log_loss"),
                "brier": champion_payload["probability_metrics"].get("brier"),
                "sample_size": champion_payload["probability_metrics"].get("sample_size"),
            },
            extra={
                "classification_type": "multiclass",
                "classes": list(OUTCOME_LABELS),
                "pipeline": "live_separate_from_pre_match",
                "matches": matches_count,
                "minute_window_metrics": minute_report,
            },
            stage="candidate",
        )

    return LiveMatchOutcomeTrainResult(
        market=MARKET_NAME,
        rows=len(df),
        matches=matches_count,
        status="trained",
        champion=champion_name,
        best_score=champion_payload["selection_score"],
        minute_window_metrics=minute_report,
        details={
            "cv_strategy": "match_level_expanding_window",
            "cv_folds": len(cv_splits),
            "classes": list(OUTCOME_LABELS),
            "models": model_results,
            "run": run_metadata,
        },
    )


@dataclass
class LiveMatchOutcomeExpert:
    """Interfaccia comune multiclasse per il modello LIVE (stesso contratto
    di `Market1x2Expert`, ma pipeline/registry SEMPRE quella live -
    `market='1x2_live'`, mai quella pre-match)."""

    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("L'estimator caricato non espone predict_proba: interfaccia comune non rispettata")

    def predict_proba(self, X: Any) -> np.ndarray:
        ordered_X = X[self.feature_names] if self.feature_names else X
        raw = self.estimator.predict_proba(ordered_X)
        classes = getattr(self.estimator, "classes_", OUTCOME_LABELS)
        return reorder_probabilities_to_labels(classes, raw, OUTCOME_LABELS)

    def predict_proba_dict(self, X: Any) -> list[dict[str, float]]:
        """Tre probabilita' esplicite per riga: {"HOME": .., "DRAW": .., "AWAY": ..}."""
        matrix = self.predict_proba(X)
        return [
            {label: float(matrix[row_idx, col_idx]) for col_idx, label in enumerate(OUTCOME_LABELS)}
            for row_idx in range(matrix.shape[0])
        ]

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return np.asarray([OUTCOME_LABELS[i] for i in idx])

    @classmethod
    def load_production(cls, registry: Optional[Any] = None) -> "LiveMatchOutcomeExpert":
        """Carica lo stage 'production' (vincolo generale: latest != production)."""
        registry = registry or ModelRegistry()
        run = registry.get_production(market=MARKET_NAME)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per il mercato '{MARKET_NAME}'")
        return cls._from_run(run)

    @classmethod
    def load_latest(cls, registry: Optional[Any] = None) -> "LiveMatchOutcomeExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        run = registry.get_latest(market=MARKET_NAME)
        if run is None:
            raise LookupError(f"Nessun modello registrato per il mercato '{MARKET_NAME}'")
        return cls._from_run(run)

    @classmethod
    def _from_run(cls, run: dict[str, Any]) -> "LiveMatchOutcomeExpert":
        model_path = run.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path non trovato per il run {run.get('run_id')}: {model_path}")

        estimator = joblib.load(model_path)
        return cls(
            estimator=estimator,
            feature_names=list(run.get("feature_names") or []),
            run_id=run.get("run_id"),
            stage=run.get("current_stage") or run.get("stage"),
        )

    @classmethod
    def from_estimator(cls, estimator: Any, feature_names: Optional[list[str]] = None) -> "LiveMatchOutcomeExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(estimator=estimator, feature_names=list(feature_names or []))

