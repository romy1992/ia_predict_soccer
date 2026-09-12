"""Cards O/U specializzato con linea configurabile (MARKET-06).

Analogo a Corners O/U (MARKET-05): generalizza il mercato 'cards' esistente
(soglia FISSA a 4.5, cioe' `total_cards >= 5`, in
`FilterMarketService._label_by_market` — non modificata qui per preservare
compatibilita' con quel mercato legacy) introducendo la linea come PARAMETRO
esplicito (es. 3.5, 4.5, 5.5, 6.5).

Aggiunge, oltre alle feature generiche gia' estratte da `FilterMarketService`
(che includono gia' 'Yellow Cards'/'Red Cards'/'Fouls' medi per squadra —
le feature "team/style" richieste dal task, gia' disponibili in
`mean_statistics` senza bisogno di nuova logica):
- feature ARBITRO (NUOVE, calcolate qui): media storica cartellini totali
  (gialli+rossi, entrambe le squadre) elargiti nelle partite PRECEDENTI
  arbitrate dallo stesso arbitro, point-in-time (stesso principio anti-leakage
  di `TeamStrengthExpert`, EXP-01, ma raggruppato per `Match.referee` invece
  che per squadra), con fallback a un prior neutro quando l'arbitro non ha
  ancora storico sufficiente nel dataset;
- **raffinamento 2026-09-12** (discusso e confermato con l'operatore): il
  prior arbitro grezzo (media semplice) e' rumoroso su pochi precedenti (un
  arbitro con 1-2 partite osservate puo' avere una media estrema per puro
  caso) - `referee_avg_cards_prior` applica ora uno SHRINKAGE bayesiano
  verso un baseline (media di LEGA point-in-time se disponibile, altrimenti
  media globale, altrimenti `DEFAULT_REFEREE_PRIOR_CARDS`), con peso
  crescente sulla media grezza man mano che l'arbitro accumula partite
  (vedi `_shrink_toward_baseline`). Nuova feature aggiuntiva
  `referee_severity_index_prior` = prior (gia' shrunk) diviso per lo stesso
  baseline: un indice NORMALIZZATO PER LEGA (1.0 = nella media della sua
  lega, >1 = piu' severo, <1 = piu' permissivo) - un arbitro severo in una
  lega "dura" e uno altrettanto severo in una lega "morbida" non sono
  altrimenti comparabili sulla scala assoluta dei cartellini. Un arbitro
  MANCANTE/sconosciuto continua a ricevere sempre e solo il prior neutro
  fisso (mai un baseline di lega per un'identita' che non conosciamo);
- calibrazione (Platt/isotonic) via `CalibrationService` (ML-06, riusata,
  non duplicata) per il modello di ciascuna linea;
- report con metriche PER LINEA, ora COMPLETO (acceptance criteria +
  richiesta esplicita "tutte le metriche possibili", 2026-09-12): oltre a
  log loss/Brier/ECE/AUC pre/post calibrazione (gia' esistenti), ogni linea
  espone anche un report di classificazione completo (accuracy, confusion
  matrix, precision/recall/F1 per classe e pesati, curva ROC+PR, soglia
  ottimale di Youden) via `classification_report.py` (NUOVO, non duplicato
  qui ne' in corners_market.py, che lo usa identico).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.ml.calibration.calibration_service import CalibrationResult, CalibrationService
from src.ml.evaluation.classification_report import compute_full_classification_report
from src.ml.evaluation.probability_metrics import champion_probability_score
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import _select_champion_via_model_search
from src.service_ia.utility.utils import convert_orm_match_to_dict

MARKET_NAME = "cards"
ODDS_MARKET = "cards"  # colonna Odds.cards esistente, riusata come fonte feature quote
DEFAULT_LINES: tuple[float, ...] = (3.5, 4.5, 5.5, 6.5)
DEFAULT_REFEREE_PRIOR_CARDS = 4.0  # cartellini totali medi neutri quando ne' l'arbitro ne' la sua lega hanno storico
REFEREE_SHRINKAGE_K = 10.0  # "peso" in pseudo-partite del baseline nello shrinkage bayesiano del prior arbitro

_META_COLUMNS = ["id_fixture", "season", "league", "market", "prediction_at"]


def _line_label(line: float) -> str:
    return f"line_{str(float(line)).replace('.', '_')}"


def label_cards_over(total_cards: Any, line: float) -> Any:
    """Target reale (dal risultato, MAI dalle quote): 1 se cartellini totali > line.

    Accetta sia uno scalare sia un array; la linea e' un PARAMETRO libero,
    non una soglia hardcoded (acceptance criteria 'Linee configurabili')."""
    result = (np.asarray(total_cards, dtype=float) > float(line)).astype(int)
    return result if result.shape else int(result)


def _team_total_cards(stat: dict[str, Any]) -> int:
    yellow = int(stat.get("yellow_cards") or 0)
    red = int(stat.get("red_cards") or 0)
    return yellow + red


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sorted_matches_chronologically(matches: list[dict[str, Any]]) -> list[tuple[datetime, dict[str, Any]]]:
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for match in matches:
        prediction_at = _parse_datetime(match.get("date_match"))
        if prediction_at is None:
            continue
        parsed.append((prediction_at, match))
    parsed.sort(key=lambda item: (item[0], item[1].get("id_fixture") or 0))
    return parsed


@dataclass
class _RefereeState:
    matches_officiated: int = 0
    total_cards_sum: float = 0.0

    @property
    def average_cards(self) -> Optional[float]:
        if self.matches_officiated == 0:
            return None
        return self.total_cards_sum / self.matches_officiated

    def update(self, total_cards: int) -> None:
        self.total_cards_sum += float(total_cards)
        self.matches_officiated += 1


def _shrink_toward_baseline(raw_average: float, matches_officiated: int, baseline: float, k: float) -> float:
    """Shrinkage bayesiano (media pesata "conteggio vs pseudo-conteggio"):
    con pochi precedenti (`matches_officiated` basso) il risultato resta
    vicino al `baseline` (lega/globale/costante); con molti precedenti tende
    alla media grezza dell'arbitro. Formula standard "credibility weighting"
    (equivalente a un prior Beta/Normale coniugato con `k` osservazioni
    virtuali pari al baseline)."""
    n = float(matches_officiated)
    return ((n * raw_average) + (k * baseline)) / (n + k)


def build_referee_features_dataset(
    matches: list[dict[str, Any]],
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> pd.DataFrame:
    """1 riga per fixture: statistiche ARBITRO point-in-time (PRE-match).

    Nessun leakage: per la riga N si usano SOLO le partite (dello stesso
    arbitro, della sua lega, o dell'intero dataset) con indice cronologico
    < N.

    `referee_avg_cards_prior` e' la media storica cartellini dell'arbitro
    con SHRINKAGE bayesiano verso un baseline via via piu' generico quando
    manca informazione piu' specifica: media di LEGA point-in-time (se la
    lega ha gia' storico) -> media GLOBALE point-in-time (se nessuna partita
    di quella lega e' ancora stata vista) -> `default_prior_cards` (nessuno
    storico affatto, es. primissima partita del dataset). Un arbitro
    assente/sconosciuto (stringa vuota) e' un caso a parte, deliberatamente
    SEMPRE `default_prior_cards`: non abbiamo nessuna identita' su cui
    ragionare, quindi niente stima "furba" basata sulla lega.

    `referee_severity_index_prior` (NUOVO) e' il prior (gia' shrunk) diviso
    per lo stesso baseline usato per lo shrinkage: un indice di severita'
    NORMALIZZATO PER LEGA (1.0 = nella media, >1 = piu' severo). Per un
    arbitro assente resta fisso a 1.0 (nessuna informazione = nessuna
    deviazione dalla norma dichiarabile).
    """
    referee_states: dict[str, _RefereeState] = defaultdict(_RefereeState)
    league_states: dict[Any, _RefereeState] = defaultdict(_RefereeState)
    global_state = _RefereeState()
    rows: list[dict[str, Any]] = []

    for prediction_at, match in _sorted_matches_chronologically(matches):
        referee = str(match.get("referee") or "").strip()
        league = match.get("current_league")
        stats = match.get("statistics") or []
        home_id = match.get("id_team_home")
        away_id = match.get("id_team_away")
        stat_home = next((s for s in stats if s.get("statistics_team_id") == home_id), None)
        stat_away = next((s for s in stats if s.get("statistics_team_id") == away_id), None)
        if stat_home is None or stat_away is None:
            continue

        league_state = league_states[league]
        baseline = league_state.average_cards
        if baseline is None:
            baseline = global_state.average_cards
        if baseline is None:
            baseline = float(default_prior_cards)

        if not referee:
            avg_value = float(default_prior_cards)
            matches_officiated = 0
            has_history = 0
            severity_index = 1.0
        else:
            state = referee_states[referee]
            raw_average = state.average_cards
            matches_officiated = state.matches_officiated
            has_history = int(raw_average is not None)
            avg_value = baseline if raw_average is None else _shrink_toward_baseline(
                raw_average=raw_average, matches_officiated=matches_officiated, baseline=baseline, k=shrinkage_k
            )
            severity_index = (avg_value / baseline) if baseline > 0 else 1.0

        rows.append(
            {
                "id_fixture": match.get("id_fixture"),
                "prediction_at": prediction_at.isoformat(),
                "referee_avg_cards_prior": float(avg_value),
                "referee_severity_index_prior": float(severity_index),
                "referee_matches_officiated_prior": int(matches_officiated),
                "referee_has_history": int(has_history),
            }
        )

        total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
        if referee:
            referee_states[referee].update(total_cards)
        league_state.update(total_cards)
        global_state.update(total_cards)

    if not rows:
        return pd.DataFrame(
            columns=[
                "id_fixture",
                "referee_avg_cards_prior",
                "referee_severity_index_prior",
                "referee_matches_officiated_prior",
                "referee_has_history",
            ]
        )

    return pd.DataFrame(rows)


def build_cards_frame_from_records(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    referee_shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> pd.DataFrame:
    """Dataset (feature odds/mean_stats generiche "team/style" + feature
    arbitro dedicate + target reale PER OGNI linea) sulle fixture con quote
    'cards' disponibili."""
    service = FilterMarketService()
    rows: list[dict[str, Any]] = []
    for match in matches:
        row = service._build_row(match=match, market=odds_market, with_target=False)
        if not row:
            continue

        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            continue

        total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
        row["total_cards"] = total_cards
        for line in lines:
            row[f"y_{_line_label(line)}"] = int(label_cards_over(total_cards, line))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["id_fixture"] = frame["id_fixture"].astype(int)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)

    referee_frame = build_referee_features_dataset(
        matches, default_prior_cards=default_prior_cards, shrinkage_k=referee_shrinkage_k
    )
    referee_columns = [
        "referee_avg_cards_prior",
        "referee_severity_index_prior",
        "referee_matches_officiated_prior",
        "referee_has_history",
    ]
    referee_defaults = {
        "referee_avg_cards_prior": default_prior_cards,
        "referee_severity_index_prior": 1.0,
        "referee_matches_officiated_prior": 0,
        "referee_has_history": 0,
    }
    if referee_frame.empty:
        for col in referee_columns:
            frame[col] = referee_defaults[col]
        return frame

    referee_frame = referee_frame[["id_fixture", *referee_columns]].copy()
    referee_frame["id_fixture"] = referee_frame["id_fixture"].astype(int)
    frame = frame.merge(referee_frame, on="id_fixture", how="left")
    for col in referee_columns:
        frame[col] = frame[col].fillna(referee_defaults[col])
    return frame


def _feature_columns_for(frame: pd.DataFrame, lines: tuple[float, ...]) -> list[str]:
    y_columns = {f"y_{_line_label(line)}" for line in lines}
    excluded = set(_META_COLUMNS) | {"total_cards"} | y_columns
    return [col for col in frame.columns if col not in excluded]


@dataclass
class CardsLineTrainResult:
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


def train_cards_line(
    frame: pd.DataFrame,
    line: float,
    cv_splits: list[tuple[list[int], list[int]]],
    lines_in_frame: tuple[float, ...] = DEFAULT_LINES,
    selection_method: str = "kbest",
) -> CardsLineTrainResult:
    """Addestra + calibra il modello per una singola linea, con validazione
    temporale OOF (mai split random).

    **Model search completo (2026-09-12)**: non piu' un singolo
    `RandomForestClassifier` a iperparametri fissi - riusa
    `_select_champion_via_model_search` (la STESSA ricerca gia' usata da
    `train_market()` per h2h/goal_no_goal/under_over_*, mai duplicata):
    grid search su logistic/random_forest/random_forest_smote, ensemble
    voting+stacking sui 2 migliori, selezione del champion per
    `selection_score`. Il champion (calibrato) e' poi passato a
    `CalibrationService.calibrate_estimator`, come prima."""
    label = f"y_{_line_label(line)}"
    if label not in frame.columns:
        raise ValueError(f"Linea non presente nel dataset: {line}")

    feature_columns = _feature_columns_for(frame, lines_in_frame)
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
    )
    calibration = CalibrationService.calibrate_estimator(
        estimator=search_result.champion_estimator, X=X, y=y, cv_splits=cv_splits
    )

    return CardsLineTrainResult(
        line=float(line),
        sample_size=int(len(y)),
        feature_names=feature_columns,
        calibration=calibration,
        champion_name=search_result.champion_name,
        model_results=search_result.model_results,
    )


@dataclass
class CardsBenchmarkReport:
    """Report multi-linea (acceptance criteria 'Metriche per linea')."""

    lines: tuple[float, ...]
    results: dict[str, CardsLineTrainResult]

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


def train_cards_all_lines(
    frame: pd.DataFrame,
    lines: tuple[float, ...] = DEFAULT_LINES,
    cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
    selection_method: str = "kbest",
) -> CardsBenchmarkReport:
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

    results: dict[str, CardsLineTrainResult] = {}
    for line in lines:
        try:
            results[_line_label(line)] = train_cards_line(
                frame=frame, line=line, cv_splits=cv_splits, lines_in_frame=lines, selection_method=selection_method
            )
        except ValueError:
            continue

    if not results:
        raise ValueError("Nessuna linea addestrabile con questo dataset (classi troppo sbilanciate per tutte le linee)")

    return CardsBenchmarkReport(lines=lines, results=results)


@dataclass
class CardsExpert:
    """Interfaccia comune (`predict_proba`) parametrica sulla linea: OGNI
    linea e' un mercato/modello indipendente nel registry (es.
    'cards_line_4_5'), analogamente a `CornersExpert` (MARKET-05)."""

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
    def _from_run(cls, line: float, run: dict[str, Any]) -> "CardsExpert":
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
    def load_production(cls, line: float, registry: Optional[Any] = None) -> "CardsExpert":
        """Carica lo stage 'production' per la linea richiesta (vincolo
        generale: latest != production)."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_production(market=market)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def load_latest(cls, line: float, registry: Optional[Any] = None) -> "CardsExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        market = cls._market_for_line(line)
        run = registry.get_latest(market=market)
        if run is None:
            raise LookupError(f"Nessun modello registrato per '{market}'")
        return cls._from_run(line=line, run=run)

    @classmethod
    def from_estimator(cls, line: float, estimator: Any, feature_names: Optional[list[str]] = None) -> "CardsExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(line=float(line), estimator=estimator, feature_names=list(feature_names or []))


@dataclass
class CardsBenchmarkRunResult:
    market: str
    rows: int
    status: str
    lines_trained: list[float]
    details: dict[str, Any]


def _save_line_model(result: CardsLineTrainResult) -> dict[str, Any]:
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


def run_cards_benchmark(
    matches: Optional[list[dict[str, Any]]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
    frame: Optional[pd.DataFrame] = None,
) -> CardsBenchmarkRunResult:
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
        frame = build_cards_frame_from_records(matches or [], lines=lines, odds_market=odds_market)
    if frame.empty:
        return CardsBenchmarkRunResult(market=MARKET_NAME, rows=0, status="skipped_no_data", lines_trained=[], details={})

    try:
        report = train_cards_all_lines(frame, lines=lines)
    except ValueError as exc:
        return CardsBenchmarkRunResult(
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

    return CardsBenchmarkRunResult(
        market=MARKET_NAME,
        rows=len(frame),
        status="benchmarked",
        lines_trained=[result.line for result in report.results.values()],
        details={"metrics_summary": report.metrics_summary(), "runs": run_metadata},
    )


def run_cards_benchmark_from_db(
    seasons: Optional[list[int]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    save_model: bool = True,
) -> CardsBenchmarkRunResult:
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

    return run_cards_benchmark(matches=matches, lines=lines, odds_market=odds_market, save_model=save_model)
