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


def _referee_prior_features(
    referee: str,
    league: Any,
    referee_states: dict[str, "_RefereeState"],
    league_states: dict[Any, "_RefereeState"],
    global_state: "_RefereeState",
    default_prior_cards: float,
    shrinkage_k: float,
) -> dict[str, Any]:
    """Le 4 feature arbitro per UNA partita, dato lo STATO CORRENTE
    (point-in-time) di arbitro/lega/globale - estratta da
    `build_referee_features_dataset` (2026-09-13, comportamento INVARIATO)
    per essere riusata anche da `current_referee_state` (serving live),
    senza duplicare la logica di shrinkage/fallback."""
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

    return {
        "referee_avg_cards_prior": float(avg_value),
        "referee_severity_index_prior": float(severity_index),
        "referee_matches_officiated_prior": int(matches_officiated),
        "referee_has_history": int(has_history),
    }


@dataclass
class _RefereeIndex:
    """Stato arbitro/lega/globale AGGIORNATO (dopo l'ultima partita conclusa
    disponibile) - risultato del replay, riusabile per PIU' lookup senza
    rifare il replay ogni volta (vedi `get_cached_referee_index` sotto)."""

    referee_states: dict[str, _RefereeState]
    league_states: dict[Any, _RefereeState]
    global_state: _RefereeState


def build_current_referee_index(matches: list[dict[str, Any]]) -> _RefereeIndex:
    """Replay dell'intero storico cronologico disponibile (stessa identica
    logica di accumulo di `build_referee_features_dataset`) per ottenere lo
    STATO FINALE di ogni arbitro/lega/globale - stesso principio di
    `TeamStrengthExpert.current_ratings` (`team_strength_expert.py`), ma
    esposto come indice riusabile (una volta calcolato, si puo' interrogare
    per QUALUNQUE arbitro/lega senza rifare il replay - vedi
    `referee_features_from_index`)."""
    referee_states: dict[str, _RefereeState] = defaultdict(_RefereeState)
    league_states: dict[Any, _RefereeState] = defaultdict(_RefereeState)
    global_state = _RefereeState()

    for _, match in _sorted_matches_chronologically(matches):
        match_referee = str(match.get("referee") or "").strip()
        match_league = match.get("current_league")
        stats = match.get("statistics") or []
        home_id = match.get("id_team_home")
        away_id = match.get("id_team_away")
        stat_home = next((s for s in stats if s.get("statistics_team_id") == home_id), None)
        stat_away = next((s for s in stats if s.get("statistics_team_id") == away_id), None)
        if stat_home is None or stat_away is None:
            continue

        total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
        if match_referee:
            referee_states[match_referee].update(total_cards)
        league_states[match_league].update(total_cards)
        global_state.update(total_cards)

    return _RefereeIndex(referee_states=referee_states, league_states=league_states, global_state=global_state)


def referee_features_from_index(
    index: _RefereeIndex,
    referee: str,
    league: Any,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> dict[str, Any]:
    """Lookup ECONOMICO (nessun replay) delle 4 feature arbitro per una
    fixture futura/live, dato un indice gia' calcolato da
    `build_current_referee_index`."""
    return _referee_prior_features(
        referee=str(referee or "").strip(),
        league=league,
        referee_states=index.referee_states,
        league_states=index.league_states,
        global_state=index.global_state,
        default_prior_cards=default_prior_cards,
        shrinkage_k=shrinkage_k,
    )


def current_referee_state(
    matches: list[dict[str, Any]],
    referee: str,
    league: Any,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> dict[str, Any]:
    """Stato arbitro AGGIORNATO per una fixture FUTURA/live, in UNA sola
    chiamata (replay + lookup) - comodo per un uso una tantum/nei test.
    `matches` deve includere tutte le partite concluse rilevanti (quelle
    dello stesso arbitro, della sua lega, e - per il fallback globale - un
    campione quanto piu' ampio possibile): la query e' a carico del
    chiamante, stesso principio di targeting gia' usato in
    `oracle_match_detail_service.py` (mai l'intero DB in un colpo solo).
    Per risolvere PIU' fixture nella stessa richiesta (es. Dashboard),
    preferire `get_cached_referee_index`/`referee_features_from_index`
    (un replay solo, non uno per fixture) invece di richiamare questa più
    volte."""
    index = build_current_referee_index(matches)
    return referee_features_from_index(
        index, referee=referee, league=league, default_prior_cards=default_prior_cards, shrinkage_k=shrinkage_k
    )


_REFEREE_INDEX_CACHE: dict[str, Any] = {"index": None, "computed_at": None}
_REFEREE_INDEX_CACHE_TTL_SECONDS = 900.0


def get_cached_referee_index(ttl_seconds: float = _REFEREE_INDEX_CACHE_TTL_SECONDS) -> _RefereeIndex:
    """Indice arbitro CACHED a livello di MODULO (TTL 15 minuti, stesso
    compromesso "dato quasi fresco invece di ricalcolo costante" gia'
    scelto per `ModelDiagnosticsService`/`DashboardService._api_cache") -
    evita di rifare una query + replay dell'intero storico arbitri per
    OGNI fixture richiesta in Dashboard (altrimenti lo stesso problema di
    performance N+1 gia' risolto altrove in questo progetto per
    `FilterMarketService.build_prediction_frames`). Query LEGGERA: servono
    solo referee/current_league/date_match/statistics, MAI odds/
    mean_statistics (non usati dal replay arbitro)."""
    now = datetime.now(timezone.utc)
    cached_index = _REFEREE_INDEX_CACHE.get("index")
    computed_at = _REFEREE_INDEX_CACHE.get("computed_at")
    if cached_index is not None and computed_at is not None and (now - computed_at).total_seconds() < ttl_seconds:
        return cached_index

    match_repo = MatchRepository()
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters={"statistics": "not None", "status": ["FT"]}))
    index = build_current_referee_index(matches)
    _REFEREE_INDEX_CACHE["index"] = index
    _REFEREE_INDEX_CACHE["computed_at"] = now
    return index


def current_referee_features_cached(
    referee: str,
    league: Any,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> dict[str, Any]:
    """Feature arbitro per una fixture futura/live usando l'indice CACHED
    (vedi `get_cached_referee_index`) - il percorso da usare in serving."""
    index = get_cached_referee_index()
    return referee_features_from_index(
        index, referee=referee, league=league, default_prior_cards=default_prior_cards, shrinkage_k=shrinkage_k
    )


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

        row_values = _referee_prior_features(
            referee=referee,
            league=league,
            referee_states=referee_states,
            league_states=league_states,
            global_state=global_state,
            default_prior_cards=default_prior_cards,
            shrinkage_k=shrinkage_k,
        )
        rows.append(
            {
                "id_fixture": match.get("id_fixture"),
                "prediction_at": prediction_at.isoformat(),
                **row_values,
            }
        )

        total_cards = _team_total_cards(stat_home) + _team_total_cards(stat_away)
        if referee:
            referee_states[referee].update(total_cards)
        league_states[league].update(total_cards)
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


def _line_specific_odds_features(match: dict[str, Any], odds_market: str, lines: tuple[float, ...]) -> dict[str, Any]:
    """Feature quote SEPARATE per linea (2026-09-12 - vedi
    `FilterMarketService._extract_line_specific_odds_features`): a
    differenza degli Under/Over gol (dove l'ingestion separa gia' le quote
    per soglia), 'Cards Over/Under' mette TUTTE le linee in un unico bucket
    piatto - qui si estrae, PER OGNI linea configurata, solo le quote di
    quella linea (nome colonna con suffisso `_{line_label}`, es.
    `odds_mean_line_3_5`), cosi' il modello di ciascuna linea vede SOLO le
    quote di mercato pertinenti a se stesso."""
    odds_list = match.get("odds") or []
    if not odds_list:
        return {}
    market_odds = (odds_list[0] or {}).get(odds_market)
    if not isinstance(market_odds, dict) or not market_odds:
        return {}

    features: dict[str, Any] = {}
    for line in lines:
        line_features = FilterMarketService._extract_line_specific_odds_features(market_odds, line)
        for key, value in line_features.items():
            features[f"{key}_{_line_label(line)}"] = value
    return features


def build_cards_prediction_row(
    match: dict[str, Any],
    referee_matches: Optional[list[dict[str, Any]]] = None,
    referee_features: Optional[dict[str, Any]] = None,
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    referee_shrinkage_k: float = REFEREE_SHRINKAGE_K,
) -> Optional[dict[str, Any]]:
    """Riga di feature per UNA fixture live/futura (mai training - nessun
    target). A differenza di Corners, l'arbitro e' STATEFUL (dipende dallo
    storico), quindi NON riusata dentro `build_cards_frame_from_records`
    (che calcola il prior arbitro in batch per efficienza, O(n) sull'intero
    dataset invece di un replay per riga).

    Due modi per fornire le feature arbitro (mutuamente esclusivi):
    - `referee_features` gia' calcolate (tipicamente da
      `current_referee_features_cached`, indice condiviso/cached - il
      percorso da preferire in serving, un replay solo per l'intera
      richiesta invece di uno per fixture);
    - `referee_matches` (storico grezzo, replay fatto qui via
      `current_referee_state` - comodo per un uso una tantum/nei test)."""
    service = FilterMarketService()
    row = service._build_row(match=match, market=odds_market, with_target=False)
    if not row:
        return None
    row.update(_line_specific_odds_features(match, odds_market, lines))
    if referee_features is None:
        referee_features = current_referee_state(
            referee_matches or [],
            referee=str(match.get("referee") or ""),
            league=match.get("current_league"),
            default_prior_cards=default_prior_cards,
            shrinkage_k=referee_shrinkage_k,
        )
    row.update(referee_features)
    return row


def build_cards_frame_from_records(
    matches: list[dict[str, Any]],
    lines: tuple[float, ...] = DEFAULT_LINES,
    odds_market: str = ODDS_MARKET,
    default_prior_cards: float = DEFAULT_REFEREE_PRIOR_CARDS,
    referee_shrinkage_k: float = REFEREE_SHRINKAGE_K,
    fill_missing: bool = True,
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
        row.update(_line_specific_odds_features(match, odds_market, lines))
        for line in lines:
            row[f"y_{_line_label(line)}"] = int(label_cards_over(total_cards, line))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if fill_missing:
        frame = frame.fillna(0)
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


_ODDS_METRIC_KEYS = ["odds_count", "odds_mean", "odds_std", "odds_min", "odds_max"] + [f"odds_slot_{i}" for i in range(1, 11)]


def _line_specific_odds_columns(line: float, columns: Optional[Iterable[str]] = None) -> set[str]:
    """Colonne quote riferite a QUESTA linea.

    Con `columns` (le colonne reali del frame) riconosce anche le feature
    per esito introdotte il 2026-09-13 (`odds_mean_over_3_5`,
    `prob_norm_under_3_5`, `overround_line_3_5`, ...), che non derivano da
    `_ODDS_METRIC_KEYS` e quindi non sarebbero enumerabili a priori. Senza
    `columns` resta il comportamento originario (solo le legacy).

    Senza questo riconoscimento le nuove colonne sfuggirebbero
    all'esclusione in `_feature_columns_for` e il modello di una linea
    vedrebbe le quote delle ALTRE linee - esattamente il leak che le
    quote per-linea erano nate per eliminare.
    """
    label = _line_label(line)
    legacy = {f"{key}_{label}" for key in _ODDS_METRIC_KEYS}
    if columns is None:
        return legacy
    # Due suffissi: `_line_specific_odds_features` produce
    # '..._over_3_5_line_3_5', mentre `_build_row` sul bucket intero produce
    # '..._over_3_5' (linea gia' dentro il nome dell'esito). Senza il secondo
    # le colonne delle ALTRE linee sfuggirebbero all'esclusione.
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
    quote pooled originarie. Le feature arbitro (`referee_*`) non sono mai
    toccate: sono gia' indipendenti dalla linea."""
    y_columns = {f"y_{_line_label(line)}" for line in lines}
    excluded = set(_META_COLUMNS) | {"total_cards"} | y_columns

    all_line_odds_columns: set[str] = set()
    for line in lines:
        all_line_odds_columns |= _line_specific_odds_columns(line, frame.columns)

    if use_line_specific_odds and active_line is not None:
        active_columns = _line_specific_odds_columns(active_line, frame.columns)
        excluded |= (all_line_odds_columns - active_columns)
        # Quote pooled legacy + `overround` sull'intero bucket (somma le
        # probabilita' implicite di TUTTE le linee: privo di senso per una
        # linea singola, a differenza di `overround_line_X_Y`).
        excluded |= set(_ODDS_METRIC_KEYS) | {"overround"}
    else:
        excluded |= all_line_odds_columns

    return [col for col in frame.columns if col not in excluded]


# ---------------------------------------------------------------------------
# Monotonicita' tra linee (2026-09-12, "le linee seguono lo stesso principio
# di Under/Over gol? Over 2.5 e' sicuramente anche Over 1.5" - stesso
# principio gia' in produzione per i gol, MARKET-04, `totals_market.py`):
# P(Over 3.5) >= P(Over 4.5) >= P(Over 5.5) >= P(Over 6.5) riga per riga sono
# eventi ANNIDATI sullo stesso conteggio totale, non variabili indipendenti -
# modelli indipendenti per linea NON garantiscono questo ordine da soli.
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
    logico Over 3.5 >= Over 4.5 >= Over 5.5 >= Over 6.5, e proiezione che lo
    impone SEMPRE (`enforce_monotonic_over_probabilities`, riusata identica
    da `totals_market.py` - stesso principio, non duplicata). `metrics_by_line`
    (2026-09-12, "quindi le metriche dopo aver applicato questo fix quali
    sono?"): confronto PRIMA/DOPO la proiezione per ogni linea - la linea
    piu' bassa non cambia mai (e' l'ancora della proiezione cumulativa),
    le altre di solito MIGLIORANO (mai peggiorano di molto): la proiezione
    corregge violazioni logicamente impossibili, non aggiunge rumore."""
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
    use_line_specific_odds: bool = True,
    search_strategy: str = "random",
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
    `CalibrationService.calibrate_estimator`, come prima.

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


def train_cards_all_lines(
    frame: pd.DataFrame,
    lines: tuple[float, ...] = DEFAULT_LINES,
    cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
    selection_method: str = "kbest",
    use_line_specific_odds: bool = True,
    search_strategy: str = "random",
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

    return CardsBenchmarkReport(lines=lines, results=results, monotonicity=monotonicity)


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
        details={"metrics_summary": report.metrics_summary(), "runs": run_metadata, "monotonicity": report.monotonicity},
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
