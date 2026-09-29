"""Feature store LIVE (LIVE-02, Fase LIVE ORACLE).

Trasforma le osservazioni GREZZE del dataset live (LIVE-01: `live_fixture_
snapshot`/`live_match_event`/`live_fixture_stat_snapshot`, tramite
`LiveDataRepository`) in feature NUMERICHE point-in-time, per un dato
`fixture_id` ad un istante `as_of` esplicito.

Nessuna scrittura, nessun training: questo modulo NON accoppia il dataset
live al dataset/pipeline pre-match (`src/ml/datasets/point_in_time_builder.py`
/ ML-01) - stesso principio di separazione gia' applicato in LIVE-01 ("Non
mischiare training pre-match e live"). L'addestramento di modelli che
CONSUMANO queste feature e' scope di LIVE-03.

Principio cardine (stesso di `PointInTimeDatasetBuilder`): OGNI feature e'
timestamped e MAI calcolata usando dati con timestamp successivo ad `as_of`
(acceptance criteria "Ogni feature live timestamped"). Per questo:
- lo snapshot fixture usato e' il piu' recente con `captured_at <= as_of`;
- eventi/statistiche usati hanno `captured_at <= as_of`;
- il prior pre-match (Prediction Ledger, BET-06) usa `created_at <= as_of`
  (sempre vero in pratica: e' salvato PRIMA del kickoff, quindi sempre prima
  di qualunque istante live, ma il filtro resta esplicito per sicurezza/
  documentazione, non per fiducia implicita).
`LiveFeatureRow.feature_available_at_max` rende questo verificabile
(`LiveFeatureStore.assert_no_leakage`, stesso pattern di
`PointInTimeDatasetBuilder.assert_no_leakage`).

## Schema (una colonna = una feature, prefissi stabili)
- Identita'/tempo: `fixture_id`, `as_of` (ISO8601, istante richiesto),
  `feature_available_at_max` (ISO8601, timestamp piu' recente EFFETTIVAMENTE
  usato tra tutte le fonti - sempre <= `as_of`).
- **minute**: `minute` (elapsed), `minute_extra` (recupero), `minute_total`
  (minute + extra), `status` (short status API-Sports: '1H'/'2H'/'FT'/...).
- **scoreline**: `home_goals`, `away_goals`, `goal_diff` (home-away),
  `total_goals`.
- **cards** (dagli EVENTI, non dalle statistiche aggregate: ogni cartellino
  ha gia' il proprio timestamp/minuto, piu' preciso di un conteggio
  aggregato): `yellow_cards_home`, `yellow_cards_away`, `yellow_cards_diff`,
  `red_cards_home` (include "second yellow"), `red_cards_away`,
  `red_cards_diff`.
- **shots/xG se disponibili** (dalle statistiche grezze `fixtures/
  statistics`, payload `[{type, value}, ...]`): per ciascun tipo NELLA
  whitelist (`_STAT_TYPE_FEATURE_MAP`) tre colonne `{nome}_home`/
  `{nome}_away`/`{nome}_diff` - presenti SOLO se il provider ha
  effettivamente valorizzato quel tipo per quella squadra (niente zero
  finto quando il piano API non fornisce `expected_goals`, spesso `null`).
  Include anche `yellow_cards_stat`/`red_cards_stat` (fonte aggregata
  alternativa alle `cards` da eventi, utile se il polling eventi perde
  un aggiornamento) e `corner_kicks`/`fouls`/`ball_possession_pct`.
- **pre-match prior**: per ciascuna coppia (market, outcome) con una
  prediction salvata pre-kickoff (BET-06), colonne
  `prior__{market}__{outcome}__p_model`,
  `prior__{market}__{outcome}__p_market_fair`,
  `prior__{market}__{outcome}__prob_edge`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.data.live.live_models import LiveFixtureSnapshot, LiveFixtureStatSnapshot, LiveMatchEvent
from src.repository.live_data_repository import LiveDataRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import PredictionLedger

_CARD_EVENT_TYPE = "card"
_YELLOW_CARD_DETAILS = {"yellow card"}
_RED_CARD_DETAILS = {"red card", "second yellow card"}

# Whitelist statistiche grezze (API-Sports `fixtures/statistics`) esposte come
# feature - "se disponibili": una entry mancante o con `value` non numerico
# (es. null, tipico per `expected_goals` sui piani base) viene SEMPLICEMENTE
# omessa, mai sostituita con uno zero (evita di introdurre falso segnale).
_STAT_TYPE_FEATURE_MAP: dict[str, str] = {
    "shots on goal": "shots_on_goal",
    "shots off goal": "shots_off_goal",
    "total shots": "total_shots",
    "blocked shots": "blocked_shots",
    "shots insidebox": "shots_insidebox",
    "shots outsidebox": "shots_outsidebox",
    "expected_goals": "expected_goals",
    "corner kicks": "corner_kicks",
    "yellow cards": "yellow_cards_stat",
    "red cards": "red_cards_stat",
    "fouls": "fouls",
    "offsides": "offsides",
    "ball possession": "ball_possession_pct",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _safe_float(value: Any) -> Optional[float]:
    """`None`/non numerico -> `None` (feature omessa dal chiamante), MAI 0.0
    finto: distingue "non disponibile" da "valore zero osservato"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite (usato nei test/dev) perde il tzinfo dei `DateTime(timezone=
    True)` alla lettura; Postgres (produzione) lo preserva. Un timestamp
    naive letto dal DB e' comunque stato scritto in UTC (default dei model
    `live_models.py`/`PredictionLedger`), quindi normalizzarlo ad UTC qui e'
    corretto e permette confronti sicuri con `as_of` (sempre timezone-aware)
    senza `TypeError: can't compare offset-naive and offset-aware
    datetimes`."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def minute_features(snapshot: Optional[LiveFixtureSnapshot]) -> dict[str, Any]:
    """'minute': minuto di gioco al momento dello snapshot selezionato."""
    if snapshot is None:
        return {"minute": None, "minute_extra": None, "minute_total": None, "status": None}
    minute = snapshot.elapsed_minute
    extra = snapshot.elapsed_extra
    total = minute + (extra or 0) if minute is not None else None
    return {"minute": minute, "minute_extra": extra, "minute_total": total, "status": snapshot.status}


def scoreline_features(snapshot: Optional[LiveFixtureSnapshot]) -> dict[str, Any]:
    """'scoreline'."""
    if snapshot is None:
        return {"home_goals": None, "away_goals": None, "goal_diff": None, "total_goals": None}
    home, away = snapshot.home_goals, snapshot.away_goals
    diff = home - away if home is not None and away is not None else None
    total = home + away if home is not None and away is not None else None
    return {"home_goals": home, "away_goals": away, "goal_diff": diff, "total_goals": total}


def card_features_from_events(
    events: Iterable[LiveMatchEvent],
    home_team_id: Optional[int],
    away_team_id: Optional[int],
) -> dict[str, Any]:
    """'cards': conteggio timestamped dagli EVENTI. Il chiamante deve passare
    SOLO eventi gia' filtrati per `captured_at <= as_of`
    (`LiveFeatureStore.build_feature_row` lo garantisce) - questa funzione
    resta pura e non conosce `as_of`."""
    counts = {"yellow_cards_home": 0, "yellow_cards_away": 0, "red_cards_home": 0, "red_cards_away": 0}
    for event in events:
        if (event.event_type or "").strip().lower() != _CARD_EVENT_TYPE:
            continue
        if event.team_id == home_team_id:
            side = "home"
        elif event.team_id == away_team_id:
            side = "away"
        else:
            continue

        detail = (event.event_detail or "").strip().lower()
        if detail in _YELLOW_CARD_DETAILS:
            counts[f"yellow_cards_{side}"] += 1
        elif detail in _RED_CARD_DETAILS:
            counts[f"red_cards_{side}"] += 1

    counts["yellow_cards_diff"] = counts["yellow_cards_home"] - counts["yellow_cards_away"]
    counts["red_cards_diff"] = counts["red_cards_home"] - counts["red_cards_away"]
    return counts


def stat_features(
    stat_snapshots: Iterable[LiveFixtureStatSnapshot],
    home_team_id: Optional[int],
    away_team_id: Optional[int],
) -> dict[str, Any]:
    """'shots/xG se disponibili' (+ cards/corner/fouls/possession dalla
    stessa fonte grezza, vedi `_STAT_TYPE_FEATURE_MAP`). Il chiamante deve
    passare al piu' UNO snapshot per team (gia' filtrato/ultimo <= as_of)."""
    by_team: dict[Optional[int], LiveFixtureStatSnapshot] = {row.team_id: row for row in stat_snapshots}

    def _extract(raw: Optional[LiveFixtureStatSnapshot]) -> dict[str, float]:
        if raw is None or not isinstance(raw.stats, list):
            return {}
        values: dict[str, float] = {}
        for entry in raw.stats:
            mapped = _STAT_TYPE_FEATURE_MAP.get(str(entry.get("type") or "").strip().lower())
            if not mapped:
                continue
            numeric = _safe_float(entry.get("value"))
            if numeric is None:
                continue
            values[mapped] = numeric
        return values

    home_values = _extract(by_team.get(home_team_id))
    away_values = _extract(by_team.get(away_team_id))

    features: dict[str, Any] = {}
    for name in sorted(set(home_values) | set(away_values)):
        if name in home_values:
            features[f"{name}_home"] = home_values[name]
        if name in away_values:
            features[f"{name}_away"] = away_values[name]
        if name in home_values and name in away_values:
            features[f"{name}_diff"] = home_values[name] - away_values[name]
    return features


def pre_match_prior_features(predictions: Iterable[PredictionLedger]) -> dict[str, Any]:
    """'pre-match prior': probabilita' salvate PRIMA del kickoff nel
    Prediction Ledger (BET-06) - l'unica fonte di "cosa pensava l'Oracle
    prima che la partita iniziasse". Se per la stessa (market, outcome)
    esistono piu' prediction (`model_run_id` diversi/re-run), vince quella
    con `created_at` piu' recente."""
    latest_by_key: dict[tuple[str, str], PredictionLedger] = {}
    for prediction in predictions:
        key = (prediction.market, prediction.outcome)
        current = latest_by_key.get(key)
        if current is None:
            latest_by_key[key] = prediction
            continue
        current_at = _ensure_utc(current.created_at)
        candidate_at = _ensure_utc(prediction.created_at)
        if candidate_at is not None and (current_at is None or candidate_at > current_at):
            latest_by_key[key] = prediction

    features: dict[str, Any] = {}
    for (market, outcome), prediction in latest_by_key.items():
        prefix = f"prior__{_normalize_key(market)}__{_normalize_key(outcome)}"
        if prediction.p_model is not None:
            features[f"{prefix}__p_model"] = prediction.p_model
        if prediction.p_market_fair is not None:
            features[f"{prefix}__p_market_fair"] = prediction.p_market_fair
        if prediction.prob_edge is not None:
            features[f"{prefix}__prob_edge"] = prediction.prob_edge
    return features


@dataclass
class LiveFeatureRow:
    """Una riga di feature live per UNA fixture ad UN istante `as_of`."""

    fixture_id: int
    as_of: str
    feature_available_at_max: Optional[str]
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "fixture_id": self.fixture_id,
            "as_of": self.as_of,
            "feature_available_at_max": self.feature_available_at_max,
        }
        payload.update(self.features)
        return payload


class LiveFeatureStore:
    """Assembla `LiveFeatureRow` combinando dataset LIVE (LIVE-01) e prior
    pre-match (BET-06). Dependency injection di repository (stesso pattern
    di `LiveDataService`/`DataQualityService`) per restare testabile senza
    DB reale."""

    def __init__(
        self,
        live_repository: Optional[LiveDataRepository] = None,
        prediction_repository: Optional[PredictionLedgerRepository] = None,
    ):
        self.live_repository = live_repository or LiveDataRepository()
        self.prediction_repository = prediction_repository or PredictionLedgerRepository()

    def build_feature_row(self, fixture_id: int, as_of: Optional[datetime] = None) -> Optional[LiveFeatureRow]:
        """Ritorna `None` se non esiste ANCORA nessuno snapshot fixture con
        `captured_at <= as_of` (nessuna riga costruibile per quell'istante -
        stesso stile "riga non costruibile" di
        `PointInTimeDatasetBuilder._build_row`)."""
        as_of = _ensure_utc(as_of) or datetime.now(timezone.utc)
        fixture_id = int(fixture_id)

        snapshots = [
            row
            for row in self.live_repository.list_fixture_snapshots(fixture_id)
            if _ensure_utc(row.captured_at) is not None and _ensure_utc(row.captured_at) <= as_of
        ]
        if not snapshots:
            return None
        current_snapshot = max(snapshots, key=lambda row: _ensure_utc(row.captured_at))

        events = [
            row
            for row in self.live_repository.list_events_for_fixture(fixture_id)
            if _ensure_utc(row.captured_at) is not None and _ensure_utc(row.captured_at) <= as_of
        ]

        eligible_stats = [
            row
            for row in self.live_repository.list_stat_snapshots_for_fixture(fixture_id)
            if _ensure_utc(row.captured_at) is not None and _ensure_utc(row.captured_at) <= as_of
        ]
        latest_stats_by_team: dict[Optional[int], LiveFixtureStatSnapshot] = {}
        for row in sorted(eligible_stats, key=lambda item: _ensure_utc(item.captured_at)):
            latest_stats_by_team[row.team_id] = row

        predictions = [
            row
            for row in self.prediction_repository.list_for_fixture(fixture_id)
            if _ensure_utc(row.created_at) is not None and _ensure_utc(row.created_at) <= as_of
        ]

        features: dict[str, Any] = {}
        features.update(minute_features(current_snapshot))
        features.update(scoreline_features(current_snapshot))
        features.update(
            card_features_from_events(events, current_snapshot.home_team_id, current_snapshot.away_team_id)
        )
        features.update(
            stat_features(
                latest_stats_by_team.values(), current_snapshot.home_team_id, current_snapshot.away_team_id
            )
        )
        features.update(pre_match_prior_features(predictions))

        available_at_candidates = (
            [_ensure_utc(current_snapshot.captured_at)]
            + [_ensure_utc(row.captured_at) for row in events]
            + [_ensure_utc(row.captured_at) for row in latest_stats_by_team.values()]
            + [_ensure_utc(row.created_at) for row in predictions]
        )
        feature_available_at_max = max((ts for ts in available_at_candidates if ts is not None), default=None)

        return LiveFeatureRow(
            fixture_id=fixture_id,
            as_of=as_of.isoformat(),
            feature_available_at_max=feature_available_at_max.isoformat() if feature_available_at_max else None,
            features=features,
        )

    def build_feature_history(self, fixture_id: int) -> list[LiveFeatureRow]:
        """Una riga per OGNI snapshot fixture noto (storico completo, base
        per un futuro dataset di training LIVE-03): ogni riga ricalcola
        `as_of` = `captured_at` di quello snapshot, quindi resta point-in-time
        per costruzione (mai eventi/statistiche successivi a quello
        snapshot)."""
        fixture_id = int(fixture_id)
        snapshots = self.live_repository.list_fixture_snapshots(fixture_id)
        rows: list[LiveFeatureRow] = []
        for snapshot in snapshots:
            row = self.build_feature_row(fixture_id=fixture_id, as_of=snapshot.captured_at)
            if row is not None:
                rows.append(row)
        return rows

    @staticmethod
    def assert_no_leakage(rows: Iterable[LiveFeatureRow]) -> bool:
        """Stesso principio di `PointInTimeDatasetBuilder.assert_no_leakage`:
        nessuna riga puo' avere `feature_available_at_max` successivo al
        proprio `as_of` dichiarato."""
        for row in rows:
            if row.feature_available_at_max is None:
                continue
            if datetime.fromisoformat(row.feature_available_at_max) > datetime.fromisoformat(row.as_of):
                return False
        return True
