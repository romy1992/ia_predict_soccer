from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict


class FilterMarketService:
    """Build training and prediction datasets for multiple betting markets."""

    SUPPORTED_MARKETS = {
        "h2h",
        "under_over_1_5",
        "under_over_2_5",
        "under_over_3_5",
        "under_over_4_5",
        "goal_no_goal",
        "corners",
        "cards",
        "dc",
    }

    def __init__(self):
        self.match_repo = MatchRepository()

    @staticmethod
    def _safe_float(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            clean = value.replace("%", "").strip()
            if clean == "":
                return 0.0
            try:
                return float(clean)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _normalize_feature_name(key: str) -> str:
        key = key.strip().lower()
        key = re.sub(r"[^a-z0-9]+", "_", key)
        return key.strip("_")

    def _search_matches(self, seasons: Optional[list[int]] = None, status: str = "FT") -> list[dict]:
        filters = {
            "mean_statistics": "not None",
            "odds": "not None",
            "statistics": "not None",
            "status": [status],
        }
        if seasons:
            filters["season"] = seasons

        records = self.match_repo.search_filter(filters=filters)
        return convert_orm_match_to_dict(records)

    @staticmethod
    def _resolve_team_stats(match: dict, with_full_stats: bool = True):
        stats = match.get("statistics") or []
        if len(stats) < 2 and with_full_stats:
            return None, None

        id_home = match.get("id_team_home")
        id_away = match.get("id_team_away")

        stat_home = next((s for s in stats if s.get("statistics_team_id") == id_home), None)
        stat_away = next((s for s in stats if s.get("statistics_team_id") == id_away), None)
        return stat_home, stat_away

    @staticmethod
    def _resolve_mean_stats(match: dict):
        mean_stats = match.get("mean_statistics")
        id_home = match.get("id_team_home")
        id_away = match.get("id_team_away")

        if isinstance(mean_stats, list) and len(mean_stats) >= 2:
            mean_home = next((m for m in mean_stats if m.get("id_team") == id_home), None)
            mean_away = next((m for m in mean_stats if m.get("id_team") == id_away), None)
            return mean_home, mean_away

        # Se il JSON e' salvato come singolo dict non e' possibile ricostruire entrambe le squadre.
        return None, None

    @staticmethod
    def _label_by_market(market: str, stat_home: dict, stat_away: dict) -> Optional[int]:
        home_ft = stat_home.get("score_ft") if stat_home else None
        away_ft = stat_away.get("score_ft") if stat_away else None
        if home_ft is None or away_ft is None:
            return None

        total_goals = int(home_ft) + int(away_ft)

        if market == "h2h":
            return 1 if int(home_ft) > int(away_ft) else 0
        if market == "goal_no_goal":
            return 1 if int(home_ft) > 0 and int(away_ft) > 0 else 0
        if market == "dc":
            return 1 if int(home_ft) >= int(away_ft) else 0  # 1X
        if market.startswith("under_over_"):
            threshold = float(market.split("under_over_")[1].replace("_", "."))
            return 1 if total_goals > threshold else 0
        if market == "corners":
            total_corners = int(stat_home.get("corners") or 0) + int(stat_away.get("corners") or 0)
            return 1 if total_corners >= 10 else 0
        if market == "cards":
            home_cards = int(stat_home.get("yellow_cards") or 0) + int(stat_home.get("red_cards") or 0)
            away_cards = int(stat_away.get("yellow_cards") or 0) + int(stat_away.get("red_cards") or 0)
            return 1 if (home_cards + away_cards) >= 5 else 0

        return None

    @staticmethod
    def _extract_market_odds_features(market_odds: dict) -> dict:
        odds_values = [FilterMarketService._safe_float(v) for v in market_odds.values() if v is not None]
        odds_values = [v for v in odds_values if v > 0]
        if not odds_values:
            return {}

        features = {
            "odds_count": float(len(odds_values)),
            "odds_mean": float(np.mean(odds_values)),
            "odds_std": float(np.std(odds_values)),
            "odds_min": float(np.min(odds_values)),
            "odds_max": float(np.max(odds_values)),
        }

        # Le prime 10 quote ordinate mantengono anche informazione di forma distribuzione.
        sorted_values = sorted(odds_values)
        for idx in range(10):
            slot = sorted_values[idx] if idx < len(sorted_values) else 0.0
            features[f"odds_slot_{idx + 1}"] = float(slot)

        return features

    _ODDS_LINE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

    @staticmethod
    def _extract_line_from_odds_key(key: str) -> Optional[float]:
        """Estrae il valore numerico di linea da una chiave quote tipo
        'over 8.5_bet365'/'under 9.5_pinnacle' (formato prodotto da
        `map_odds()` per Corners/Cards: `f'{alternate_value}_{name_book}'`,
        dove `alternate_value` e' la stringa RAW dell'esito dal provider,
        es. 'over 8.5' - la linea compare SEMPRE prima del nome bookmaker,
        quindi il primo numero decimale trovato e' sempre quello giusto,
        indipendentemente da eventuali cifre nel nome del bookmaker)."""
        match = FilterMarketService._ODDS_LINE_PATTERN.search(key)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_line_specific_odds_features(market_odds: dict, line: float, tolerance: float = 0.01) -> dict:
        """Come `_extract_market_odds_features`, ma filtra PRIMA le chiavi
        alla sola linea richiesta (2026-09-12, scoperto investigando perche'
        Corners/Cards (MARKET-05/06, linea configurabile) avessero un AUC
        vicino al coin-flip): a differenza dei mercati Under/Over gol (dove
        `map_odds()` separa gia' le quote per soglia in bucket dedicati,
        `under_over_1_5`/`_2_5`/...), per 'Corners Over Under'/'Cards
        Over/Under' l'ingestion mette TUTTE le linee (8.5/9.5/10.5/11.5,
        entrambi i lati Over/Under, tutti i bookmaker) in un UNICO bucket
        piatto ('corners'/'cards') - la linea resta identificabile SOLO nella
        chiave (mai estratta prima d'ora per queste feature aggregate, a
        differenza di quanto gia' fatto per `OddsSnapshot` via
        `_extract_line_value` in `download_match_service.py`). Il pooling
        indiscriminato attuale mischia quote di linee radicalmente diverse
        (es. "quasi certo" per una linea bassa insieme a "quasi impossibile"
        per una alta) nello stesso odds_mean/std/slot - rumore, non segnale,
        per il modello di QUALUNQUE singola linea."""
        filtered = {}
        for key, value in market_odds.items():
            extracted_line = FilterMarketService._extract_line_from_odds_key(key)
            if extracted_line is not None and abs(extracted_line - line) < tolerance:
                filtered[key] = value
        return FilterMarketService._extract_market_odds_features(filtered)

    @staticmethod
    def _extract_mean_features(match: dict) -> dict:
        mean_home, mean_away = FilterMarketService._resolve_mean_stats(match)
        if not mean_home or not mean_away:
            return {}

        features = {}
        keys = set(mean_home.keys()).union(set(mean_away.keys()))
        for key in keys:
            if key == "id_team":
                continue

            home_v = FilterMarketService._safe_float(mean_home.get(key))
            away_v = FilterMarketService._safe_float(mean_away.get(key))
            normalized = FilterMarketService._normalize_feature_name(key)
            if normalized == "":
                continue

            features[f"{normalized}_home_stat"] = home_v
            features[f"{normalized}_away_stat"] = away_v
            features[f"{normalized}_diff_stat"] = home_v - away_v

        return features

    def _build_row(self, match: dict, market: str, with_target: bool) -> Optional[dict]:
        odds_list = match.get("odds") or []
        if not odds_list:
            return None

        market_odds = (odds_list[0] or {}).get(market)
        if not isinstance(market_odds, dict) or len(market_odds) == 0:
            return None

        row = {
            "id_fixture": match.get("id_fixture"),
            "season": match.get("season"),
            "league": match.get("current_league"),
            "market": market,
            "prediction_at": match.get("date_match"),
        }

        row.update(self._extract_market_odds_features(market_odds))
        row.update(self._extract_mean_features(match))

        # Senza feature utili non ha senso produrre la riga.
        if len(row) <= 3:
            return None

        if with_target:
            stat_home, stat_away = self._resolve_team_stats(match=match, with_full_stats=True)
            if not stat_home or not stat_away:
                return None

            target = self._label_by_market(market=market, stat_home=stat_home, stat_away=stat_away)
            if target is None:
                return None
            row["y"] = int(target)

        return row

    def build_dataset(self, market: str, seasons: Optional[list[int]] = None) -> pd.DataFrame:
        if market not in self.SUPPORTED_MARKETS:
            raise ValueError(f"Mercato non supportato: {market}")

        rows = []
        for match in self._search_matches(seasons=seasons, status="FT"):
            row = self._build_row(match=match, market=market, with_target=True)
            if row:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def _fetch_match_dict(self, fixture_id: int) -> Optional[dict]:
        match = self.match_repo.filter_by(dict_search={"id_fixture": fixture_id}).first()
        if not match:
            return None
        return convert_orm_match_to_dict([match])[0]

    def build_prediction_frames_from_match(self, match, markets: list[str]) -> dict[str, pd.DataFrame]:
        """Come `build_prediction_frame` ma per PIU' mercati sulla STESSA
        fixture GIA' caricata (`match`: ORM `Match` o dict gia' convertito),
        SENZA alcuna query: `_build_row` e' puro calcolo in-memory (nessun
        accesso a `self.match_repo`/DB qui sotto).

        Fix performance (cambio giorno lento in Dashboard): il chiamante puo'
        riusare un `Match` gia' caricato in BATCH altrove (es.
        `DashboardService._fetch_matches`, una query SELECT...IN unica per
        l'intera finestra di date, con `statistics`/`odds` gia' "lazy=selectin")
        invece di rifare una query dedicata per fixture - vedi
        `build_prediction_frames`/`build_prediction_frame` sotto per il
        percorso "serve ancora una query" (fixture non gia' in mano)."""
        frames: dict[str, pd.DataFrame] = {}
        if not markets:
            return frames

        match_dict = match if isinstance(match, dict) else convert_orm_match_to_dict([match])[0]
        for market in markets:
            if market not in self.SUPPORTED_MARKETS:
                continue
            row = self._build_row(match=match_dict, market=market, with_target=False)
            if not row:
                continue
            frames[market] = pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan).fillna(0)
        return frames

    def build_prediction_frames(self, fixture_id: int, markets: list[str]) -> dict[str, pd.DataFrame]:
        """Come `build_prediction_frame` ma per PIU' mercati in un colpo
        solo: la query Match (+ relazioni statistics/odds) viene eseguita
        UNA VOLTA SOLA e riusata in memoria per costruire la riga di
        ciascun mercato, invece di ripetere le stesse query per OGNI
        mercato (fix performance: con le 9 SUPPORTED_MARKETS, il vecchio
        `build_prediction_frame` chiamato in loop da
        `DashboardService._predict_fixture` faceva 9x le query per singola
        fixture - causa principale della lentezza al cambio giorno in
        Dashboard, con centinaia di round-trip DB per un giorno con decine
        di fixture)."""
        match_dict = self._fetch_match_dict(fixture_id)
        if not match_dict:
            return {}
        return self.build_prediction_frames_from_match(match_dict, markets)

    def build_prediction_frame(self, market: str, fixture_id: int) -> Optional[pd.DataFrame]:
        if market not in self.SUPPORTED_MARKETS:
            raise ValueError(f"Mercato non supportato: {market}")
        return self.build_prediction_frames(fixture_id=fixture_id, markets=[market]).get(market)





