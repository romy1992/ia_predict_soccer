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

    # Corners/Cards a linea configurabile (MARKET-05/06, 2026-09-13):
    # mercati INDIPENDENTI da quelli legacy "corners"/"cards" sopra (soglia
    # fissa) - stringhe hardcoded (non importate da corners_market.py/
    # cards_market.py per evitare un import circolare: quei moduli
    # importano gia' `FilterMarketService`) - devono restare in sync con
    # `DEFAULT_LINES`/`MARKET_NAME`/`_line_label` li' definiti.
    LINE_MARKETS = {
        "corners_line_8_5",
        "corners_line_9_5",
        "corners_line_10_5",
        "corners_line_11_5",
        "cards_line_3_5",
        "cards_line_4_5",
        "cards_line_5_5",
        "cards_line_6_5",
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

    # Prefisso delle feature quote AGGREGATE "legacy" (una media sola su
    # tutto il bucket del mercato, esiti mescolati). Restano emesse per non
    # degradare i modelli gia' promossi - i loro `feature_names` le elencano
    # e `PredictionSnapshotService` riempirebbe con 0.0 quelle mancanti,
    # azzerando in silenzio tutta l'informazione quote in serving. Il
    # training dei modelli NUOVI le esclude via `LEGACY_ODDS_FEATURES`.
    LEGACY_ODDS_FEATURES: frozenset[str] = frozenset(
        {"odds_count", "odds_mean", "odds_std", "odds_min", "odds_max"}
        | {f"odds_slot_{i}" for i in range(1, 11)}
    )

    @staticmethod
    def _split_outcome_and_bookmaker(key: str) -> tuple[str, str]:
        """Spezza una chiave quote nel formato prodotto da `map_odds()`,
        `f'{alternate_value}_{name_book}'`, nella coppia (esito, bookmaker).

        Lo split e' sull'ULTIMO underscore perche' l'esito ne contiene
        spesso uno (`no_goal_`, `goal_`) mentre il nome del bookmaker,
        nei dati osservati, non ne contiene mai (es. 'Bet365', '1xBet',
        'William Hill' - gli spazi restano spazi). Se il nome di un
        bookmaker contenesse un underscore lo split cadrebbe nel punto
        sbagliato: e' l'unico caso in cui questo parsing sbaglia, ed e'
        per questo che le chiavi anomale vanno verificate sui dati reali
        prima di addestrare.
        """
        outcome, separator, bookmaker = key.rpartition("_")
        if not separator:
            # Nessun underscore: chiave non conforme, l'intera stringa e'
            # trattata come esito e il bookmaker resta ignoto (mai scartata
            # in silenzio: finirebbe comunque in un gruppo esito suo).
            return key.strip(), ""
        return outcome.strip(), bookmaker.strip()

    # Prefissi che il provider antepone all'esito senza cambiarne il
    # significato: 'alternate over 1.5' e 'over 1.5' sono la STESSA
    # scommessa (2026-09-13, verificato sui dati - le due forme coprono
    # periodi disgiunti, `alternate` fino a giugno 2025 e la forma semplice
    # da agosto 2025, con distribuzioni di quota e correlazione col target
    # sovrapponibili). Analogamente 'corner over 8.5' e 'over 8.5' dentro il
    # mercato corners, 'card over 3.5' e 'over 3.5' dentro cards.
    #
    # Senza questa unificazione ogni feature per esito resta vuota sulla
    # meta' delle partite in cui il provider usava l'altro nome: su
    # Under/Over 1.5 la copertura passa dal 44% al 100%.
    _OUTCOME_ALIAS_PREFIXES: tuple[str, ...] = ("alternate", "corner", "card")

    @staticmethod
    def _normalize_outcome_name(outcome: str) -> str:
        """Slug stabile per comporre il nome della feature: 'over 2.5' ->
        'over_2_5', 'no_goal_' -> 'no_goal', '1X' -> '1x',
        'alternate over 1.5' -> 'over_1_5'."""
        slug = outcome.strip().lower()
        for char in (" ", ".", "-", "/"):
            slug = slug.replace(char, "_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug.strip("_")

        # Un solo prefisso alla volta e SOLO se resta qualcosa dopo: cosi'
        # l'esito 'card' di un ipotetico mercato binario non diventa vuoto.
        for prefisso in FilterMarketService._OUTCOME_ALIAS_PREFIXES:
            atteso = f"{prefisso}_"
            if slug.startswith(atteso) and len(slug) > len(atteso):
                return slug[len(atteso):]
        return slug

    @staticmethod
    def _extract_per_outcome_odds_features(market_odds: dict) -> dict:
        """Feature quote SEPARATE PER ESITO (2026-09-13, richiesto
        esplicitamente dall'operatore: "per ogni mercato devi prendere la
        media delle quote bookmakers... sia per il goal e no goal separati,
        under *.5 e over *.5 separati").

        `_extract_market_odds_features` calcola UNA media su tutto il bucket
        del mercato, che pero' contiene TUTTI gli esiti (home+draw+away per
        h2h, over+under per i totali, goal+no_goal per BTTS): la media
        risultante non corrisponde alla quota di nessuna scommessa reale
        (verificato sui dati: su h2h, con casa a 1.95 e trasferta a 4.20, al
        modello arrivava 3.23). Qui le quote vengono prima RAGGRUPPATE per
        esito leggendolo dalla chiave, poi aggregate dentro ogni gruppo -
        cioe' la media fra bookmaker DELLO STESSO esito.

        Oltre alle statistiche descrittive per esito produce:
        - `implied_prob_<esito>`: 1/quota media, la probabilita' implicita
          GREZZA (somma > 1 su tutti gli esiti: include il margine);
        - `overround`: la somma di quelle probabilita', cioe' il margine del
          bookmaker su questo mercato/partita;
        - `prob_norm_<esito>`: la probabilita' implicita normalizzata per
          l'overround, cioe' la stima del mercato ripulita dal margine -
          l'unica direttamente confrontabile con la probabilita' del modello.

        Ritorna `{}` se nessuna quota valida, come la funzione legacy.
        """
        grouped: dict[str, list[float]] = {}
        for key, raw_value in market_odds.items():
            value = FilterMarketService._safe_float(raw_value)
            if value <= 0:
                continue
            outcome, _bookmaker = FilterMarketService._split_outcome_and_bookmaker(str(key))
            slug = FilterMarketService._normalize_outcome_name(outcome)
            if not slug:
                continue
            grouped.setdefault(slug, []).append(value)

        if not grouped:
            return {}

        features: dict[str, float] = {}
        implied_by_outcome: dict[str, float] = {}
        for slug, values in grouped.items():
            mean = float(np.mean(values))
            features[f"odds_count_{slug}"] = float(len(values))
            features[f"odds_mean_{slug}"] = mean
            features[f"odds_std_{slug}"] = float(np.std(values))
            features[f"odds_min_{slug}"] = float(np.min(values))
            features[f"odds_max_{slug}"] = float(np.max(values))
            implied = 1.0 / mean
            features[f"implied_prob_{slug}"] = implied
            implied_by_outcome[slug] = implied

        overround = float(sum(implied_by_outcome.values()))
        features["overround"] = overround
        for slug, implied in implied_by_outcome.items():
            features[f"prob_norm_{slug}"] = implied / overround if overround > 0 else 0.0

        return features

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

        # Filtrare per linea non basta: dentro UNA linea restano comunque i
        # due lati ('over 8.5_X' e 'under 8.5_X' contengono entrambi "8.5" e
        # passano entrambi il filtro sopra). Le feature per esito separano
        # anche quelli; le legacy restano per i modelli gia' promossi.
        features = FilterMarketService._extract_market_odds_features(filtered)
        features.update(FilterMarketService._extract_per_outcome_odds_features(filtered))
        return features

    @staticmethod
    def _float_or_nan(value) -> float:
        """Come `_safe_float`, ma un dato ASSENTE resta assente (NaN) invece
        di diventare 0.0 (2026-09-13).

        `_safe_float` azzera i mancanti alla fonte, prima ancora che il
        DataFrame esista: per il modello "non lo sappiamo" diventa cosi'
        "squadra che non tira mai in porta". Su `expected_goals` (assente sul
        46% delle partite) e `goals_prevented` (73%) questo falsa proprio le
        statistiche piu' legate ai gol. Usato SOLO dal percorso di analisi
        (`build_dataset(fill_missing=False)`): il default resta l'azzeramento,
        perche' i modelli gia' registrati sono stati addestrati cosi'.
        """
        if value is None:
            return float("nan")
        if isinstance(value, str) and value.replace("%", "").strip() == "":
            return float("nan")
        return FilterMarketService._safe_float(value)

    @staticmethod
    def _extract_mean_features(match: dict, keep_missing: bool = False) -> dict:
        mean_home, mean_away = FilterMarketService._resolve_mean_stats(match)
        if not mean_home or not mean_away:
            return {}

        converti = FilterMarketService._float_or_nan if keep_missing else FilterMarketService._safe_float
        features = {}
        keys = set(mean_home.keys()).union(set(mean_away.keys()))
        for key in keys:
            if key == "id_team":
                continue

            home_v = converti(mean_home.get(key))
            away_v = converti(mean_away.get(key))
            normalized = FilterMarketService._normalize_feature_name(key)
            if normalized == "":
                continue

            features[f"{normalized}_home_stat"] = home_v
            features[f"{normalized}_away_stat"] = away_v
            features[f"{normalized}_diff_stat"] = home_v - away_v

        return features

    def _build_row(self, match: dict, market: str, with_target: bool, keep_missing: bool = False) -> Optional[dict]:
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

        # Entrambe di proposito: le legacy (esiti mescolati) tengono in vita
        # i modelli gia' promossi finche' non vengono riaddestrati, le nuove
        # per esito sono quelle che i modelli nuovi useranno davvero.
        row.update(self._extract_market_odds_features(market_odds))
        row.update(self._extract_per_outcome_odds_features(market_odds))
        row.update(self._extract_mean_features(match, keep_missing=keep_missing))

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

    def build_dataset(
        self, market: str, seasons: Optional[list[int]] = None, fill_missing: bool = True
    ) -> pd.DataFrame:
        """`fill_missing=False` (2026-09-13) restituisce i NaN cosi' come sono,
        invece di azzerarli.

        L'azzeramento di default e' storico e resta il comportamento per il
        training (i modelli registrati sono stati addestrati cosi'), ma rende
        impossibile QUALUNQUE analisi sui valori mancanti: dopo `fillna(0)`
        un dato assente e uno zero reale sono lo stesso numero. Su
        `expected_goals`, assente sul 46% delle partite, questo significa
        dare in pasto al modello "squadra che non tira mai in porta" invece
        di "non lo sappiamo" - e infatti la statistica piu' predittiva per i
        gol risulta penultima per importanza. Serve quindi un modo di
        estrarre il dataset GREZZO per l'EDA.
        """
        if market not in self.SUPPORTED_MARKETS:
            raise ValueError(f"Mercato non supportato: {market}")

        rows = []
        for match in self._search_matches(seasons=seasons, status="FT"):
            row = self._build_row(
                match=match, market=market, with_target=True, keep_missing=not fill_missing
            )
            if row:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        return df.fillna(0) if fill_missing else df

    def _fetch_match_dict(self, fixture_id: int) -> Optional[dict]:
        match = self.match_repo.filter_by(dict_search={"id_fixture": fixture_id}).first()
        if not match:
            return None
        return convert_orm_match_to_dict([match])[0]

    @staticmethod
    def _build_line_market_row(match: dict, market: str) -> Optional[dict]:
        """Riga di feature per un mercato Corners/Cards a linea configurabile
        (`LINE_MARKETS`) - import LOCALE (lazy) per evitare il ciclo
        d'importazione (`corners_market.py`/`cards_market.py` importano
        gia' `FilterMarketService` a livello di modulo). Cards usa l'indice
        arbitro CACHED (`current_referee_features_cached`, TTL 15 minuti,
        un replay condiviso per l'intera richiesta invece di uno per
        fixture) - unica eccezione al "nessuna query qui sotto" della
        pooled `_build_row`, ma economicamente innocua (cache hit nella
        stragrande maggioranza delle chiamate)."""
        if market.startswith("corners_line_"):
            from src.ml.markets.corners.corners_market import build_corners_prediction_row

            return build_corners_prediction_row(match)
        if market.startswith("cards_line_"):
            from src.ml.markets.cards.cards_market import build_cards_prediction_row, current_referee_features_cached

            referee_features = current_referee_features_cached(
                referee=str(match.get("referee") or ""), league=match.get("current_league")
            )
            return build_cards_prediction_row(match, referee_features=referee_features)
        return None

    def build_prediction_frames_from_match(self, match, markets: list[str]) -> dict[str, pd.DataFrame]:
        """Come `build_prediction_frame` ma per PIU' mercati sulla STESSA
        fixture GIA' caricata (`match`: ORM `Match` o dict gia' convertito),
        SENZA alcuna query (eccetto l'indice arbitro cached per i mercati
        Cards a linea configurabile, vedi `_build_line_market_row`):
        `_build_row` resta puro calcolo in-memory per tutti gli altri
        mercati.

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
            if market in self.LINE_MARKETS:
                row = self._build_line_market_row(match_dict, market)
            elif market in self.SUPPORTED_MARKETS:
                row = self._build_row(match=match_dict, market=market, with_target=False)
            else:
                continue
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
        if market not in self.SUPPORTED_MARKETS and market not in self.LINE_MARKETS:
            raise ValueError(f"Mercato non supportato: {market}")
        return self.build_prediction_frames(fixture_id=fixture_id, markets=[market]).get(market)





