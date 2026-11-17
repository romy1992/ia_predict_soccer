"""Estrae da un database `match_db` (tipicamente un dump ripristinato in locale)
il dataset compatto usato da `walk_forward_1x2_benchmark.py`.

NON tocca il database di produzione: serve un URL esplicito, in sola lettura.

Procedura usata per generare `_export/top5_matches_for_benchmark.csv.gz`
(2026-11-11, dump `src/backup_db/dump-match_db-20260803093803-08-26.backup`,
formato custom di PostgreSQL 17):

    createdb match_bench
    pg_restore --no-owner --no-privileges -d match_bench <dump>
    python scripts/analysis/export_top5_matches_for_benchmark.py \
        --database-url postgresql://<utente>:<password>@localhost:5433/match_bench

Una riga per partita di campionato conclusa (`league_match` nei 5 campionati;
`current_league` NON va usato come filtro: e' il campionato della squadra e
include le coppe). Le probabilita' di mercato sono no-vig: per ogni bookmaker
che quota TUTTI gli esiti si normalizzano le probabilita' implicite, poi si fa
la media fra i bookmaker. Nessun valore mancante viene sostituito.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text

TOP5_LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1"}
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "_export", "top5_matches_for_benchmark.csv.gz")

QUERY = text(
    """
    SELECT m.id_fixture, m.date_match, m.league_match AS league, m.season, m.round, m.status,
           m.id_team_home, m.id_team_away, m.name_home, m.name_away,
           sh.score_ft AS home_goals, sa.score_ft AS away_goals,
           sh.score_ht AS home_ht, sa.score_ht AS away_ht,
           sh.generic_statistics ->> 'expected_goals' AS home_xg,
           sa.generic_statistics ->> 'expected_goals' AS away_xg,
           o.odds_from, o.h2h::text AS h2h, o.under_over_2_5::text AS ou25, o.goal_no_goal::text AS btts
    FROM match m
    LEFT JOIN statistics sh ON sh.id_match = m.id_match_fk AND sh.statistics_team_id = m.id_team_home
    LEFT JOIN statistics sa ON sa.id_match = m.id_match_fk AND sa.statistics_team_id = m.id_team_away
    LEFT JOIN odds o ON o.id_match = m.id_match_fk
    WHERE m.league_match IN (39, 140, 135, 78, 61)
      AND m.status IN ('FT', 'AET', 'PEN')
    ORDER BY m.date_match, m.id_fixture
    """
)


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def no_vig_from_books(raw: Optional[str], outcome_prefixes: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """`raw` e' il JSON `{"<esito>_<bookmaker>": quota}`. `outcome_prefixes`
    mappa nome esito -> prefissi possibili della chiave (i due import storici,
    The Odds API e API-Sports, usano grafie diverse). Usa solo i bookmaker con
    tutti gli esiti quotati (> 1.0). `dispersion` e' il rapporto massimo/minimo
    fra le quote dei bookmaker sul primo esito: valori alti indicano quote non
    confrontabili fra loro (linee diverse finite sotto la stessa chiave)."""
    empty = {"books": 0, "overround": None, "dispersion": None, **{name: None for name in outcome_prefixes}}
    if not isinstance(raw, str) or raw in ("", "null", "{}"):
        return empty
    try:
        payload = json.loads(raw)
    except ValueError:
        return empty
    if not isinstance(payload, dict):
        return empty

    # Prefissi piu' lunghi per primi: "no_goal_" va riconosciuto prima di "goal_".
    ordered = sorted(
        ((name, prefix) for name, prefixes in outcome_prefixes.items() for prefix in prefixes),
        key=lambda item: -len(item[1]),
    )
    by_book: dict[str, dict[str, float]] = defaultdict(dict)
    for key, value in payload.items():
        odd = _to_float(value)
        if odd is None or odd <= 1.0:
            continue
        for name, prefix in ordered:
            if key.startswith(prefix):
                book = key[len(prefix):].lstrip("_")
                by_book[book][name] = odd
                break

    first_outcome = next(iter(outcome_prefixes))
    probs: dict[str, list[float]] = {name: [] for name in outcome_prefixes}
    overrounds: list[float] = []
    first_odds: list[float] = []
    for quotes in by_book.values():
        if set(quotes) != set(outcome_prefixes):
            continue
        implied = {name: 1.0 / odd for name, odd in quotes.items()}
        total = sum(implied.values())
        overrounds.append(total - 1.0)
        first_odds.append(quotes[first_outcome])
        for name, value in implied.items():
            probs[name].append(value / total)
    if not overrounds:
        return empty
    result: dict[str, Any] = {
        "books": len(overrounds),
        "overround": sum(overrounds) / len(overrounds),
        "dispersion": max(first_odds) / min(first_odds),
    }
    for name, values in probs.items():
        result[name] = sum(values) / len(values)
    return result


def build_frame(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        h2h = no_vig_from_books(row.h2h, {"home": ("home_",), "draw": ("draw_",), "away": ("away_",)})
        ou = no_vig_from_books(row.ou25, {"over": ("over 2.5_", "over_2.5_"), "under": ("under 2.5_", "under_2.5_")})
        btts = no_vig_from_books(row.btts, {"yes": ("goal_",), "no": ("no_goal_",)})
        records.append(
            {
                "id_fixture": row.id_fixture,
                "date_utc": row.date_match,
                "league": row.league,
                "season": row.season,
                "round": row.round,
                "home_team_id": row.id_team_home,
                "away_team_id": row.id_team_away,
                "home_name": row.name_home,
                "away_name": row.name_away,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "home_ht": row.home_ht,
                "away_ht": row.away_ht,
                "home_xg": _to_float(row.home_xg),
                "away_xg": _to_float(row.away_xg),
                "odds_source": row.odds_from,
                "h2h_books": h2h["books"],
                "p_mkt_home": h2h["home"],
                "p_mkt_draw": h2h["draw"],
                "p_mkt_away": h2h["away"],
                "h2h_overround": h2h["overround"],
                "h2h_dispersion": h2h["dispersion"],
                "ou25_books": ou["books"],
                "p_mkt_over25": ou["over"],
                "ou25_overround": ou["overround"],
                "ou25_dispersion": ou["dispersion"],
                "btts_books": btts["books"],
                "p_mkt_btts_yes": btts["yes"],
            }
        )
    frame = pd.DataFrame.from_records(records)
    for column in (
        "p_mkt_home", "p_mkt_draw", "p_mkt_away", "h2h_overround", "h2h_dispersion",
        "p_mkt_over25", "ou25_overround", "ou25_dispersion", "p_mkt_btts_yes",
    ):
        frame[column] = frame[column].astype(float).round(5)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=os.environ.get("BENCHMARK_DATABASE_URL"))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("Serve --database-url (o BENCHMARK_DATABASE_URL): un database locale ripristinato da dump.")

    engine = create_engine(args.database_url)
    with engine.connect() as connection:
        rows = pd.read_sql(QUERY, connection)
    frame = build_frame(rows)
    frame.to_csv(args.output, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{len(frame)} partite -> {args.output}")
    print(frame.groupby("season").agg(matches=("id_fixture", "size"), with_h2h=("h2h_books", lambda s: int((s > 0).sum())),
                                      with_ou25=("ou25_books", lambda s: int((s > 0).sum())),
                                      with_btts=("btts_books", lambda s: int((s > 0).sum())),
                                      with_xg=("home_xg", lambda s: int(s.notna().sum()))))


if __name__ == "__main__":
    main()
