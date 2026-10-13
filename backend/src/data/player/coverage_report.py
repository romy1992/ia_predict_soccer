"""Report di COPERTURA dei dati per il mercato "giocatori che segnano" - SOLA LETTURA.

Risponde, prima di addestrare, a: i dati scaricati (eventi per giocatore e
formazioni) sono completi e coerenti? Un modello "ha segnato si/no" addestrato
su buchi impara i buchi, non il calcio.

Controlli, per lega e stagione, sulle sole fixture CONCLUSE nello scope:
  1. quante hanno gli eventi e quante le formazioni (`player_match_event`,
     `player_lineup`);
  2. qualita' delle formazioni: 22 titolari (11 + 11), ruolo e nome presenti;
  3. aggancio dei gol: ogni gol non-autogol deve avere `player_id` ed essere
     un giocatore presente nella formazione di quella fixture;
  4. coerenza col punteggio finale: gol+autogol negli eventi == gol totali di
     `match.score_home + score_away` (esclusi i rigori della lotteria finale);
  5. tassi di base per ruolo: quanti titolari segnano (serve a sapere con quale
     squilibrio di classi si addestra).

Nessuna scrittura, nessuna chiamata a API-Sports: solo SELECT sul DB.
Il modulo prende una `Session` (iniettabile, testabile con SQLite in-memory);
lo script `scripts/check_player_data_coverage.py` usa `SessionLocal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.service_ia.model.match import Match

LEAGUE_NAMES = {135: "Serie A", 39: "Premier League", 140: "La Liga", 78: "Bundesliga", 61: "Ligue 1"}
TOP5_LEAGUES = [135, 39, 140, 78, 61]

# Soglie sotto le quali un controllo viene segnalato come ATTENZIONE.
DEFAULT_THRESHOLDS = {
    "events_pct": 95.0,
    "lineups_pct": 95.0,
    "full_lineup_pct": 95.0,
    "goal_in_lineup_pct": 97.0,
    "score_consistent_pct": 95.0,
}

_POS_LABEL = {"G": "Portiere", "D": "Difensore", "M": "Centrocampista", "F": "Attaccante"}


def _is_scoreboard_goal():
    """Gol che entra nel punteggio: tipo Goal, non rigore sbagliato, non lotteria finale."""
    return and_(
        PlayerMatchEvent.event_type == "Goal",
        func.coalesce(PlayerMatchEvent.event_detail, "") != "Missed Penalty",
        func.coalesce(PlayerMatchEvent.comments, "") != "Penalty Shootout",
    )


def _is_scorer_goal():
    """Gol che conta per "ha segnato": come sopra ma senza autogol."""
    return and_(_is_scoreboard_goal(), func.coalesce(PlayerMatchEvent.event_detail, "") != "Own Goal")


def _scope(leagues: Sequence[int], seasons: Sequence[int], finished_statuses: Iterable[str]):
    return (
        select(
            Match.id_fixture.label("fixture_id"),
            Match.league_match.label("league"),
            Match.season.label("season"),
            Match.score_home.label("score_home"),
            Match.score_away.label("score_away"),
        )
        .where(
            Match.id_fixture.isnot(None),
            Match.league_match.in_(list(leagues)),
            Match.season.in_(list(seasons)),
            Match.status.in_(list(finished_statuses)),
        )
        .subquery("scope")
    )


def _frame(session: Session, stmt, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([tuple(r) for r in session.execute(stmt).all()], columns=columns)


@dataclass
class CoverageReport:
    by_competition: pd.DataFrame
    base_rates: pd.DataFrame
    anomalies: pd.DataFrame
    thresholds: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def warnings(self) -> list[str]:
        out: list[str] = []
        for _, row in self.by_competition.iterrows():
            for key, limit in self.thresholds.items():
                value = row[key]
                if pd.notna(value) and value < limit:
                    out.append(f"{row['lega']} {int(row['stagione'])}: {key} {value:.1f}% < {limit:.0f}%")
        return out


def build_report(
    session: Session,
    leagues: Sequence[int],
    seasons: Sequence[int],
    finished_statuses: Iterable[str],
    thresholds: dict[str, float] | None = None,
    max_anomalies: int = 15,
) -> CoverageReport:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    scope = _scope(leagues, seasons, finished_statuses)
    scope_ids = select(scope.c.fixture_id)

    fixtures = _frame(
        session,
        select(scope.c.fixture_id, scope.c.league, scope.c.season, scope.c.score_home, scope.c.score_away),
        ["fixture_id", "league", "season", "score_home", "score_away"],
    )
    if fixtures.empty:
        return CoverageReport(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), thresholds,
            ["Nessuna fixture conclusa nello scope: controlla leghe, stagioni e stati."],
        )

    lineup_agg = _frame(
        session,
        select(
            PlayerLineup.fixture_id,
            func.count().label("n"),
            func.sum(case((PlayerLineup.is_starter.is_(True), 1), else_=0)).label("starters"),
            func.sum(case((PlayerLineup.position.is_(None), 1), else_=0)).label("no_pos"),
            func.sum(case((PlayerLineup.player_name.is_(None), 1), else_=0)).label("no_name"),
        )
        .where(PlayerLineup.fixture_id.in_(scope_ids))
        .group_by(PlayerLineup.fixture_id),
        ["fixture_id", "lineup_n", "starters", "no_pos", "no_name"],
    )
    event_agg = _frame(
        session,
        select(
            PlayerMatchEvent.fixture_id,
            func.count().label("events"),
            func.sum(case((_is_scoreboard_goal(), 1), else_=0)).label("goal_events"),
        )
        .where(PlayerMatchEvent.fixture_id.in_(scope_ids))
        .group_by(PlayerMatchEvent.fixture_id),
        ["fixture_id", "events", "goal_events"],
    )
    lineup_pairs = PlayerLineup.__table__
    goal_match = _frame(
        session,
        select(
            PlayerMatchEvent.fixture_id,
            func.count().label("scorer_goals"),
            func.sum(case((PlayerMatchEvent.player_id.isnot(None), 1), else_=0)).label("with_player"),
            func.sum(case((lineup_pairs.c.player_id.isnot(None), 1), else_=0)).label("in_lineup"),
        )
        .select_from(PlayerMatchEvent)
        .outerjoin(
            lineup_pairs,
            and_(
                lineup_pairs.c.fixture_id == PlayerMatchEvent.fixture_id,
                lineup_pairs.c.player_id == PlayerMatchEvent.player_id,
            ),
        )
        .where(_is_scorer_goal(), PlayerMatchEvent.fixture_id.in_(scope_ids))
        .group_by(PlayerMatchEvent.fixture_id),
        ["fixture_id", "scorer_goals", "with_player", "in_lineup"],
    )

    df = (
        fixtures.merge(lineup_agg, on="fixture_id", how="left")
        .merge(event_agg, on="fixture_id", how="left")
        .merge(goal_match, on="fixture_id", how="left")
    )
    for col in ["lineup_n", "starters", "no_pos", "no_name", "events", "goal_events",
                "scorer_goals", "with_player", "in_lineup"]:
        df[col] = df[col].fillna(0)
    df["has_lineup"] = df["lineup_n"] > 0
    df["has_events"] = df["events"] > 0
    df["full_lineup"] = df["starters"] == 22
    scored = df["score_home"].notna() & df["score_away"].notna() & df["has_events"]
    df["score_total"] = df["score_home"].fillna(0) + df["score_away"].fillna(0)
    df["score_checked"] = scored
    df["score_ok"] = scored & (df["goal_events"] == df["score_total"])

    rows = []
    for (league, season), g in df.groupby(["league", "season"]):
        lineups = g[g["has_lineup"]]
        checked = g[g["score_checked"]]
        goals = g["scorer_goals"].sum()
        rows.append({
            "lega": LEAGUE_NAMES.get(int(league), str(int(league))),
            "stagione": int(season),
            "partite": len(g),
            "con_eventi": int(g["has_events"].sum()),
            "con_formazioni": int(g["has_lineup"].sum()),
            "events_pct": 100 * g["has_events"].mean(),
            "lineups_pct": 100 * g["has_lineup"].mean(),
            "full_lineup_pct": (100 * lineups["full_lineup"].mean()) if len(lineups) else float("nan"),
            "gol_non_autogol": int(goals),
            "goal_in_lineup_pct": (100 * g["in_lineup"].sum() / goals) if goals else float("nan"),
            "score_consistent_pct": (100 * checked["score_ok"].mean()) if len(checked) else float("nan"),
            "ruolo_mancante": int(g["no_pos"].sum()),
            "nome_mancante": int(g["no_name"].sum()),
        })
    by_competition = pd.DataFrame(rows).sort_values(["lega", "stagione"]).reset_index(drop=True)

    # Tassi di base: solo fixture che hanno SIA formazione SIA eventi, altrimenti un
    # giocatore di una partita senza eventi verrebbe contato come "non ha segnato".
    both_ids = df.loc[df["has_lineup"] & df["has_events"], "fixture_id"].tolist()
    base_rates = pd.DataFrame(columns=["ruolo", "ruolo_codice", "titolari", "segnato", "tasso_pct",
                                       "panchina", "panchina_segnato"])
    if both_ids:
        scorers = (
            select(PlayerMatchEvent.fixture_id.label("fixture_id"), PlayerMatchEvent.player_id.label("player_id"))
            .where(_is_scorer_goal(), PlayerMatchEvent.fixture_id.in_(both_ids))
            .distinct()
            .subquery("scorers")
        )
        raw = _frame(
            session,
            select(
                PlayerLineup.position,
                PlayerLineup.is_starter,
                func.count().label("n"),
                func.sum(case((scorers.c.player_id.isnot(None), 1), else_=0)).label("scored"),
            )
            .select_from(PlayerLineup)
            .outerjoin(
                scorers,
                and_(scorers.c.fixture_id == PlayerLineup.fixture_id,
                     scorers.c.player_id == PlayerLineup.player_id),
            )
            .where(PlayerLineup.fixture_id.in_(both_ids))
            .group_by(PlayerLineup.position, PlayerLineup.is_starter),
            ["pos", "starter", "n", "scored"],
        )
        out = []
        for pos, g in raw.groupby(raw["pos"].fillna("?")):
            s = g[g["starter"].astype(bool)]
            b = g[~g["starter"].astype(bool)]
            titolari, segnato = int(s["n"].sum()), int(s["scored"].sum())
            out.append({
                "ruolo": _POS_LABEL.get(pos, "Ruolo mancante" if pos == "?" else pos),
                "ruolo_codice": pos, "titolari": titolari, "segnato": segnato,
                "tasso_pct": (100 * segnato / titolari) if titolari else float("nan"),
                "panchina": int(b["n"].sum()), "panchina_segnato": int(b["scored"].sum()),
            })
        base_rates = pd.DataFrame(out).sort_values("titolari", ascending=False).reset_index(drop=True)
        tot_t, tot_s = int(base_rates["titolari"].sum()), int(base_rates["segnato"].sum())
        if tot_t:
            base_rates.loc[len(base_rates)] = {
                "ruolo": "TUTTI", "ruolo_codice": "-", "titolari": tot_t, "segnato": tot_s,
                "tasso_pct": 100 * tot_s / tot_t, "panchina": int(base_rates["panchina"].sum()),
                "panchina_segnato": int(base_rates["panchina_segnato"].sum()),
            }

    issues = df[df["has_events"] & df["has_lineup"] & (~df["full_lineup"] | (df["score_checked"] & ~df["score_ok"]))]
    anomalies = issues.assign(
        problema=issues.apply(
            lambda r: "; ".join(
                p for p in (
                    f"titolari={int(r['starters'])} (attesi 22)" if not r["full_lineup"] else "",
                    f"gol negli eventi={int(r['goal_events'])} vs punteggio={int(r['score_total'])}"
                    if (r["score_checked"] and not r["score_ok"]) else "",
                ) if p
            ),
            axis=1,
        )
    )[["fixture_id", "league", "season", "problema"]].head(max_anomalies).reset_index(drop=True)

    notes = [
        "Tassi di base calcolati solo sulle fixture che hanno sia formazione sia eventi.",
        "I panchinari che non sono entrati risultano \"non ha segnato\": per il training vanno esclusi "
        "(minuti giocati derivabili dalle sostituzioni, oggi non salvati).",
    ]
    return CoverageReport(by_competition, base_rates, anomalies, thresholds, notes)


def _fmt(df: pd.DataFrame, pct_cols: Sequence[str] = ()) -> str:
    if df.empty:
        return "(nessun dato)"
    shown = df.copy()
    for c in pct_cols:
        if c in shown:
            shown[c] = shown[c].map(lambda v: "n.d." if pd.isna(v) else f"{v:.1f}%")
    cols = list(shown.columns)
    widths = [max(len(str(c)), *(len(str(v)) for v in shown[c])) for c in cols]
    line = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(r) for r in shown.itertuples(index=False)])


def render_report(report: CoverageReport) -> str:
    parts = ["# Copertura dati marcatori (sola lettura)", ""]
    if report.by_competition.empty:
        return "\n".join(parts + report.notes)
    parts += [
        "## Per lega e stagione",
        _fmt(report.by_competition, ["events_pct", "lineups_pct", "full_lineup_pct",
                                      "goal_in_lineup_pct", "score_consistent_pct"]),
        "",
        "events_pct / lineups_pct: partite concluse con eventi / formazioni. full_lineup_pct: formazioni con 22 titolari.",
        "goal_in_lineup_pct: gol (non autogol) agganciati a un giocatore in formazione.",
        "score_consistent_pct: gol+autogol negli eventi == punteggio finale.",
        "",
        "## Tassi di base per ruolo (titolari che segnano almeno un gol non autogol)",
        _fmt(report.base_rates, ["tasso_pct"]),
        "",
    ]
    warns = report.warnings()
    parts.append("## Esito")
    parts += (["ATTENZIONE:"] + [f"- {w}" for w in warns]) if warns else ["Tutti i controlli sopra soglia."]
    if not report.anomalies.empty:
        parts += ["", "## Partite anomale (prime)", _fmt(report.anomalies)]
    parts += ["", "## Note"] + [f"- {n}" for n in report.notes]
    return "\n".join(parts)
