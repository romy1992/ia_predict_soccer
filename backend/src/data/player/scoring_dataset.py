"""Dataset "il giocatore segna?" (mercato marcatori) - una riga per giocatore e partita.

Costruito SOLO dai dati gia' a DB (`match`, `player_lineup`, `player_match_event`), nessuna
chiamata API e nessuna quota. Tutte le caratteristiche sono POINT-IN-TIME: per la partita P usano
soltanto partite con data strettamente precedente (mai P ne' il suo futuro).

Bersaglio `scored`: il giocatore ha segnato almeno un gol NON autogol nei tempi regolamentari
(minuto <= 90, recupero incluso). Rigori in partita contano, la lotteria finale e i supplementari no.

Righe escluse: portieri; giocatori che non sono mai entrati (panchina inutilizzata); partite senza
eventi, senza formazione o con formazione incompleta (titolari != 22).

Minuti giocati: derivati dagli eventi. Convenzione API-Sports per `subst`: `player_id` = chi ESCE,
`assist_id` = chi ENTRA (da confermare su una partita reale). Un rosso chiude la presenza.
Non ci sono i tiri per giocatore: non sono tra i dati scaricati.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.player_models import PlayerMatchEvent
from src.data.player.player_stats_models import PlayerMatchStats
from src.service_ia.model.match import Match

TARGET = "scored"
META_COLUMNS = ["fixture_id", "player_id", "team_id", "match_date", "season", "league"]
FEATURE_COLUMNS = [
    "pos_F", "pos_M", "pos_D", "is_starter", "is_home",
    "n_prior_apps", "prior_goals_last5", "prior_goals_last10", "prior_goals_last20",
    "rate_last10_per90", "rate_last10_shrunk", "rate_last20_shrunk",
    "starter_rate_last5", "avg_minutes_last5", "pens_last20",
    "team_gf10", "team_ga10", "opp_gf10", "opp_ga10",
    # v2: dai soli dati gia' a DB
    "goal_share_last10", "pen_taker_share_last20", "rest_days",
    "rate_samevenue_shrunk", "opp_conc_role10", "teammates_rate_sum",
]
# v2: statistiche per giocatore (tiri, passaggi chiave, rigori guadagnati, voto) dal backfill `player_match_stats`.
# Sono ASSENTI (NaN) per le partite senza statistiche: il modello le riempie con la mediana; `stats_games_l10`
# dice quante delle ultime 10 partite avevano statistiche.
STATS_FEATURES = [
    "shots_p90_l10", "shots_on_p90_l10", "shots_p90_l5", "key_passes_p90_l10",
    "pen_won_l20", "rating_l5", "stats_games_l10",
]
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS + STATS_FEATURES
STATS_RAW_COLUMNS = ["shots_total", "shots_on", "key_passes", "pen_won", "rating", "fouls_committed", "fouls_drawn"]
SHRINK_K = 3.0  # peso (in "partite da 90'") del tasso di ruolo nel tasso del giocatore
# Usati solo quando non esiste ancora nessuna storia del ruolo (inizio dataset).
FALLBACK_RATE_PER90 = {"F": 0.30, "M": 0.10, "D": 0.04}
REGULATION_END = 90


def _frame(session: Session, stmt, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([tuple(r) for r in session.execute(stmt).all()], columns=list(columns))


def load_raw(session: Session, leagues: Sequence[int], seasons: Sequence[int],
             finished_statuses: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = _frame(
        session,
        select(Match.id_fixture, Match.date_match, Match.league_match, Match.season,
               Match.id_team_home, Match.id_team_away)
        .where(Match.id_fixture.isnot(None), Match.league_match.in_(list(leagues)),
               Match.season.in_(list(seasons)), Match.status.in_(list(finished_statuses))),
        ["fixture_id", "date_match", "league", "season", "home_id", "away_id"],
    )
    ids = matches["fixture_id"].tolist()
    lineups = _frame(
        session,
        select(PlayerLineup.fixture_id, PlayerLineup.player_id, PlayerLineup.team_id,
               PlayerLineup.position, PlayerLineup.is_starter)
        .where(PlayerLineup.fixture_id.in_(ids)),
        ["fixture_id", "player_id", "team_id", "position", "is_starter"],
    )
    events = _frame(
        session,
        select(PlayerMatchEvent.fixture_id, PlayerMatchEvent.event_type, PlayerMatchEvent.event_detail,
               PlayerMatchEvent.comments, PlayerMatchEvent.elapsed_minute, PlayerMatchEvent.team_id,
               PlayerMatchEvent.player_id, PlayerMatchEvent.assist_id)
        .where(PlayerMatchEvent.fixture_id.in_(ids)),
        ["fixture_id", "event_type", "event_detail", "comments", "elapsed", "team_id", "player_id", "assist_id"],
    )
    return matches, lineups, events


def load_stats(session: Session, fixture_ids: Sequence[int]) -> pd.DataFrame:
    """Statistiche per giocatore e partita (`player_match_stats`). Vuoto se la tabella non ha righe."""
    cols = ["fixture_id", "player_id", *STATS_RAW_COLUMNS]
    frames = []
    ids = list(fixture_ids)
    for i in range(0, len(ids), 5000):  # a blocchi: la lista di ID puo' essere molto lunga
        stmt = select(PlayerMatchStats.fixture_id, PlayerMatchStats.player_id, PlayerMatchStats.shots_total,
                      PlayerMatchStats.shots_on, PlayerMatchStats.key_passes, PlayerMatchStats.pen_won,
                      PlayerMatchStats.rating, PlayerMatchStats.fouls_committed, PlayerMatchStats.fouls_drawn
                      ).where(PlayerMatchStats.fixture_id.in_(ids[i:i + 5000]))
        frames.append(_frame(session, stmt, cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def attach_stats(a: pd.DataFrame, stats: pd.DataFrame | None) -> pd.DataFrame:
    """Aggiunge le colonne grezze delle statistiche (NaN dove la partita non ne ha)."""
    if stats is None or stats.empty:
        return a.assign(**{c: np.nan for c in STATS_RAW_COLUMNS})
    keep = stats.drop_duplicates(["fixture_id", "player_id"])[["fixture_id", "player_id", *STATS_RAW_COLUMNS]]
    keep = keep.copy()
    keep[STATS_RAW_COLUMNS] = keep[STATS_RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return a.merge(keep, on=["fixture_id", "player_id"], how="left")


def rolling_per90(p: pd.DataFrame, col: str, window: int) -> pd.Series:
    """`col` per 90' nelle `window` partite PRECEDENTI del giocatore, contando solo le partite con statistiche."""
    has = p[col].notna()
    p = p.assign(_mins=p["minutes"].where(has))
    g = p.groupby("player_id", sort=False)
    num = g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    den = g["_mins"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    return (num / (den / 90.0)).where(den > 0)


def stats_coverage(p: pd.DataFrame, window: int = 10) -> pd.Series:
    """Quante delle `window` partite precedenti avevano statistiche (0..window)."""
    has = p["shots_total"].notna().astype(float)
    return has.groupby(p["player_id"], sort=False).transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0.0)


def _stats_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["shots_p90_l10"] = rolling_per90(p, "shots_total", 10)
    p["shots_on_p90_l10"] = rolling_per90(p, "shots_on", 10)
    p["shots_p90_l5"] = rolling_per90(p, "shots_total", 5)
    p["key_passes_p90_l10"] = rolling_per90(p, "key_passes", 10)
    g = p.groupby("player_id", sort=False)
    p["pen_won_l20"] = g["pen_won"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum())
    p["rating_l5"] = g["rating"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    p["stats_games_l10"] = stats_coverage(p)
    return p


def _appearances(matches: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Una riga per giocatore che e' stato davvero in campo, con minuti e gol."""
    events = events.assign(elapsed=pd.to_numeric(events["elapsed"], errors="coerce"))
    typ = events["event_type"].fillna("").str.lower()
    det = events["event_detail"].fillna("")
    com = events["comments"].fillna("")
    elapsed = events["elapsed"]

    subst = events[typ == "subst"]
    out_min = subst.dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"])["elapsed"].min().rename("minute_out")
    in_min = (subst.dropna(subset=["assist_id"]).rename(columns={"assist_id": "pid"})
              .groupby(["fixture_id", "pid"])["elapsed"].min().rename("minute_in"))
    in_min.index.names = ["fixture_id", "player_id"]
    reds = events[(typ == "card") & det.str.lower().isin(["red card", "second yellow card"])]
    red_min = reds.dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"])["elapsed"].min().rename("minute_red")

    goal_mask = ((typ == "goal") & (~det.isin(["Own Goal", "Missed Penalty"])) & (com != "Penalty Shootout")
                 & (elapsed <= REGULATION_END))
    goals = (events[goal_mask].dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).size().rename("goals"))
    pen_mask = ((typ == "goal") & (det == "Penalty") | (det == "Missed Penalty")) & (com != "Penalty Shootout")
    pens = events[pen_mask].dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).size().rename("pens")

    a = lineups.merge(matches[["fixture_id", "date_match", "league", "season", "home_id", "away_id"]], on="fixture_id")
    a = a.join(out_min, on=["fixture_id", "player_id"]).join(in_min, on=["fixture_id", "player_id"])
    a = a.join(red_min, on=["fixture_id", "player_id"]).join(goals, on=["fixture_id", "player_id"])
    a = a.join(pens, on=["fixture_id", "player_id"])
    a["goals"] = a["goals"].fillna(0).astype(int)
    a["pens"] = a["pens"].fillna(0).astype(int)
    a["is_starter"] = a["is_starter"].astype(bool)
    a["played"] = a["is_starter"] | a["minute_in"].notna()
    start = np.where(a["is_starter"], 0.0, a["minute_in"].astype(float))
    end = np.minimum(a[["minute_out", "minute_red"]].min(axis=1).fillna(REGULATION_END), REGULATION_END)
    a["minutes"] = np.where(a["played"], np.clip(end - start, 0, REGULATION_END), 0.0)
    a["match_date"] = pd.to_datetime(a["date_match"], utc=True, errors="coerce")
    a["is_home"] = (a["team_id"] == a["home_id"]).astype(int)
    a["opp_id"] = np.where(a["is_home"] == 1, a["away_id"], a["home_id"])
    a["scored"] = (a["goals"] > 0).astype(int)
    # Gol e rigori della squadra nella partita (somma dei suoi giocatori, per le quote di peso).
    a["team_goals"] = a.groupby(["fixture_id", "team_id"])["goals"].transform("sum")
    a["team_pens"] = a.groupby(["fixture_id", "team_id"])["pens"].transform("sum")
    return a.dropna(subset=["match_date"])


def _role_prior(played: pd.DataFrame, value_col: str = "goals",
                fallback_rates: dict[str, float] | None = None) -> pd.Series:
    """Eventi (default: gol) per 90' del ruolo calcolati SOLO su date precedenti (somme cumulate spostate di una data)."""
    daily = (played.groupby(["position", played["match_date"].dt.normalize()])[[value_col, "minutes"]].sum()
             .reset_index().rename(columns={"match_date": "day"}).sort_values(["position", "day"]))
    cum = daily.groupby("position")[[value_col, "minutes"]].cumsum()
    shifted = cum.groupby(daily["position"]).shift(1)
    daily["role_rate"] = shifted[value_col] / (shifted["minutes"] / 90.0)
    rate = played[["position"]].assign(day=played["match_date"].dt.normalize()).merge(
        daily[["position", "day", "role_rate"]], on=["position", "day"], how="left")["role_rate"]
    fallback = played["position"].map(fallback_rates or FALLBACK_RATE_PER90)
    return pd.Series(rate.to_numpy(), index=played.index).where(lambda s: s.notna() & (s > 0), fallback)


def _player_features(played: pd.DataFrame) -> pd.DataFrame:
    p = played.sort_values(["player_id", "match_date", "fixture_id"]).copy()
    p["role_rate"] = _role_prior(p)
    g = p.groupby("player_id", sort=False)

    def prior_sum(col: str, window: int) -> pd.Series:
        return g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    p["n_prior_apps"] = g.cumcount()
    for w in (5, 10, 20):
        p[f"prior_goals_last{w}"] = prior_sum("goals", w)
    minutes10, minutes20 = prior_sum("minutes", 10), prior_sum("minutes", 20)
    p["rate_last10_per90"] = (p["prior_goals_last10"] / (minutes10 / 90.0)).where(minutes10 > 0)
    p["rate_last10_shrunk"] = (p["prior_goals_last10"].fillna(0) + SHRINK_K * p["role_rate"]) / (
        minutes10.fillna(0) / 90.0 + SHRINK_K)
    p["rate_last20_shrunk"] = (p["prior_goals_last20"].fillna(0) + SHRINK_K * p["role_rate"]) / (
        minutes20.fillna(0) / 90.0 + SHRINK_K)
    starter = p["is_starter"].astype(float)
    p["_starter"] = starter
    p["starter_rate_last5"] = p.groupby("player_id", sort=False)["_starter"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    p["avg_minutes_last5"] = g["minutes"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    p["pens_last20"] = prior_sum("pens", 20)

    # --- v2 ---
    # Gol/rigori della squadra nelle stesse partite in cui il giocatore ha giocato.
    p["goal_share_last10"] = p["prior_goals_last10"] / prior_sum("team_goals", 10).where(lambda s: s > 0)
    p["pen_taker_share_last20"] = (p["pens_last20"] / prior_sum("team_pens", 20).where(lambda s: s > 0)).fillna(0.0)
    prev_date = g["match_date"].shift(1)
    p["rest_days"] = ((p["match_date"] - prev_date).dt.total_seconds() / 86400.0).clip(upper=30)
    # Tasso (gol/90') solo nelle partite nello stesso campo (casa/trasferta), con lo stesso restringimento.
    gv = p.groupby(["player_id", "is_home"], sort=False)
    gv_goals = gv["goals"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    gv_min = gv["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    p["rate_samevenue_shrunk"] = (gv_goals.fillna(0) + SHRINK_K * p["role_rate"]) / (gv_min.fillna(0) / 90.0 + SHRINK_K)
    return _stats_features(p) if "shots_total" in p.columns else p


def _team_form(matches: pd.DataFrame, events: pd.DataFrame,
               extra_fixture_ids: Iterable[int] = ()) -> pd.DataFrame:
    """Gol fatti/subiti (non autogol) nelle 10 partite precedenti, per squadra. Solo partite con eventi."""
    typ = events["event_type"].fillna("").str.lower()
    det = events["event_detail"].fillna("")
    com = events["comments"].fillna("")
    elapsed = pd.to_numeric(events["elapsed"], errors="coerce")
    goal = events[(typ == "goal") & (~det.isin(["Own Goal", "Missed Penalty"])) & (com != "Penalty Shootout")
                  & (elapsed <= REGULATION_END)]
    by_team = goal.groupby(["fixture_id", "team_id"]).size().rename("gf")
    with_events = set(events["fixture_id"].unique()) | set(extra_fixture_ids)
    m = matches[matches["fixture_id"].isin(with_events)].copy()
    m["match_date"] = pd.to_datetime(m["date_match"], utc=True, errors="coerce")
    rows = []
    for side, other in (("home_id", "away_id"), ("away_id", "home_id")):
        t = m[["fixture_id", "match_date", side, other]].rename(columns={side: "team_id", other: "opp_id"})
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(by_team.rename("gf").reset_index(), on=["fixture_id", "team_id"], how="left")
    t = t.merge(by_team.rename("ga").reset_index().rename(columns={"team_id": "opp_id"}),
                on=["fixture_id", "opp_id"], how="left")
    t[["gf", "ga"]] = t[["gf", "ga"]].fillna(0)
    t = t.sort_values(["team_id", "match_date", "fixture_id"])
    g = t.groupby("team_id", sort=False)
    t["team_gf10"] = g["gf"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    t["team_ga10"] = g["ga"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    return t[["fixture_id", "team_id", "team_gf10", "team_ga10"]]


def _opp_conceded_by_role(played: pd.DataFrame, form_pairs: pd.DataFrame) -> pd.DataFrame:
    """Gol subiti da ogni squadra per ruolo del marcatore (media sulle 10 partite precedenti)."""
    goals = played.groupby(["fixture_id", "team_id", "position"])["goals"].sum().rename("g").reset_index()
    grid = form_pairs.merge(pd.DataFrame({"position": ["F", "M", "D"]}), how="cross")
    grid = grid.merge(goals.rename(columns={"team_id": "opp_id", "g": "conc"}),
                      on=["fixture_id", "opp_id", "position"], how="left")
    grid["conc"] = grid["conc"].fillna(0)
    grid = grid.sort_values(["team_id", "position", "match_date", "fixture_id"])
    grid["opp_conc_role10"] = grid.groupby(["team_id", "position"], sort=False)["conc"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    # `team_id` e' la squadra che SUBISCE; per il giocatore sara' la squadra avversaria.
    return grid[["fixture_id", "team_id", "position", "opp_conc_role10"]].rename(columns={"team_id": "opp_id"})


VALID_ROLES = {"G", "D", "M", "F"}


def fill_missing_roles(matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Completa il ruolo mancante di un giocatore con quello che ha avuto nelle altre formazioni.

    Nelle coppe molte formazioni arrivano senza ruolo, ma lo stesso giocatore ha quasi sempre un ruolo in
    un'altra partita. Si usa l'ultimo ruolo noto PRIMA della partita; solo se non ce n'e', il primo ruolo
    noto dopo (il ruolo e' un attributo stabile, non un dato di risultato)."""
    out = lineups.copy()
    out["position"] = out["position"].where(out["position"].isin(VALID_ROLES))
    if out["position"].notna().all():
        return out
    dates = pd.to_datetime(matches.set_index("fixture_id")["date_match"], utc=True, errors="coerce")
    order = out.assign(_d=out["fixture_id"].map(dates)).sort_values(["player_id", "_d", "fixture_id"])
    grouped = order.groupby("player_id")["position"]
    filled = grouped.ffill()
    order["position"] = filled.groupby(order["player_id"]).bfill()
    out["position"] = order["position"].reindex(out.index)
    return out


def build_dataset(matches: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame,
                  upcoming_fixture_ids: Iterable[int] = (), stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dataset finale: META + FEATURE + TARGET, ordinato per data.

    `upcoming_fixture_ids`: partite NON ancora giocate (nessun evento) di cui si vuole la riga di
    previsione per i soli titolari della formazione ufficiale. Le feature restano point-in-time
    (usano solo il passato); `scored` vale 0 e non va usato."""
    upcoming = set(int(x) for x in upcoming_fixture_ids)
    if matches.empty or lineups.empty:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS_V2 + [TARGET])
    lineups = fill_missing_roles(matches, lineups)
    a = attach_stats(_appearances(matches, lineups, events), stats)
    played = a[a["played"]].copy()
    feats = _player_features(played)
    form = _team_form(matches, events, upcoming)
    pairs = (feats[["fixture_id", "team_id", "opp_id", "match_date"]].drop_duplicates())
    conc = _opp_conceded_by_role(played, pairs)
    feats = feats.merge(conc, on=["fixture_id", "opp_id", "position"], how="left")
    # Forza offensiva dei compagni titolari (somma del loro tasso restringimento, senza se stessi).
    st = feats["is_starter"].astype(bool)
    tot = feats.assign(_r=np.where(st, feats["rate_last20_shrunk"], 0.0)).groupby(
        ["fixture_id", "team_id"])["_r"].transform("sum")
    feats["teammates_rate_sum"] = tot - np.where(st, feats["rate_last20_shrunk"], 0.0)
    feats = feats.merge(form, on=["fixture_id", "team_id"], how="left")
    opp = form.rename(columns={"team_id": "opp_id", "team_gf10": "opp_gf10", "team_ga10": "opp_ga10"})
    feats = feats.merge(opp, on=["fixture_id", "opp_id"], how="left")

    # Fixture ammesse: eventi presenti e 22 titolari.
    starters = lineups[lineups["is_starter"].astype(bool)].groupby("fixture_id").size()
    ok = set(starters[starters == 22].index) & (set(events["fixture_id"].unique()) | upcoming)
    out = feats[feats["fixture_id"].isin(ok) & feats["position"].isin(["F", "M", "D"])].copy()
    for pos in ("F", "M", "D"):
        out[f"pos_{pos}"] = (out["position"] == pos).astype(int)
    out["is_starter"] = out["is_starter"].astype(int)
    cols = META_COLUMNS + FEATURE_COLUMNS_V2 + [TARGET]
    return out[cols].sort_values(["match_date", "fixture_id", "player_id"]).reset_index(drop=True)


def temporal_cv_splits(df: pd.DataFrame, n_folds: int = 3) -> list[tuple[list[int], list[int]]]:
    """Finestra espandibile per DATA: addestra sul passato, controlla sul blocco successivo.
    Una partita ha un solo istante, quindi non finisce mai a cavallo tra train e test.
    Restituisce posizioni (0..n-1) come richiede `cv=` di scikit-learn."""
    dates = df["match_date"].reset_index(drop=True)
    edges = dates.quantile(np.linspace(0, 1, n_folds + 2)).tolist()
    splits = []
    for k in range(1, n_folds + 1):
        train = dates.index[dates < edges[k]].tolist()
        test = dates.index[(dates >= edges[k]) & (dates < edges[k + 1] if k < n_folds else dates <= edges[k + 1])].tolist()
        if train and test:
            splits.append((train, test))
    return splits


def build_from_session(session: Session, leagues: Sequence[int], seasons: Sequence[int],
                       finished_statuses: Iterable[str]) -> pd.DataFrame:
    matches, lineups, events = load_raw(session, leagues, seasons, finished_statuses)
    stats = load_stats(session, matches["fixture_id"].tolist()) if not matches.empty else None
    return build_dataset(matches, lineups, events, stats=stats)
