"""Dataset "il giocatore prende un giallo?" (mercato cartellini giocatori) - una riga per giocatore e partita.

Stessa impostazione del dataset marcatori (`scoring_dataset.py`, di cui riusa minuti/presenze/ruoli):
costruito SOLO dai dati gia' a DB (`match`, `player_lineup`, `player_match_event`), nessuna chiamata API
e nessuna quota; caratteristiche POINT-IN-TIME (per la partita P solo partite precedenti).

Bersaglio `yellow`: il giocatore ha ricevuto almeno un'ammonizione nei tempi regolamentari (minuto <= 90,
recupero incluso). Il secondo giallo conta (e' comunque un'ammonizione); un rosso diretto no.
Sono solo i gialli: i rossi sono troppo rari per un modello affidabile.

Righe escluse: portieri; giocatori mai entrati; partite senza eventi o con formazione incompleta (titolari != 22).

Specifico dei cartellini: l'ARBITRO (media di gialli per partita dei suoi match precedenti, ristretta verso la
media generale) e i gialli di squadra/avversario. I falli per giocatore (`player_match_stats`) NON sono usati
in questa versione: si aggiungeranno quando il backfill delle statistiche e' completo.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.player.scoring_dataset import (
    META_COLUMNS, REGULATION_END, SHRINK_K, _appearances, _role_prior, fill_missing_roles, load_raw,
)
from src.service_ia.model.match import Match

TARGET = "yellow"
FEATURE_COLUMNS = [
    "pos_F", "pos_M", "pos_D", "is_starter", "is_home",
    "n_prior_apps", "prior_yellows_last5", "prior_yellows_last10", "prior_yellows_last20",
    "rate_last10_shrunk", "rate_last20_shrunk", "rate_samevenue_shrunk",
    "starter_rate_last5", "avg_minutes_last5", "rest_days",
    "team_yc10", "opp_yc10", "ref_yc_per_match", "ref_known",
]
YELLOW_DETAILS = {"yellow card", "second yellow card"}
# Usati solo quando non esiste ancora nessuna storia del ruolo (inizio dataset): gialli per 90'.
FALLBACK_YELLOW_PER90 = {"F": 0.10, "M": 0.17, "D": 0.15}
REFEREE_SHRINK_K = 5.0  # peso (in "partite arbitrate") della media generale nella media dell'arbitro
FALLBACK_TOTAL_YELLOWS = 3.8  # gialli totali per partita, solo se non c'e' ancora nessuna partita nel dataset


def _yellow_events(events: pd.DataFrame) -> pd.DataFrame:
    typ = events["event_type"].fillna("").str.lower()
    det = events["event_detail"].fillna("").str.lower()
    elapsed = pd.to_numeric(events["elapsed"], errors="coerce")
    return events[(typ == "card") & det.isin(YELLOW_DETAILS) & (elapsed <= REGULATION_END)]


def load_referees(session: Session, fixture_ids: Sequence[int]) -> pd.DataFrame:
    rows = session.execute(select(Match.id_fixture, Match.referee).where(Match.id_fixture.in_(list(fixture_ids)))).all()
    return pd.DataFrame([tuple(r) for r in rows], columns=["fixture_id", "referee"])


def _player_features(played: pd.DataFrame) -> pd.DataFrame:
    p = played.sort_values(["player_id", "match_date", "fixture_id"]).copy()
    p["role_rate"] = _role_prior(p, "yellows", FALLBACK_YELLOW_PER90)
    g = p.groupby("player_id", sort=False)

    def prior_sum(col: str, window: int) -> pd.Series:
        return g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    p["n_prior_apps"] = g.cumcount()
    for w in (5, 10, 20):
        p[f"prior_yellows_last{w}"] = prior_sum("yellows", w)
    minutes10, minutes20 = prior_sum("minutes", 10), prior_sum("minutes", 20)
    p["rate_last10_shrunk"] = (p["prior_yellows_last10"].fillna(0) + SHRINK_K * p["role_rate"]) / (
        minutes10.fillna(0) / 90.0 + SHRINK_K)
    p["rate_last20_shrunk"] = (p["prior_yellows_last20"].fillna(0) + SHRINK_K * p["role_rate"]) / (
        minutes20.fillna(0) / 90.0 + SHRINK_K)
    p["_starter"] = p["is_starter"].astype(float)
    p["starter_rate_last5"] = p.groupby("player_id", sort=False)["_starter"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    p["avg_minutes_last5"] = g["minutes"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    prev_date = g["match_date"].shift(1)
    p["rest_days"] = ((p["match_date"] - prev_date).dt.total_seconds() / 86400.0).clip(upper=30)
    gv = p.groupby(["player_id", "is_home"], sort=False)
    gv_y = gv["yellows"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    gv_min = gv["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    p["rate_samevenue_shrunk"] = (gv_y.fillna(0) + SHRINK_K * p["role_rate"]) / (gv_min.fillna(0) / 90.0 + SHRINK_K)
    return p


def _team_yellow_form(matches: pd.DataFrame, events: pd.DataFrame, extra_fixture_ids: Iterable[int] = ()) -> pd.DataFrame:
    """Gialli presi dalla squadra nelle 10 partite precedenti (solo partite con eventi)."""
    by_team = _yellow_events(events).groupby(["fixture_id", "team_id"]).size().rename("yc").reset_index()
    with_events = set(events["fixture_id"].unique()) | set(extra_fixture_ids)
    m = matches[matches["fixture_id"].isin(with_events)].copy()
    m["match_date"] = pd.to_datetime(m["date_match"], utc=True, errors="coerce")
    t = pd.concat(
        [m[["fixture_id", "match_date", side]].rename(columns={side: "team_id"}) for side in ("home_id", "away_id")],
        ignore_index=True,
    )
    t = t.merge(by_team, on=["fixture_id", "team_id"], how="left")
    t["yc"] = t["yc"].fillna(0)
    t = t.sort_values(["team_id", "match_date", "fixture_id"])
    t["team_yc10"] = t.groupby("team_id", sort=False)["yc"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    return t[["fixture_id", "team_id", "team_yc10"]]


def _referee_form(matches: pd.DataFrame, events: pd.DataFrame, referees: pd.DataFrame) -> pd.DataFrame:
    """Gialli totali per partita dei match PRECEDENTI dello stesso arbitro, ristretti verso la media generale."""
    fx = matches[["fixture_id", "date_match"]].copy()
    fx["match_date"] = pd.to_datetime(fx["date_match"], utc=True, errors="coerce")
    totals = _yellow_events(events).groupby("fixture_id").size().rename("yc_total")
    has_events = set(events["fixture_id"].unique())
    fx = fx.join(totals, on="fixture_id")
    fx["has"] = fx["fixture_id"].isin(has_events).astype(int)
    fx["yc"] = fx["yc_total"].fillna(0)
    fx = fx.merge(referees.drop_duplicates("fixture_id"), on="fixture_id", how="left")
    fx["referee"] = fx["referee"].fillna("").astype(str).str.strip()
    fx = fx.dropna(subset=["match_date"]).sort_values(["match_date", "fixture_id"]).reset_index(drop=True)

    g_sum = fx["yc"].cumsum() - fx["yc"]
    g_n = fx["has"].cumsum() - fx["has"]
    global_mean = (g_sum / g_n.where(g_n > 0)).fillna(FALLBACK_TOTAL_YELLOWS)
    named = fx["referee"] != ""
    grouped = fx.groupby("referee", sort=False)
    r_sum = (grouped["yc"].cumsum() - fx["yc"]).where(named, 0.0)
    r_n = (grouped["has"].cumsum() - fx["has"]).where(named, 0)
    fx["ref_yc_per_match"] = (r_sum + REFEREE_SHRINK_K * global_mean) / (r_n + REFEREE_SHRINK_K)
    fx["ref_known"] = (r_n > 0).astype(int)
    return fx[["fixture_id", "ref_yc_per_match", "ref_known"]]


def build_dataset(matches: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame,
                  referees: pd.DataFrame | None = None,
                  upcoming_fixture_ids: Iterable[int] = ()) -> pd.DataFrame:
    """Dataset finale: META + FEATURE + TARGET, ordinato per data.

    `upcoming_fixture_ids`: partite NON ancora giocate di cui si vuole la riga di previsione per i soli
    titolari della formazione ufficiale (feature solo dal passato; `yellow` vale 0 e non va usato)."""
    upcoming = set(int(x) for x in upcoming_fixture_ids)
    if matches.empty or lineups.empty:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS + [TARGET])
    referees = referees if referees is not None else pd.DataFrame(columns=["fixture_id", "referee"])
    lineups = fill_missing_roles(matches, lineups)
    a = _appearances(matches, lineups, events)
    yellows = _yellow_events(events).dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).size().rename("yellows")
    a = a.join(yellows, on=["fixture_id", "player_id"])
    a["yellows"] = a["yellows"].fillna(0).astype(int)
    a[TARGET] = (a["yellows"] > 0).astype(int)
    played = a[a["played"]].copy()
    feats = _player_features(played)

    form = _team_yellow_form(matches, events, upcoming)
    feats = feats.merge(form, on=["fixture_id", "team_id"], how="left")
    opp = form.rename(columns={"team_id": "opp_id", "team_yc10": "opp_yc10"})
    feats = feats.merge(opp, on=["fixture_id", "opp_id"], how="left")
    feats = feats.merge(_referee_form(matches, events, referees), on="fixture_id", how="left")

    starters = lineups[lineups["is_starter"].astype(bool)].groupby("fixture_id").size()
    ok = set(starters[starters == 22].index) & (set(events["fixture_id"].unique()) | upcoming)
    out = feats[feats["fixture_id"].isin(ok) & feats["position"].isin(["F", "M", "D"])].copy()
    for pos in ("F", "M", "D"):
        out[f"pos_{pos}"] = (out["position"] == pos).astype(int)
    out["is_starter"] = out["is_starter"].astype(int)
    cols = META_COLUMNS + FEATURE_COLUMNS + [TARGET]
    return out[cols].sort_values(["match_date", "fixture_id", "player_id"]).reset_index(drop=True)


def build_from_session(session: Session, leagues: Sequence[int], seasons: Sequence[int],
                       finished_statuses: Iterable[str]) -> pd.DataFrame:
    matches, lineups, events = load_raw(session, leagues, seasons, finished_statuses)
    referees = load_referees(session, matches["fixture_id"].tolist()) if not matches.empty else None
    return build_dataset(matches, lineups, events, referees)
