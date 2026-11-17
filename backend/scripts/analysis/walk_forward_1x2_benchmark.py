"""Benchmark walk-forward OFFLINE su 1X2, Over/Under 2.5 e Goal/No Goal
per i cinque campionati principali (Premier League, La Liga, Serie A,
Bundesliga, Ligue 1).

NON e' codice di produzione: non legge il bucket, non scrive sul registry, non
promuove nulla. Serve a rispondere, con numeri riproducibili, a una domanda
sola: quali famiglie di modelli battono davvero - fuori campione - il mercato
e i baseline semplici?

Dati: `_export/top5_matches_for_benchmark.csv.gz`, estratto da un dump locale
con `export_top5_matches_for_benchmark.py` (vedi li' la procedura).

Protocollo
- Ordine cronologico rigoroso. Ogni previsione usa solo partite giocate PRIMA
  del giorno della partita (Elo, forma) o prima dell'inizio del mese (Poisson
  e Dixon-Coles, rifittati a inizio mese per campionato).
- Modelli supervisionati: finestra espandibile, un fold per trimestre solare.
- Iperparametri di Elo e del decadimento temporale scelti SOLO sul periodo che
  precede il primo fold di test.
- Confronti con il mercato: bootstrap appaiato sulle stesse partite.

Uso:
    python scripts/analysis/walk_forward_1x2_benchmark.py
    python scripts/analysis/walk_forward_1x2_benchmark.py --bootstrap-resamples 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "_export", "top5_matches_for_benchmark.csv.gz")
DEFAULT_OUTPUT = os.path.join(HERE, "walk_forward_1x2_benchmark_result.json")

SEED = 42
LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1"}
TEST_START = pd.Timestamp("2021-07-01", tz="UTC")
SELECTION_START = pd.Timestamp("2018-07-01", tz="UTC")
SUPERVISED_HISTORY_START = pd.Timestamp("2016-07-01", tz="UTC")
MIN_TRAIN_ROWS = 800
MIN_TEST_ROWS = 50
MAX_GOALS = 10
NEWCOMER = -1
NEWCOMER_MATCHES = 10
EPS = 1e-12

# Sulle righe importate da The Odds API (stagioni 2019-2024) il mercato
# "over 2.5" contiene in parte quote di ALTRE linee (bug di import corretto il
# 2026-09-14, vedi `df_odds_service.py`): le quote dei bookmaker sulla stessa
# partita divergono oltre il 15% in circa meta' dei casi. Il confronto col
# mercato sull'Over/Under usa quindi solo le righe API-Sports.
CLEAN_TOTALS_SOURCE = "sports-api"


# --------------------------------------------------------------------------
# Metriche
# --------------------------------------------------------------------------

def _clip_norm(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def multiclass_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """`probs` (n, 3) nell'ordine casa/pareggio/trasferta, `y` in {0, 1, 2}."""
    probs = _clip_norm(probs)
    n = len(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    log_loss = float(-np.mean(np.log(probs[np.arange(n), y])))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    # Ranked Probability Score: tiene conto dell'ordine casa < pareggio < trasferta.
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_y = np.cumsum(onehot, axis=1)[:, :2]
    rps = float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2.0))
    pred = probs.argmax(axis=1)
    accuracy = float(np.mean(pred == y))
    ece = expected_calibration_error(probs.max(axis=1), (pred == y).astype(float))
    home = np.clip(probs[:, 0], EPS, 1 - EPS)
    y_home = (y == 0).astype(float)
    return {
        "n": int(n),
        "log_loss": round(log_loss, 5),
        "brier": round(brier, 5),
        "rps": round(rps, 5),
        "accuracy": round(accuracy, 4),
        "ece_top_label": round(ece, 4),
        "home_win_binary_log_loss": round(float(-np.mean(y_home * np.log(home) + (1 - y_home) * np.log(1 - home))), 5),
        "home_win_binary_brier": round(float(np.mean((home - y_home) ** 2)), 5),
    }


def binary_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "accuracy": round(float(np.mean((p >= 0.5) == (y == 1))), 4),
        "ece": round(expected_calibration_error(p, y), 4),
        "positive_rate": round(float(y.mean()), 4),
        "mean_probability": round(float(p.mean()), 4),
    }


def expected_calibration_error(confidence: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    total = len(confidence)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.sum() / total * abs(confidence[mask].mean() - outcome[mask].mean())
    return float(ece)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(p[mask].mean()), 4),
                    "observed_rate": round(float(y[mask].mean()), 4),
                }
            )
    return rows


def _row_log_loss_multi(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    probs = _clip_norm(probs)
    return -np.log(probs[np.arange(len(y)), y])


def _row_log_loss_binary(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(delta: np.ndarray, resamples: int, rng: np.random.Generator) -> dict[str, float]:
    """Differenza media (candidato - riferimento) con IC 95% percentile."""
    n = len(delta)
    means = np.empty(resamples)
    for i in range(resamples):
        means[i] = delta[rng.integers(0, n, n)].mean()
    return {
        "mean_delta": round(float(delta.mean()), 5),
        "ci95_low": round(float(np.percentile(means, 2.5)), 5),
        "ci95_high": round(float(np.percentile(means, 97.5)), 5),
    }


# --------------------------------------------------------------------------
# Dati
# --------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date_utc"], utc=True)
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    frame = frame.sort_values(["date", "id_fixture"]).reset_index(drop=True)
    frame["day"] = frame["date"].dt.normalize()
    frame["y"] = np.where(frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2))
    frame["y_over25"] = (frame.home_goals + frame.away_goals >= 3).astype(int)
    frame["y_btts"] = ((frame.home_goals >= 1) & (frame.away_goals >= 1)).astype(int)
    frame["has_h2h_odds"] = frame["h2h_books"] > 0
    frame["quarter"] = frame["date"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    frame["month_start"] = frame["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    return frame


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Elo calcistico
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EloConfig:
    k: float
    home_advantage: float
    goal_margin: bool

    @property
    def name(self) -> str:
        return f"k{int(self.k)}_hfa{int(self.home_advantage)}_{'gd' if self.goal_margin else 'nogd'}"


ELO_GRID = [EloConfig(k, hfa, gd) for k in (8, 12, 15, 20, 30) for hfa in (40, 70) for gd in (True, False)]


def compute_elo(frame: pd.DataFrame, config: EloConfig) -> np.ndarray:
    """Differenza Elo pre-partita (casa - trasferta, vantaggio campo incluso).

    Le partite dello stesso giorno usano i rating di inizio giornata. Una
    squadra mai vista parte dalla media delle tre peggiori del suo campionato
    (una neopromossa non vale la media della lega), 1500 se il campionato non
    ha ancora rating."""
    ratings: dict[int, float] = {}
    league_members: dict[int, set[int]] = defaultdict(set)
    diff = np.zeros(len(frame))

    def initial(league: int) -> float:
        members = sorted(ratings[t] for t in league_members[league] if t in ratings)
        if len(members) >= 6:
            return float(np.mean(members[:3]))
        return 1500.0

    for _, day_rows in frame.groupby("day", sort=True):
        updates: list[tuple[int, int, float]] = []
        for row in day_rows.itertuples():
            for team in (row.home_team_id, row.away_team_id):
                if team not in ratings:
                    ratings[team] = initial(row.league)
                league_members[row.league].add(team)
            d = ratings[row.home_team_id] - ratings[row.away_team_id] + config.home_advantage
            diff[row.Index] = d
            expected = 1.0 / (1.0 + 10 ** (-d / 400.0))
            score = 1.0 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0.0
            margin = abs(row.home_goals - row.away_goals)
            mult = 1.0
            if config.goal_margin:
                mult = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11.0 + margin) / 8.0
            updates.append((row.home_team_id, row.away_team_id, config.k * mult * (score - expected)))
        for home, away, delta in updates:
            ratings[home] += delta
            ratings[away] -= delta
    return diff


# --------------------------------------------------------------------------
# Poisson indipendente e Dixon-Coles (forza squadre con decadimento temporale)
# --------------------------------------------------------------------------

def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam, dtype=float)
    tau = np.where((hg == 0) & (ag == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1.0 + lam * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1.0 + mu * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1.0 - rho, tau)
    return tau


def score_matrix_markets(lam: np.ndarray, mu: np.ndarray, rho: float) -> dict[str, np.ndarray]:
    """Probabilita' 1X2, Over 2.5 e Goal/Goal dalla matrice dei punteggi."""
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals[None, :], lam[:, None])
    pa = poisson.pmf(goals[None, :], mu[:, None])
    matrix = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        matrix[:, 0, 0] *= 1.0 - lam * mu * rho
        matrix[:, 0, 1] *= 1.0 + lam * rho
        matrix[:, 1, 0] *= 1.0 + mu * rho
        matrix[:, 1, 1] *= 1.0 - rho
    matrix = np.clip(matrix, 0.0, None)
    matrix /= matrix.sum(axis=(1, 2), keepdims=True)
    hg = goals[:, None]
    ag = goals[None, :]
    home = (matrix * (hg > ag)).sum(axis=(1, 2))
    draw = (matrix * (hg == ag)).sum(axis=(1, 2))
    away = (matrix * (hg < ag)).sum(axis=(1, 2))
    over25 = (matrix * (hg + ag >= 3)).sum(axis=(1, 2))
    btts = (matrix * ((hg >= 1) & (ag >= 1))).sum(axis=(1, 2))
    return {"1x2": np.column_stack([home, draw, away]), "over25": over25, "btts": btts}


def _effective_teams(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Identita' usata dal modello di forza: le prime `NEWCOMER_MATCHES`
    partite di una squadra che entra in un campionato DOPO la prima stagione
    disponibile contano per la pseudo-squadra `NEWCOMER` di quel campionato.
    Cosi' una neopromossa senza storia viene prevista con la forza media
    delle neopromosse passate, non con quella media della lega."""
    first_season = frame.groupby("league")["season"].min().to_dict()
    seen: dict[tuple[int, int], int] = defaultdict(int)
    entered_late: dict[tuple[int, int], bool] = {}
    home_eff = np.empty(len(frame), dtype=int)
    away_eff = np.empty(len(frame), dtype=int)
    for row in frame.itertuples():
        for side, team in (("h", row.home_team_id), ("a", row.away_team_id)):
            key = (row.league, team)
            if key not in entered_late:
                entered_late[key] = row.season > first_season[row.league]
            eff = NEWCOMER if (entered_late[key] and seen[key] < NEWCOMER_MATCHES) else team
            if side == "h":
                home_eff[row.Index] = eff
            else:
                away_eff[row.Index] = eff
        seen[(row.league, row.home_team_id)] += 1
        seen[(row.league, row.away_team_id)] += 1
    return home_eff, away_eff


def fit_team_strength(train: pd.DataFrame, as_of: pd.Timestamp, xi: float) -> Optional[dict[str, Any]]:
    """Poisson log-lineare: log(gol) = intercetta + casa + attacco(squadra) -
    difesa(avversaria), pesi exp(-xi * giorni). Poi `rho` di Dixon-Coles per
    massima verosimiglianza pesata a tassi fissati (stima in due passi)."""
    if len(train) < 200:
        return None
    teams = sorted(set(train.home_eff) | set(train.away_eff))
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    n = len(train)
    X = np.zeros((2 * n, 1 + 2 * n_teams))
    h_idx = train.home_eff.map(index).to_numpy()
    a_idx = train.away_eff.map(index).to_numpy()
    rows = np.arange(n)
    X[rows, 0] = 1.0
    X[rows, 1 + h_idx] = 1.0
    X[rows, 1 + n_teams + a_idx] = 1.0
    X[n + rows, 1 + a_idx] = 1.0
    X[n + rows, 1 + n_teams + h_idx] = 1.0
    y = np.concatenate([train.home_goals.to_numpy(), train.away_goals.to_numpy()])
    age_days = (as_of - train.date).dt.total_seconds().to_numpy() / 86400.0
    weights = np.exp(-xi * age_days)
    model = PoissonRegressor(alpha=1e-3, max_iter=300)
    model.fit(X, y, sample_weight=np.concatenate([weights, weights]))
    rates = model.predict(X)
    lam, mu = rates[:n], rates[n:]
    hg, ag = train.home_goals.to_numpy(), train.away_goals.to_numpy()

    def negative_ll(rho: float) -> float:
        tau = _tau(hg, ag, lam, mu, rho)
        if np.any(tau <= 0):
            return 1e12
        return -float(np.sum(weights * np.log(tau)))

    rho = float(minimize_scalar(negative_ll, bounds=(-0.3, 0.3), method="bounded").x)
    return {"model": model, "index": index, "n_teams": n_teams, "rho": rho}


def predict_team_strength(fit: dict[str, Any], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    index, n_teams = fit["index"], fit["n_teams"]
    fallback = index.get(NEWCOMER)
    n = len(rows)
    X = np.zeros((2 * n, 1 + 2 * n_teams))
    for i, (home, away) in enumerate(zip(rows.home_eff.to_numpy(), rows.away_eff.to_numpy())):
        h = index.get(home, fallback)
        a = index.get(away, fallback)
        X[i, 0] = 1.0
        if h is not None:
            X[i, 1 + h] = 1.0
            X[n + i, 1 + n_teams + h] = 1.0
        if a is not None:
            X[i, 1 + n_teams + a] = 1.0
            X[n + i, 1 + a] = 1.0
    rates = fit["model"].predict(X)
    return rates[:n], rates[n:]


def compute_team_strength_predictions(frame: pd.DataFrame, xi: float, start: pd.Timestamp) -> dict[str, np.ndarray]:
    """Tassi attesi pre-partita (lambda casa, mu trasferta) e `rho` per ogni
    partita da `start` in poi; rifit a inizio mese per campionato usando solo
    partite precedenti l'inizio del mese."""
    lam = np.full(len(frame), np.nan)
    mu = np.full(len(frame), np.nan)
    rho = np.full(len(frame), np.nan)
    for league, league_rows in frame.groupby("league"):
        for month_start, month_rows in league_rows[league_rows.date >= start].groupby("month_start"):
            train = league_rows[league_rows.date < month_start]
            fit = fit_team_strength(train, month_start, xi)
            if fit is None:
                continue
            l, m = predict_team_strength(fit, month_rows)
            lam[month_rows.index] = l
            mu[month_rows.index] = m
            rho[month_rows.index] = fit["rho"]
    return {"lam": lam, "mu": mu, "rho": rho}


# --------------------------------------------------------------------------
# Feature pre-partita per i modelli supervisionati
# --------------------------------------------------------------------------

def compute_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Medie mobili per squadra, calcolate PRIMA della partita."""
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    last_played: dict[int, pd.Timestamp] = {}
    columns = defaultdict(list)

    def summarise(team: int, when: pd.Timestamp, prefix: str) -> None:
        past = list(history[team])
        last5 = past[-5:]
        columns[f"{prefix}_matches"].append(len(past))
        columns[f"{prefix}_ppg5"].append(np.mean([m[0] for m in last5]) if last5 else np.nan)
        columns[f"{prefix}_gf10"].append(np.mean([m[1] for m in past]) if past else np.nan)
        columns[f"{prefix}_ga10"].append(np.mean([m[2] for m in past]) if past else np.nan)
        xg_for = [m[3] for m in past if m[3] == m[3]]
        xg_against = [m[4] for m in past if m[4] == m[4]]
        columns[f"{prefix}_xgf10"].append(np.mean(xg_for) if xg_for else np.nan)
        columns[f"{prefix}_xga10"].append(np.mean(xg_against) if xg_against else np.nan)
        rest = (when - last_played[team]).total_seconds() / 86400.0 if team in last_played else np.nan
        columns[f"{prefix}_rest_days"].append(min(rest, 21.0) if rest == rest else np.nan)

    for _, day_rows in frame.groupby("day", sort=True):
        for row in day_rows.itertuples():
            summarise(row.home_team_id, row.date, "home")
            summarise(row.away_team_id, row.date, "away")
        for row in day_rows.itertuples():
            home_points = 3 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 0
            away_points = 3 if row.home_goals < row.away_goals else 1 if row.home_goals == row.away_goals else 0
            history[row.home_team_id].append((home_points, row.home_goals, row.away_goals, row.home_xg, row.away_xg))
            history[row.away_team_id].append((away_points, row.away_goals, row.home_goals, row.away_xg, row.home_xg))
            last_played[row.home_team_id] = row.date
            last_played[row.away_team_id] = row.date
    return pd.DataFrame(columns, index=frame.index)


NO_ODDS_FEATURES = [
    "elo_diff", "dc_lam", "dc_mu",
    "home_ppg5", "away_ppg5", "home_gf10", "away_gf10", "home_ga10", "away_ga10",
    "home_xgf10", "away_xgf10", "home_xga10", "away_xga10",
    "home_rest_days", "away_rest_days", "home_matches", "away_matches",
]
ODDS_FEATURES = ["mkt_log_home", "mkt_log_draw", "mkt_log_away", "h2h_books", "h2h_overround"]


def _lr() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _hgb() -> Any:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, early_stopping=False, random_state=SEED
    )


def _rf() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=25, n_jobs=-1, random_state=SEED),
    )


SUPERVISED: dict[str, tuple[Callable[[], Any], list[str], bool]] = {
    # nome: (factory, feature, richiede quote in training)
    "elo_multinomial": (_lr, ["elo_diff"], False),
    "lr_no_odds": (_lr, NO_ODDS_FEATURES, False),
    "rf_no_odds": (_rf, NO_ODDS_FEATURES, False),
    "hgb_no_odds": (_hgb, NO_ODDS_FEATURES, False),
    "market_recalibrated": (_lr, ["mkt_log_home", "mkt_log_draw", "mkt_log_away"], True),
    "lr_with_odds": (_lr, NO_ODDS_FEATURES + ODDS_FEATURES, True),
    "rf_with_odds": (_rf, NO_ODDS_FEATURES + ODDS_FEATURES, True),
    "hgb_with_odds": (_hgb, NO_ODDS_FEATURES + ODDS_FEATURES, True),
}


def build_folds(frame: pd.DataFrame) -> list[dict[str, Any]]:
    folds = []
    test_pool = frame[frame.date >= TEST_START]
    for quarter, rows in test_pool.groupby("quarter", sort=True):
        if len(rows) < MIN_TEST_ROWS:
            continue
        folds.append({"fold": quarter, "start": rows.date.min().normalize(), "test_index": rows.index.to_numpy()})
    return folds


def run_supervised(
    frame: pd.DataFrame, folds: list[dict[str, Any]], target: str, n_classes: int
) -> dict[str, np.ndarray]:
    """Probabilita' fuori campione per ogni modello supervisionato."""
    out = {name: np.full((len(frame), n_classes), np.nan) for name in SUPERVISED}
    history = frame[frame.date >= SUPERVISED_HISTORY_START]
    for fold in folds:
        train_all = history[history.date < fold["start"]]
        test = frame.loc[fold["test_index"]]
        for name, (factory, features, needs_odds) in SUPERVISED.items():
            train = train_all[train_all.has_h2h_odds] if needs_odds else train_all
            rows = test[test.has_h2h_odds] if needs_odds else test
            if len(train) < MIN_TRAIN_ROWS or rows.empty or train[target].nunique() < n_classes:
                continue
            # Una colonna senza alcun valore nel training (es. xG prima del
            # 2022) non porta informazione: si esclude per quel fold.
            usable = [f for f in features if train[f].notna().any()]
            model = factory()
            model.fit(train[usable].to_numpy(dtype=float), train[target].to_numpy())
            out[name][rows.index] = model.predict_proba(rows[usable].to_numpy(dtype=float))
    return out


# --------------------------------------------------------------------------
# Selezione iperparametri (solo periodo pre-test)
# --------------------------------------------------------------------------

def select_elo(frame: pd.DataFrame) -> tuple[EloConfig, list[dict[str, Any]]]:
    window = (frame.date >= SELECTION_START) & (frame.date < TEST_START)
    y = frame.loc[window, "y"].to_numpy()
    results = []
    for config in ELO_GRID:
        diff = compute_elo(frame, config)
        x = diff[window.to_numpy()].reshape(-1, 1)
        # Mappa diff -> 1X2 stimata sulla finestra stessa: lieve ottimismo,
        # identico per tutte le configurazioni (un solo regressore).
        model = _lr().fit(x, y)
        results.append({"config": config.name, "log_loss": multiclass_metrics(model.predict_proba(x), y)["log_loss"]})
    best = min(zip(ELO_GRID, results), key=lambda pair: pair[1]["log_loss"])[0]
    return best, results


def select_decay(frame: pd.DataFrame, grid: tuple[float, ...]) -> tuple[float, list[dict[str, Any]]]:
    window = ((frame.date >= SELECTION_START) & (frame.date < TEST_START)).to_numpy()
    y = frame["y"].to_numpy()
    results = []
    for xi in grid:
        pred = compute_team_strength_predictions(frame[frame.date < TEST_START], xi, SELECTION_START)
        ok = window[: len(pred["lam"])] & ~np.isnan(pred["lam"])
        idx = np.flatnonzero(ok)
        markets = score_matrix_markets(pred["lam"][idx], pred["mu"][idx], 0.0)
        results.append({"xi": xi, "n": int(len(idx)), "log_loss_1x2": multiclass_metrics(markets["1x2"], y[idx])["log_loss"]})
    best = min(results, key=lambda r: r["log_loss_1x2"])["xi"]
    return best, results


# --------------------------------------------------------------------------
# Valutazione
# --------------------------------------------------------------------------

def evaluate_multiclass(
    frame: pd.DataFrame,
    contenders: dict[str, np.ndarray],
    mask: np.ndarray,
    reference: str,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Tutti i concorrenti sulle STESSE righe: `mask` e righe senza NaN."""
    valid = mask.copy()
    for probs in contenders.values():
        valid &= ~np.isnan(probs).any(axis=1)
    idx = np.flatnonzero(valid)
    y = frame["y"].to_numpy()[idx]
    sub = frame.iloc[idx]
    ref_loss = _row_log_loss_multi(contenders[reference][idx], y)
    report: dict[str, Any] = {"n": int(len(idx)), "reference": reference, "contenders": {}}
    for name, probs in contenders.items():
        p = probs[idx]
        entry: dict[str, Any] = {"overall": multiclass_metrics(p, y)}
        if name != reference:
            delta = _row_log_loss_multi(p, y) - ref_loss
            entry["vs_reference_log_loss"] = paired_bootstrap(delta, resamples, rng)
            per_fold = pd.Series(delta).groupby(sub["quarter"].to_numpy()).mean()
            entry["folds_better_than_reference"] = int((per_fold < 0).sum())
            entry["folds_total"] = int(len(per_fold))
        entry["by_league"] = {
            LEAGUES[league]: multiclass_metrics(p[(sub.league == league).to_numpy()], y[(sub.league == league).to_numpy()])
            for league in sorted(sub.league.unique())
        }
        entry["by_season"] = {
            str(season): multiclass_metrics(p[(sub.season == season).to_numpy()], y[(sub.season == season).to_numpy()])
            for season in sorted(sub.season.unique())
        }
        entry["calibration_home_win"] = calibration_table(_clip_norm(p)[:, 0], (y == 0).astype(float))
        entry["calibration_draw"] = calibration_table(_clip_norm(p)[:, 1], (y == 1).astype(float))
        report["contenders"][name] = entry
    return report


def evaluate_binary(
    frame: pd.DataFrame,
    target: str,
    contenders: dict[str, np.ndarray],
    mask: np.ndarray,
    reference: str,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    valid = mask.copy()
    for p in contenders.values():
        valid &= ~np.isnan(p)
    idx = np.flatnonzero(valid)
    y = frame[target].to_numpy()[idx].astype(float)
    sub = frame.iloc[idx]
    report: dict[str, Any] = {"n": int(len(idx)), "reference": reference, "contenders": {}}
    if len(idx) == 0:
        return report
    ref_loss = _row_log_loss_binary(contenders[reference][idx], y)
    for name, p_all in contenders.items():
        p = p_all[idx]
        entry: dict[str, Any] = {"overall": binary_metrics(p, y)}
        if name != reference:
            entry["vs_reference_log_loss"] = paired_bootstrap(_row_log_loss_binary(p, y) - ref_loss, resamples, rng)
        entry["by_league"] = {
            LEAGUES[league]: binary_metrics(p[(sub.league == league).to_numpy()], y[(sub.league == league).to_numpy()])
            for league in sorted(sub.league.unique())
        }
        entry["calibration"] = calibration_table(p, y)
        report["contenders"][name] = entry
    return report


def expanding_base_rate(frame: pd.DataFrame, folds: list[dict[str, Any]], target: str, n_classes: int) -> np.ndarray:
    """Frequenze osservate sul passato (per fold): il baseline piu' semplice."""
    out = np.full((len(frame), n_classes), np.nan)
    history = frame[frame.date >= SUPERVISED_HISTORY_START]
    for fold in folds:
        train = history[history.date < fold["start"]]
        freq = np.bincount(train[target].to_numpy(), minlength=n_classes) / max(len(train), 1)
        out[fold["test_index"]] = freq
    return out


def run_benchmark(dataset_path: str, resamples: int) -> dict[str, Any]:
    # I fit sono tanti e piccoli: un thread BLAS a testa e' piu' veloce di
    # molti thread che si contendono i core.
    with threadpool_limits(limits=1, user_api="blas"):
        return _run_benchmark(dataset_path, resamples)


def _run_benchmark(dataset_path: str, resamples: int) -> dict[str, Any]:
    started = time.time()
    rng = np.random.default_rng(SEED)
    frame = load_dataset(dataset_path)
    home_eff, away_eff = _effective_teams(frame)
    frame["home_eff"] = home_eff
    frame["away_eff"] = away_eff

    elo_config, elo_grid = select_elo(frame)
    decay_grid = (0.0, 0.001, 0.002, 0.003)
    xi, decay_results = select_decay(frame, decay_grid)

    frame["elo_diff"] = compute_elo(frame, elo_config)
    strength = compute_team_strength_predictions(frame, xi, SUPERVISED_HISTORY_START)
    frame["dc_lam"] = strength["lam"]
    frame["dc_mu"] = strength["mu"]
    frame = pd.concat([frame, compute_form_features(frame)], axis=1)
    for side in ("home", "draw", "away"):
        frame[f"mkt_log_{side}"] = np.log(frame[f"p_mkt_{side}"].clip(lower=1e-6))

    folds = build_folds(frame)
    test_mask = np.zeros(len(frame), dtype=bool)
    for fold in folds:
        test_mask[fold["test_index"]] = True

    has_strength = ~np.isnan(strength["lam"])
    idx = np.flatnonzero(has_strength)
    poisson_markets = score_matrix_markets(strength["lam"][idx], strength["mu"][idx], 0.0)
    dc_rho = strength["rho"][idx]
    # rho varia per (campionato, mese): si applica per gruppi di valore uguale.
    dc_markets = {"1x2": np.empty((len(idx), 3)), "over25": np.empty(len(idx)), "btts": np.empty(len(idx))}
    for value in np.unique(dc_rho):
        sel = dc_rho == value
        part = score_matrix_markets(strength["lam"][idx][sel], strength["mu"][idx][sel], float(value))
        for key in dc_markets:
            dc_markets[key][sel] = part[key]

    def spread(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        full = np.full(shape, np.nan)
        full[idx] = values
        return full

    n = len(frame)
    market_1x2 = frame[["p_mkt_home", "p_mkt_draw", "p_mkt_away"]].to_numpy(dtype=float)

    # ---- 1X2 ----
    supervised_1x2 = run_supervised(frame, folds, "y", 3)
    contenders_1x2 = {
        "market_no_vig": market_1x2,
        "base_rate": expanding_base_rate(frame, folds, "y", 3),
        "poisson_independent": spread(poisson_markets["1x2"], (n, 3)),
        "dixon_coles": spread(dc_markets["1x2"], (n, 3)),
        **supervised_1x2,
    }
    with_odds_mask = test_mask & frame.has_h2h_odds.to_numpy()
    report_1x2 = evaluate_multiclass(frame, contenders_1x2, with_odds_mask, "market_no_vig", resamples, rng)

    # ---- Over/Under 2.5 e Goal/No Goal: senza quote su tutte le partite di test ----
    def binary_block(target: str, key: str) -> dict[str, np.ndarray]:
        supervised = run_supervised(frame, folds, target, 2)
        return {
            "base_rate": expanding_base_rate(frame, folds, target, 2)[:, 1],
            "poisson_independent": spread(poisson_markets[key], (n,)),
            "dixon_coles": spread(dc_markets[key], (n,)),
            "lr_no_odds": supervised["lr_no_odds"][:, 1],
            "rf_no_odds": supervised["rf_no_odds"][:, 1],
            "hgb_no_odds": supervised["hgb_no_odds"][:, 1],
        }

    over_models = binary_block("y_over25", "over25")
    btts_models = binary_block("y_btts", "btts")
    report_over_all = evaluate_binary(frame, "y_over25", over_models, test_mask, "base_rate", resamples, rng)
    report_btts_all = evaluate_binary(frame, "y_btts", btts_models, test_mask, "base_rate", resamples, rng)

    # ---- Confronto col mercato dove le quote sono affidabili ----
    clean_totals = test_mask & (frame.ou25_books > 0).to_numpy() & (frame.odds_source == CLEAN_TOTALS_SOURCE).to_numpy()
    over_with_market = {"market_no_vig": frame["p_mkt_over25"].to_numpy(dtype=float), **over_models}
    report_over_market = evaluate_binary(frame, "y_over25", over_with_market, clean_totals, "market_no_vig", resamples, rng)

    btts_quoted = test_mask & (frame.btts_books > 0).to_numpy()
    btts_with_market = {"market_no_vig": frame["p_mkt_btts_yes"].to_numpy(dtype=float), **btts_models}
    report_btts_market = evaluate_binary(frame, "y_btts", btts_with_market, btts_quoted, "market_no_vig", resamples, rng)

    odds_rows = frame[frame.has_h2h_odds]
    totals_rows = frame[frame.ou25_books > 0]
    return {
        "benchmark": "top5_walk_forward_1x2_ou25_btts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "command": "python scripts/analysis/walk_forward_1x2_benchmark.py",
        "dataset": {
            "path": os.path.relpath(dataset_path, os.path.dirname(HERE)),
            "sha256": sha256_of(dataset_path),
            "matches": int(len(frame)),
            "date_min": frame.date.min().isoformat(),
            "date_max": frame.date.max().isoformat(),
            "matches_by_league": {LEAGUES[k]: int(v) for k, v in frame.league.value_counts().sort_index().items()},
            "matches_with_h2h_odds": int(frame.has_h2h_odds.sum()),
            "h2h_odds_date_min": odds_rows.date.min().isoformat(),
            "matches_per_season": {str(k): int(v) for k, v in frame.season.value_counts().sort_index().items()},
            "matches_by_league_and_season": {
                LEAGUES[league]: {str(k): int(v) for k, v in rows.season.value_counts().sort_index().items()}
                for league, rows in frame.groupby("league")
            },
            "known_gaps": "Ligue 1 stagione 2022 assente dal dump; l'ultima stagione arriva solo fino alla data del dump.",
            "odds_timing": "Uno snapshot per partita, istante di rilevazione non registrato: NON e' garantito che sia la quota di chiusura.",
            "totals_odds_quality": {
                "share_matches_with_bookmaker_dispersion_above_15pct_by_source": {
                    source: round(float((rows.ou25_dispersion > 1.15).mean()), 4)
                    for source, rows in totals_rows.groupby("odds_source")
                },
                "note": "Rapporto massimo/minimo fra le quote Over 2.5 dei bookmaker della stessa partita.",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "protocol": {
            "test_start": TEST_START.isoformat(),
            "folds": [{"fold": f["fold"], "test_rows": int(len(f["test_index"]))} for f in folds],
            "supervised_training": "finestra espandibile dal 2016-07-01; i modelli con quote si addestrano solo su righe quotate",
            "team_strength_refit": "inizio di ogni mese solare, per campionato, sulle sole partite precedenti",
            "newcomer_rule": f"prime {NEWCOMER_MATCHES} partite di una squadra entrata dopo la prima stagione -> pseudo-squadra comune",
            "market_probability": "media fra bookmaker delle probabilita' implicite normalizzate (solo bookmaker con tutti gli esiti)",
            "bootstrap_resamples": resamples,
            "min_train_rows": MIN_TRAIN_ROWS,
        },
        "hyperparameter_selection": {
            "window": f"{SELECTION_START.date()} .. {TEST_START.date()} (prima del test)",
            "elo": {"selected": elo_config.name, "grid": elo_grid},
            "time_decay_xi_per_day": {"selected": xi, "grid": decay_results},
            "dixon_coles_rho": {
                "mean": round(float(np.nanmean(strength["rho"])), 4),
                "min": round(float(np.nanmin(strength["rho"])), 4),
                "max": round(float(np.nanmax(strength["rho"])), 4),
            },
        },
        "results": {
            "1x2_on_quoted_test_matches": report_1x2,
            "over_2_5_all_test_matches": report_over_all,
            "over_2_5_vs_market_api_sports_odds_only": report_over_market,
            "btts_all_test_matches": report_btts_all,
            "btts_vs_market_two_sided_quotes_only": report_btts_market,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }


def ranking(report: dict[str, Any]) -> list[tuple[str, float]]:
    return sorted(((name, entry["overall"]["log_loss"]) for name, entry in report["contenders"].items()), key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    report = run_benchmark(args.dataset, args.bootstrap_resamples)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    for key, block in report["results"].items():
        print(f"\n== {key} (n={block['n']}, riferimento: {block['reference']}) ==")
        for name, log_loss in ranking(block):
            entry = block["contenders"][name]
            delta = entry.get("vs_reference_log_loss")
            extra = f"  delta={delta['mean_delta']:+.5f} [{delta['ci95_low']:+.5f}, {delta['ci95_high']:+.5f}]" if delta else ""
            print(f"  {name:24s} log_loss={log_loss:.5f} brier={entry['overall']['brier']:.5f}{extra}")
    print(f"\nReport: {args.output} ({report['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
