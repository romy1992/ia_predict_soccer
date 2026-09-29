"""Prototipo (2026-09-21): modelli generativi sui gol vs classificatori binari.

NON e' codice di produzione: nessun modello qui e' registrato o promosso.
E' l'artefatto dell'indagine chiesta dall'operatore ("si possono creare
algoritmi custom con formule specifiche che estendano il base estimator?
i soliti random forest o logistic non sono idonei al 100%?"). I risultati
completi stanno in `report_prototipo_dixon_coles.md`.

## L'idea

I modelli attuali trattano ogni mercato come una domanda si'/no indipendente
("ci saranno piu' di 2.5 gol?") e addestrano un classificatore per ciascuna.
Un modello generativo stima invece il PROCESSO - quanti gol segna la squadra
di casa e quanti la trasferta - e da quella distribuzione legge ogni mercato
come una somma di celle della matrice dei punteggi:

    Over 2.5    = somma celle con casa+ospite >= 3
    Goal/Goal   = somma celle con casa >= 1 e ospite >= 1
    1 (casa)    = somma celle con casa > ospite

Un fit solo, tutti i mercati, coerenti per costruzione.

## Le due classi

- `PoissonTotalsClassifier`: Poisson sul TOTALE gol. Risponde a qualunque
  soglia Over/Under senza rifittare. Sklearn-compatibile (`BaseEstimator`/
  `ClassifierMixin`), quindi utilizzabile nella pipeline del progetto -
  con l'avvertenza che `fit(X, y)` vuole y = CONTEGGIO, non il binario.
- `DixonColesModel`: due Poisson separate (casa/trasferta) piu' la
  correzione Dixon-Coles (1997) sui quattro punteggi bassi (0-0, 1-0,
  0-1, 1-1), dove il Poisson indipendente sbaglia sistematicamente. Il
  parametro `rho` e' stimato dai dati per massima verosimiglianza: sui
  dati di questo progetto risulta stabilmente negativo (-0.05 / -0.12 su
  ogni fold di ogni test fatto), coerente con la letteratura.

`feature_dixon_coles()` estrae lambda e probabilita' derivate per l'uso
IBRIDO: darle in pasto al random forest come feature aggiuntive e' la
configurazione che ha dato i risultati migliori (vedi report).

Uso:
    python scripts/analysis/prototipo_dixon_coles.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

SEED = 42
MAX_GOL = 10
EXPORT = os.path.join("scripts", "analysis", "_export")
LINEE_OVER = (1.5, 2.5, 3.5)
GRANDEZZE = ["shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal",
             "shots_outsidebox", "blocked_shots", "yellow_cards", "red_cards", "fouls"]
DC_FEATURES = ["dc_lambda_casa", "dc_lambda_trasferta", "dc_lambda_totale", "dc_lambda_diff",
               "dc_p_over_1_5", "dc_p_over_2_5", "dc_p_over_3_5", "dc_p_gng", "dc_p_casa"]


def _glm_poisson(alpha: float = 1e-6, max_iter: int = 1000) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("glm", PoissonRegressor(alpha=alpha, max_iter=max_iter))])


class PoissonTotalsClassifier(BaseEstimator, ClassifierMixin):
    """Over/Under a soglia, con un Poisson GLM sul totale gol sotto.

    `fit(X, y)` vuole y = CONTEGGIO reale dei gol, non il target binario:
    e' il punto dell'approccio - il modello impara la distribuzione, non
    la singola domanda si'/no. `predict_proba` risponde per la soglia
    configurata; `predict_proba_linea` per QUALUNQUE altra soglia senza
    rifittare nulla (ed e' garantita monotona fra soglie, per costruzione).
    """

    def __init__(self, line: float = 2.5, alpha: float = 1e-6, max_iter: int = 1000):
        self.line = line
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        conteggi = np.asarray(y, dtype=float)
        if np.any(conteggi < 0):
            raise ValueError("Il target di un Poisson non puo' essere negativo")
        self.classes_ = np.array([0, 1])
        self._glm = _glm_poisson(self.alpha, self.max_iter).fit(X, conteggi)
        return self

    def lambda_(self, X):
        """Il numero atteso di eventi: l'unico parametro stimato."""
        return np.clip(self._glm.predict(X), 1e-6, None)

    def predict_proba_linea(self, X, line: float):
        p_over = 1.0 - poisson.cdf(np.floor(line), self.lambda_(X))
        return np.column_stack([1.0 - p_over, p_over])

    def predict_proba(self, X):
        return self.predict_proba_linea(X, self.line)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tau(i: int, j: int, lam_casa: float, lam_trasferta: float, rho: float) -> float:
    """Correzione Dixon-Coles: agisce SOLO sui quattro punteggi bassi."""
    if i == 0 and j == 0:
        return 1.0 - lam_casa * lam_trasferta * rho
    if i == 0 and j == 1:
        return 1.0 + lam_casa * rho
    if i == 1 and j == 0:
        return 1.0 + lam_trasferta * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Due Poisson (casa/trasferta) + correzione sui punteggi bassi.

    Un fit -> `matrice_punteggi()` -> ogni mercato e' una somma di celle.
    """

    def fit(self, X, gol_casa, gol_trasferta, stima_rho: bool = True):
        self.glm_home_ = _glm_poisson().fit(X, np.asarray(gol_casa, dtype=float))
        self.glm_away_ = _glm_poisson().fit(X, np.asarray(gol_trasferta, dtype=float))
        self.rho_ = 0.0
        if stima_rho:
            self.rho_ = self._stima_rho(X, np.asarray(gol_casa, dtype=int),
                                        np.asarray(gol_trasferta, dtype=int))
        return self

    def _stima_rho(self, X, h, a) -> float:
        lh = np.clip(self.glm_home_.predict(X), 1e-6, None)
        la = np.clip(self.glm_away_.predict(X), 1e-6, None)
        basso = (h <= 1) & (a <= 1)  # altrove tau == 1, non informa su rho
        if not basso.any():
            return 0.0
        hh, aa, lhh, laa = h[basso], a[basso], lh[basso], la[basso]

        def neg_loglik(rho: float) -> float:
            if abs(rho) >= 0.99:
                return 1e9
            taus = np.array([_tau(i, j, x, y, rho) for i, j, x, y in zip(hh, aa, lhh, laa)])
            if np.any(taus <= 0):
                return 1e9
            return -np.log(taus).sum()

        ris = minimize_scalar(neg_loglik, bounds=(-0.2, 0.2), method="bounded")
        return float(ris.x) if ris.success else 0.0

    def matrice_punteggi(self, X) -> np.ndarray:
        """(n_partite, MAX_GOL+1, MAX_GOL+1): P(casa=i, trasferta=j)."""
        lh = np.clip(self.glm_home_.predict(X), 1e-6, None)
        la = np.clip(self.glm_away_.predict(X), 1e-6, None)
        gol = np.arange(MAX_GOL + 1)
        M = (poisson.pmf(gol[None, :], lh[:, None])[:, :, None]
             * poisson.pmf(gol[None, :], la[:, None])[:, None, :])
        if self.rho_:
            for i in (0, 1):
                for j in (0, 1):
                    M[:, i, j] *= np.array([_tau(i, j, x, y, self.rho_) for x, y in zip(lh, la)])
        M = np.clip(M, 1e-12, None)
        return M / M.sum(axis=(1, 2), keepdims=True)

    def p_over(self, X, linea: float):
        i, j = np.indices((MAX_GOL + 1, MAX_GOL + 1))
        return self.matrice_punteggi(X)[:, (i + j) > linea].sum(axis=1)

    def p_goal_goal(self, X):
        return self.matrice_punteggi(X)[:, 1:, 1:].sum(axis=(1, 2))

    def p_home_win(self, X):
        i, j = np.indices((MAX_GOL + 1, MAX_GOL + 1))
        return self.matrice_punteggi(X)[:, i > j].sum(axis=1)


def feature_dixon_coles(dc: DixonColesModel, X) -> pd.DataFrame:
    """Lambda + probabilita' derivate, da usare come feature aggiuntive."""
    lam_c = np.clip(dc.glm_home_.predict(X), 1e-6, None)
    lam_t = np.clip(dc.glm_away_.predict(X), 1e-6, None)
    out = {"dc_lambda_casa": lam_c, "dc_lambda_trasferta": lam_t,
           "dc_lambda_totale": lam_c + lam_t, "dc_lambda_diff": lam_c - lam_t,
           "dc_p_gng": dc.p_goal_goal(X), "dc_p_casa": dc.p_home_win(X)}
    for linea in LINEE_OVER:
        out[f"dc_p_over_{str(linea).replace('.', '_')}"] = dc.p_over(X, linea)
    return pd.DataFrame(out, index=X.index)[DC_FEATURES]


# --------------------------------------------------------------------------
# Confronto riproducibile
# --------------------------------------------------------------------------

def _calibra(y, p):
    meta = len(p) // 2
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p[:meta], y[:meta])
    return y[meta:], np.clip(iso.predict(p[meta:]), 1e-6, 1 - 1e-6)


def _metriche(y, p):
    y_c, p_c = _calibra(y, p)
    return roc_auc_score(y, p), log_loss(y_c, p_c, labels=[0, 1]), brier_score_loss(y_c, p_c)


def carica_dati() -> tuple[pd.DataFrame, list[str]]:
    """Join dei frame gia' esportati: quote 1X2 + Under/Over + statistiche
    + i gol reali casa/trasferta (necessari per il modello generativo)."""
    gol = pd.read_csv(os.path.join(EXPORT, "gol_casa_trasferta.csv"))
    gol = gol.dropna(subset=["id_fixture", "gol_home_stat", "gol_away_stat"])
    gol = gol[["id_fixture", "gol_home_stat", "gol_away_stat"]].drop_duplicates("id_fixture")

    base = pd.read_csv(os.path.join(EXPORT, "totals_evaluation_frame.csv"))
    stat = [f"mean_{g}_{l}_stat" for g in GRANDEZZE for l in ("home", "away", "diff")
            if f"mean_{g}_{l}_stat" in base.columns]
    ou = [c for c in ["prob_norm_over_2_5", "odds_mean_over_2_5", "odds_mean_under_2_5",
                      "odds_std_over_2_5"] if c in base.columns]
    base = base[["id_fixture", "prediction_at"] + ou + stat + ["odds_count", "overround"]].rename(
        columns={"odds_count": "odds_count_ou", "overround": "overround_ou"})

    h2h = pd.read_csv(os.path.join(EXPORT, "h2h_raw.csv"))
    h2h_q = [c for c in ["prob_norm_home", "prob_norm_draw", "prob_norm_away",
                         "odds_mean_home", "odds_mean_draw", "odds_mean_away",
                         "odds_std_home", "odds_count", "overround"] if c in h2h.columns]
    h2h = h2h[["id_fixture"] + h2h_q].drop_duplicates("id_fixture").rename(
        columns={"odds_count": "odds_count_h2h", "overround": "overround_h2h"})
    h2h_feat = [("odds_count_h2h" if c == "odds_count" else
                 "overround_h2h" if c == "overround" else c) for c in h2h_q]

    df = base.merge(h2h, on="id_fixture", how="inner").merge(gol, on="id_fixture", how="inner")
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    feature = ou + ["odds_count_ou", "overround_ou"] + h2h_feat + stat
    df = df.dropna(subset=["prediction_at"] + feature).sort_values("prediction_at").reset_index(drop=True)
    return df, feature


def main() -> int:
    df, FEAT = carica_dati()
    gh, ga = df["gol_home_stat"].astype(int), df["gol_away_stat"].astype(int)
    tot = gh + ga
    mercati = {f"over_{str(l).replace('.', '_')}": (tot > l).astype(int) for l in LINEE_OVER}
    mercati["goal_no_goal"] = ((gh >= 1) & (ga >= 1)).astype(int)
    mercati["h2h_casa"] = (gh > ga).astype(int)

    print(f"righe: {len(df)}   feature (identiche per ogni modello): {len(FEAT)}")
    print(f"media gol casa {gh.mean():.3f}  trasferta {ga.mean():.3f}")
    print(f"totale gol: media {tot.mean():.3f} varianza {tot.var():.3f} "
          f"(rapporto {tot.var()/tot.mean():.3f}; ~1 = Poisson)\n")

    modelli = ("dixon_coles", "logistic", "rf", "rf_ibrido")
    oof = {m: {k: [] for k in modelli + ("y",)} for m in mercati}
    rhos = []

    for tr, te in TimeSeriesSplit(n_splits=5).split(df):
        X_tr, X_te = df.iloc[tr][FEAT], df.iloc[te][FEAT]
        gh_tr, ga_tr = gh.iloc[tr], ga.iloc[tr]

        dc = DixonColesModel().fit(X_tr, gh_tr, ga_tr)
        rhos.append(dc.rho_)

        # Feature DC per il train: out-of-fold interno (mai in-sample, o
        # l'ibrido risulterebbe falsamente ottimista).
        dc_tr = pd.DataFrame(index=X_tr.index, columns=DC_FEATURES, dtype=float)
        for i_tr, i_va in KFold(n_splits=3, shuffle=False).split(X_tr):
            interno = DixonColesModel().fit(X_tr.iloc[i_tr], gh_tr.iloc[i_tr], ga_tr.iloc[i_tr],
                                            stima_rho=False)
            dc_tr.iloc[i_va] = feature_dixon_coles(interno, X_tr.iloc[i_va]).values
        X_tr_ib = pd.concat([X_tr, dc_tr], axis=1)
        X_te_ib = pd.concat([X_te, feature_dixon_coles(dc, X_te)], axis=1)

        letture = {f"over_{str(l).replace('.', '_')}": dc.p_over(X_te, l) for l in LINEE_OVER}
        letture["goal_no_goal"] = dc.p_goal_goal(X_te)
        letture["h2h_casa"] = dc.p_home_win(X_te)

        for mercato, y_all in mercati.items():
            y_tr = y_all.iloc[tr]
            oof[mercato]["y"].append(y_all.iloc[te].to_numpy())
            oof[mercato]["dixon_coles"].append(letture[mercato])

            log = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=1000, random_state=SEED))
            log.fit(X_tr, y_tr)
            oof[mercato]["logistic"].append(log.predict_proba(X_te)[:, 1])

            for nome, Xa, Xb in (("rf", X_tr, X_te), ("rf_ibrido", X_tr_ib, X_te_ib)):
                rf = make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                                          random_state=SEED, n_jobs=1))
                rf.fit(Xa, y_tr)
                oof[mercato][nome].append(rf.predict_proba(Xb)[:, 1])

    print(f"rho Dixon-Coles per fold: {[round(r, 4) for r in rhos]}\n")
    print(f"{'mercato':<14} {'modello':<14} {'AUC':>8} {'logloss_cal':>12} {'brier_cal':>10}")
    print("-" * 64)
    prob = {}
    for mercato in mercati:
        y = np.concatenate(oof[mercato]["y"])
        righe = []
        for nome in modelli:
            p = np.concatenate(oof[mercato][nome])
            prob.setdefault(nome, {})[mercato] = p
            righe.append((nome,) + _metriche(y, p))
        best = max(righe, key=lambda r: r[1])[0]
        for nome, auc, ll, br in righe:
            print(f"{mercato:<14} {nome:<14} {auc:8.4f} {ll:12.4f} {br:10.4f}"
                  f"{'  <-- best' if nome == best else ''}")
        print("-" * 64)

    chiavi = [f"over_{str(l).replace('.', '_')}" for l in LINEE_OVER]
    n = len(prob["dixon_coles"][chiavi[0]])
    print(f"\nVIOLAZIONI DI MONOTONIA sulla scala Over (su {n} partite):")
    for nome in modelli:
        v = sum(int((prob[nome][a] < prob[nome][b] - 1e-9).sum())
                for a, b in zip(chiavi, chiavi[1:]))
        print(f"  {nome:<14} {v:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
