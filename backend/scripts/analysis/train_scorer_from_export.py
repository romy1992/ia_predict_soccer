"""Addestramento mercato 'giocatore segna' (anytime scorer) da dataset esportato.

Pensato per la sessione cloud (nessun accesso al DB): legge
scripts/analysis/_export/player_scoring.parquet. Confronta baseline,
random search (log loss) e voting/stacking; calibra e valuta per ruolo.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from src.data.player.scoring_dataset import FEATURE_COLUMNS, TARGET  # noqa: E402

SEED = 42
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def date_folds(dates: pd.Series, n=3):
    """Expanding window per data (mai spezza una partita: stessa data = stesso blocco)."""
    d = pd.to_datetime(dates).dt.normalize()
    cuts = [d.quantile(q) for q in np.linspace(0.4, 1, n + 1)]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        tr = np.where(d < lo)[0]
        te = np.where((d >= lo) & (d < hi if hi < d.max() else d <= hi))[0]
        out.append((tr, te))
    return out


def metrics(y, p):
    return {"logloss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p)}


def by_role(df, p):
    rows = {}
    for r in ("F", "M", "D"):
        m = (df[f"pos_{r}"] == 1).values
        rows[r] = {"n": int(m.sum()), "obs": round(df[TARGET][m].mean(), 4),
                   "pred": round(float(p[m].mean()), 4)}
    return rows


def calib_table(y, p, bins=8):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
    return g.round(4)


def main():
    df = pd.read_parquet(ROOT / "_export" / "player_scoring.parquet").sort_values("match_date")
    df = df.reset_index(drop=True)
    tr = df[df.season <= 2024]
    va = df[df.season == 2025]
    te = df[df.season == 2026]
    med = tr[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).median()  # mediane SOLO del train
    X = lambda d: d[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med).values  # noqa: E731
    Xtr, ytr, Xva, yva, Xte, yte = X(tr), tr[TARGET].values, X(va), va[TARGET].values, X(te), te[TARGET].values
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    res = {}

    # Baseline A: tasso per ruolo
    rate = {r: tr[TARGET][tr[f"pos_{r}"] == 1].mean() for r in "FMD"}
    pa = lambda d: np.select([d.pos_F == 1, d.pos_M == 1], [rate["F"], rate["M"]], rate["D"])  # noqa: E731
    res["A_role_rate"] = {"val": metrics(yva, pa(va)), "test": metrics(yte, pa(te))}

    # Baseline B: logistica piccola
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
    res["B_logistic"] = {"val": metrics(yva, lr.predict_proba(Xva)[:, 1]),
                         "test": metrics(yte, lr.predict_proba(Xte)[:, 1])}

    folds = date_folds(tr["match_date"])
    cands = {
        "logreg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                   {"logisticregression__C": np.logspace(-3, 1, 20)}),
        "hgb": (HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
                {"learning_rate": [0.02, 0.04, 0.08], "max_depth": [2, 3, 4, None],
                 "max_iter": [100, 200, 350], "min_samples_leaf": [50, 100, 300],
                 "l2_regularization": [0, 1, 10]}),
        "rf": (RandomForestClassifier(random_state=SEED, n_jobs=4),
               {"n_estimators": [200, 400], "max_depth": [4, 6, 8], "min_samples_leaf": [50, 100, 300],
                "max_features": ["sqrt", 0.5]}),
    }
    best = {}
    for name, (est, grid) in cands.items():
        s = RandomizedSearchCV(est, grid, n_iter=min(N_ITER, 20), cv=folds, scoring="neg_log_loss",
                               random_state=SEED, n_jobs=4, refit=True)
        s.fit(Xtr, ytr)
        best[name] = s.best_estimator_
        pv = s.predict_proba(Xva)[:, 1]
        res[f"search_{name}"] = {"cv_logloss": -s.best_score_, "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in s.best_params_.items()},
                                 "val": metrics(yva, pv), "test": metrics(yte, s.predict_proba(Xte)[:, 1])}
        print(name, res[f"search_{name}"]["val"], flush=True)

    ests = [(k, v) for k, v in best.items()]
    vot = VotingClassifier(ests, voting="soft").fit(Xtr, ytr)
    sta = StackingClassifier(ests, final_estimator=LogisticRegression(max_iter=2000), cv=folds,
                             stack_method="predict_proba").fit(Xtr, ytr) if False else None
    # stacking con cv temporale non e' una partizione: usiamo cv=3 sequenziale sicuro via TimeSeries-like no;
    # approssimiamo con meta-modello addestrato sulla validazione 2025 (vedi sotto).
    res["voting"] = {"val": metrics(yva, vot.predict_proba(Xva)[:, 1]),
                     "test": metrics(yte, vot.predict_proba(Xte)[:, 1])}
    base_va = np.column_stack([m.predict_proba(Xva)[:, 1] for _, m in ests])
    base_te = np.column_stack([m.predict_proba(Xte)[:, 1] for _, m in ests])
    # stacking: meta su meta-meta' della validazione (split a meta' per data)
    half = len(va) // 2
    meta = LogisticRegression(max_iter=2000).fit(np.log(base_va[:half] / (1 - base_va[:half])), yva[:half])
    lg = lambda a: np.log(np.clip(a, 1e-6, 1 - 1e-6) / (1 - np.clip(a, 1e-6, 1 - 1e-6)))  # noqa: E731
    meta = LogisticRegression(max_iter=2000).fit(lg(base_va[:half]), yva[:half])
    res["stacking_meta_on_val_firsthalf"] = {
        "val_secondhalf": metrics(yva[half:], meta.predict_proba(lg(base_va[half:]))[:, 1]),
        "test": metrics(yte, meta.predict_proba(lg(base_te))[:, 1])}

    # scelta campione per log loss su validazione tra search+voting
    pool = {k: v["val"]["logloss"] for k, v in res.items() if k.startswith("search_") or k == "voting"}
    champ = min(pool, key=pool.get)
    model = vot if champ == "voting" else best[champ.replace("search_", "")]
    pv_raw = model.predict_proba(Xva)[:, 1]
    pt_raw = model.predict_proba(Xte)[:, 1]
    # calibrazione isotonica addestrata sulla prima meta' della validazione, verificata sulla seconda
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(pv_raw[:half], yva[:half])
    res["champion"] = champ
    res["champion_raw_val_secondhalf"] = metrics(yva[half:], pv_raw[half:])
    res["champion_iso_val_secondhalf"] = metrics(yva[half:], iso.predict(pv_raw[half:]))
    res["champion_raw_test"] = metrics(yte, pt_raw)
    res["champion_iso_test"] = metrics(yte, iso.predict(pt_raw))
    res["by_role_test_raw"] = by_role(te, pt_raw)
    res["by_role_test_iso"] = by_role(te, iso.predict(pt_raw))
    print("\nCalibrazione test (raw):\n", calib_table(yte, pt_raw))
    # importanza semplice: HGB permutation-free -> coef logistica
    coefs = dict(zip(FEATURE_COLUMNS, lr[-1].coef_[0].round(3)))
    res["logistic_coefs"] = coefs
    out = ROOT / "_export" / "scorer_training_report.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in res if k not in ("logistic_coefs",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
