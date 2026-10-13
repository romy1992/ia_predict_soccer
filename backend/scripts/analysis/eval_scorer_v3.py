"""Valutazione per gruppo (cinque campionati / coppe europee / altri campionati) del modello marcatori v3.

Config A: addestra solo sui cinque campionati (come in produzione oggi).
Config B: addestra su tutto.
Config C: come B piu' i flag di competizione (coppa europea, altro campionato).
Holdout temporale unico (ultimo 20% delle date); metriche separate per gruppo e, per le coppe, per ruolo."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from src.data.player.scoring_dataset import FEATURE_COLUMNS, TARGET  # noqa: E402
from src.ml.evaluation.probability_metrics import compute_probability_metrics  # noqa: E402

TOP5, CUPS = {135, 39, 140, 78, 61}, {2, 3, 848}


def group(league):
    return "top5" if league in TOP5 else "cup" if league in CUPS else "other"


def matrix(df, med, flags):
    x = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med)
    if flags:
        x = x.assign(is_cup=(df["grp"] == "cup").astype(int), is_other=(df["grp"] == "other").astype(int))
    return x.to_numpy(dtype=float)


def metrics(y, p, base):
    m = compute_probability_metrics(y, p)
    m.pop("reliability", None)
    m["baseline_log_loss"] = float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base)))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def main():
    df = pd.read_parquet(ROOT / "_export" / "player_scoring_v3.parquet").sort_values("match_date").reset_index(drop=True)
    df["grp"] = df["league"].map(group)
    cut = df["match_date"].quantile(0.8)
    train_all, test = df[df.match_date < cut], df[df.match_date >= cut]
    print(f"holdout da {cut} | train {len(train_all)} | test {len(test)}", flush=True)
    print(test.groupby("grp").agg(n=(TARGET, "size"), rate=(TARGET, "mean")).round(4).to_string(), flush=True)
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.0113, max_iter=3000)),
        "hgb": lambda: HistGradientBoostingClassifier(learning_rate=0.08, max_depth=2, max_iter=100,
                                                      min_samples_leaf=50, l2_regularization=1,
                                                      early_stopping=False, random_state=42),
    }
    configs = {"A_top5_only": (train_all[train_all.grp == "top5"], False),
               "B_all": (train_all, False), "C_all_flags": (train_all, True)}
    out = {}
    for cname, (tr, flags) in configs.items():
        med = tr[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).median()
        for mname, make in models.items():
            model = make().fit(matrix(tr, med, flags), tr[TARGET].to_numpy())
            p = model.predict_proba(matrix(test, med, flags))[:, 1]
            res = {}
            for g in ("top5", "cup", "other"):
                mask = (test.grp == g).to_numpy()
                base = float(tr[tr.grp == g][TARGET].mean()) if (tr.grp == g).any() else float(tr[TARGET].mean())
                res[g] = metrics(test[TARGET].to_numpy()[mask], p[mask], base)
                if g == "cup":
                    res[g]["by_role"] = {
                        r: {"n": int(m.sum()), "obs": round(float(test[TARGET][mask][m[mask]].mean()), 4),
                            "pred": round(float(p[mask][m[mask]].mean()), 4)}
                        for r in "FMD" if (m := (test[f"pos_{r}"] == 1).to_numpy()).any()}
            out[f"{cname}/{mname}"] = res
            print(cname, mname, {g: (res[g]["log_loss"], res[g]["auc"], res[g]["ece"]) for g in res}, flush=True)
    (ROOT / "_export" / "scorer_v3_eval.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
