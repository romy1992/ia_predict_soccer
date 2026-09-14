"""Passo 6: precisione/volume, ROI, e la diagnostica classica di un mercato.

Su un mercato l'accuracy da sola non dice niente: con base rate 77% (Under/Over
1.5) dire sempre Over ne fa gia' 77. Quello che conta e' come si muove la
precisione quando si alza la soglia, e soprattutto se il guadagno regge quando
lo si paga alla quota reale - perche' alzando la soglia si scelgono partite
piu' probabili, quindi peggio quotate, e il vantaggio si annulla da solo.

Il modello qui e' il candidato SCELTO (non necessariamente il champion del
gate): su Under/Over 2.5 `logistic` batte `voting` su AUC, log-loss, Brier ed
ECE ma perde il punteggio composito per 0,0001, e una regressione logistica e'
piu' semplice, piu' stabile e leggibile di un ensemble.

Le probabilita' sono OUT-OF-FOLD sulla stessa CV walk-forward del training
(mai in-sample) e passano per la stessa calibrazione della pipeline
(`CalibrationService`), cosi' "0,70" significa davvero 70%.

Il candidato si sceglie da riga di comando; senza indicarlo si usa il
champion del report di training di quel mercato. Sbagliarlo non da' errore,
da' numeri di un altro modello: su Under/Over 1.5 il champion e'
`random_forest`, e misurare li' la logistica descrive un modello che non e'
mai stato promosso.

Uso:
    python scripts/analysis/passo6_soglie_e_roi.py under_over_2_5 2_5 logistic
    python scripts/analysis/passo6_soglie_e_roi.py under_over_1_5 1_5
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.rifacimento_mercato import quote_per_linea, righe_sospette  # noqa: E402
from src.ml.calibration.calibration_service import CalibrationService  # noqa: E402
from src.service_ia.pre_processing.feature_selection import FeatureSelectionService  # noqa: E402
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

EXPORT = os.path.join("scripts", "analysis", "_export")
SEED = 42

GRANDEZZE = [
    "shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal",
    "shots_outsidebox", "blocked_shots", "yellow_cards", "red_cards", "fouls",
]


def costruisci(nome: str, params: dict, n_feature: int) -> Pipeline:
    """Ricostruisce la pipeline del candidato con i parametri vinti dalla grid.

    Deve restare allineata a `_model_space` in train_multi_market.py: stessi
    passaggi, stesso seed, stesso class_weight. La logistica ha in piu' lo
    scaler, la random forest no.
    """
    selettore = FeatureSelectionService.build_selector("kbest", n_feature)
    try:
        selettore.set_params(k=params.get("selector__k", 10))
    except ValueError:
        pass

    if nome == "logistic":
        modello = LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED,
            C=params.get("model__C", 0.1), solver=params.get("model__solver", "lbfgs"),
        )
        passaggi = [("scaler", StandardScaler()), ("model", modello)]
    elif nome == "random_forest":
        modello = RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced",
            max_depth=params.get("model__max_depth"),
            min_samples_split=params.get("model__min_samples_split", 2),
            min_samples_leaf=params.get("model__min_samples_leaf", 1),
        )
        passaggi = [("model", modello)]
    else:
        raise ValueError(f"Candidato non ricostruibile qui: {nome}")

    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                           ("selector", selettore)] + passaggi)


def prepara(mercato: str, linea: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(EXPORT, f"{mercato}_raw.csv"))
    quote = quote_per_linea(linea)
    statistiche = [
        f"mean_{g}_{lato}_stat" for g in GRANDEZZE for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]
    feature = quote + statistiche
    df = df[df[quote].notna().all(axis=1)].copy()
    df = df[~righe_sospette(df, linea)].copy()
    df = df[df[statistiche].notna().all(axis=1)].copy()
    return df.sort_values("prediction_at").reset_index(drop=True), feature


def roi(y: np.ndarray, scelte: np.ndarray, quote: np.ndarray) -> float:
    """Puntando 1 su ogni partita scelta: si incassa la quota se esce, 0 se no."""
    if scelte.sum() == 0:
        return float("nan")
    incasso = np.where(y[scelte] == 1, quote[scelte], 0.0).sum()
    return (incasso - scelte.sum()) / scelte.sum() * 100


def intervallo_roi(y: np.ndarray, scelte: np.ndarray, quote: np.ndarray, n: int = 2000) -> tuple[float, float]:
    idx = np.flatnonzero(scelte)
    if len(idx) < 30:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    campioni = []
    for _ in range(n):
        presi = rng.choice(idx, size=len(idx), replace=True)
        incasso = np.where(y[presi] == 1, quote[presi], 0.0).sum()
        campioni.append((incasso - len(presi)) / len(presi) * 100)
    return float(np.percentile(campioni, 2.5)), float(np.percentile(campioni, 97.5))


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python scripts/analysis/passo6_soglie_e_roi.py <mercato> <linea> [candidato]")
        return 1
    mercato, linea = sys.argv[1], sys.argv[2]

    percorso_report = os.path.join(EXPORT, f"train_{mercato}_selected.json")
    if not os.path.exists(percorso_report):
        print(f"Manca il report di training: {percorso_report}")
        return 1
    report = json.load(open(percorso_report))
    candidato = sys.argv[3] if len(sys.argv) > 3 else report["champion"]
    params = report["details"]["models"][candidato]["best_params"]

    df, feature = prepara(mercato, linea)
    X, y_serie = df[feature], df["y"].astype(int)
    splits = _filter_valid_splits(y_serie, _build_temporal_cv(df) or [])

    modello = costruisci(candidato, params, n_feature=len(feature))

    print(f"mercato   : {mercato}   linea: {linea}   candidato: {candidato}"
          f"{'  (champion)' if candidato == report['champion'] else '  (NON e il champion)'}")
    print(f"righe     : {len(df):,}   feature in ingresso: {len(feature)}   usate dal selettore: {params.get('selector__k')}")
    print(f"parametri : {params}")
    print(f"base rate : {y_serie.mean():.4f}   fold: {len(splits)}\n")

    ris = CalibrationService.calibrate_estimator(estimator=modello, X=X, y=y_serie, cv_splits=splits)
    p = np.asarray(ris.post_probabilities, dtype=float)
    y = np.asarray(ris.post_y_true, dtype=int)

    # Le righe OOF escono nell'ordine in cui i fold le producono: ricostruisco
    # gli indici concatenando i valid_idx nello stesso ordine, cosi' posso
    # riagganciare la quota di ciascuna partita per il ROI. Se le lunghezze non
    # coincidono mi fermo invece di allineare a caso quote ed esiti.
    indici = np.concatenate([np.asarray(va) for tr, va in splits if len(tr) and len(va)])
    if len(indici) != len(p):
        print(f"Disallineamento: {len(indici)} indici contro {len(p)} probabilita'. Mi fermo.")
        return 1
    print(f"calibrazione: {ris.method}   ECE {ris.pre_metrics.get('ece'):.4f} -> {ris.post_metrics.get('ece'):.4f}")
    q_over = df[f"odds_mean_over_{linea}"].to_numpy()[indici]
    q_max_over = df[f"odds_max_over_{linea}"].to_numpy()[indici]
    q_under = df[f"odds_mean_under_{linea}"].to_numpy()[indici]
    q_max_under = df[f"odds_max_under_{linea}"].to_numpy()[indici]

    print("=" * 78)
    print("QUALITA' DELLA PROBABILITA' (out-of-fold, calibrata)")
    print("=" * 78)
    print(f"  AUC {roc_auc_score(y,p):.4f}   log-loss {log_loss(y,p):.4f}   "
          f"Brier {brier_score_loss(y,p):.4f}   n {len(y):,}")

    print("\n" + "=" * 78)
    print("DIAGNOSTICA A SOGLIA 0.50")
    print("=" * 78)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    print(f"\n  accuracy {accuracy_score(y,pred):.4f}   (dire sempre la classe piu' frequente: "
          f"{max(y.mean(), 1-y.mean()):.4f})")
    print("\n  matrice di confusione")
    print("                  previsto Under   previsto Over")
    print(f"     reale Under  {cm[0,0]:14,} {cm[0,1]:15,}")
    print(f"     reale Over   {cm[1,0]:14,} {cm[1,1]:15,}")
    print("\n" + classification_report(y, pred, target_names=["Under", "Over"], digits=4))

    for direzione, positivo in (("OVER", True), ("UNDER", False)):
        prezzo_medio = q_over if positivo else q_under
        prezzo_max = q_max_over if positivo else q_max_under
        base = y.mean() if positivo else 1 - y.mean()
        # Le soglie alte servono sui mercati sbilanciati: su Under/Over 1.5 il
        # base rate e' 76%, quindi sotto 0,80 non si sta scegliendo niente.
        soglie = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90) if positivo else (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20)
        esito = y if positivo else 1 - y

        print("=" * 78)
        print(f"PUNTANDO {direzione}   (dire sempre {direzione.title()}: precisione {base:.1%})")
        print("=" * 78)
        print(f"  {'soglia':>7} {'precis.':>9} {'partite':>9} {'%tot':>6} {'q.media':>8} "
              f"{'ROI med':>9} {'ROI max':>9} {'IC 95% sul ROI max':>24}")
        print("  " + "-" * 92)
        for s in soglie:
            scelte = (p >= s) if positivo else (p <= s)
            n = int(scelte.sum())
            if n < 30:
                print(f"  {s:7.2f} {'-':>9} {n:9} {'poche':>6}")
                continue
            prec = esito[scelte].mean()
            lo, hi = intervallo_roi(esito, scelte, prezzo_max)
            print(f"  {s:7.2f} {prec:8.1%} {n:9,} {n/len(y):5.1%} {prezzo_medio[scelte].mean():8.3f} "
                  f"{roi(esito,scelte,prezzo_medio):8.1f}% {roi(esito,scelte,prezzo_max):8.1f}% "
                  f"{f'[{lo:+.2f}%, {hi:+.2f}%]':>24}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
