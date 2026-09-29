"""Coerenza fra le linee: P(Over 1.5) non puo' essere sotto P(Over 2.5).

Una partita che supera i 2,5 gol ha necessariamente superato gli 1,5. Quindi
la probabilita' di Over 1.5 deve stare SEMPRE sopra quella di Over 2.5, e lo
stesso vale scendendo lungo tutte le linee. I due modelli pero' sono
addestrati separatamente, su dataset diversi e con feature diverse: niente
nella pipeline li obbliga a rispettarla.

Promemoria esplicito dell'operatore: "la situazione monotona di cui mi hai
sempre parlato dove un under 1.5 non potra' mai essere un over 2.5".

Le probabilita' sono OUT-OF-FOLD e calibrate, le stesse del passo 6, e i due
mercati vengono confrontati solo sulle partite che hanno in comune.

Uso:
    python scripts/analysis/coerenza_linee.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.passo6_soglie_e_roi import costruisci, prepara  # noqa: E402
from src.ml.calibration.calibration_service import CalibrationService  # noqa: E402
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

EXPORT = os.path.join("scripts", "analysis", "_export")


def probabilita_oof(mercato: str, linea: str, candidato: str | None) -> pd.DataFrame:
    report = json.load(open(os.path.join(EXPORT, f"train_{mercato}_selected.json")))
    nome = candidato or report["champion"]
    params = report["details"]["models"][nome]["best_params"]

    df, feature = prepara(mercato, linea)
    X, y = df[feature], df["y"].astype(int)
    splits = _filter_valid_splits(y, _build_temporal_cv(df) or [])
    modello = costruisci(nome, params, n_feature=len(feature))

    ris = CalibrationService.calibrate_estimator(estimator=modello, X=X, y=y, cv_splits=splits)
    indici = np.concatenate([np.asarray(va) for tr, va in splits if len(tr) and len(va)])
    p = np.asarray(ris.post_probabilities, dtype=float)
    if len(indici) != len(p):
        raise RuntimeError(f"{mercato}: {len(indici)} indici contro {len(p)} probabilita'")

    fuori = df.iloc[indici][["id_fixture", "prediction_at"]].copy()
    fuori[f"p_{linea}"] = p
    fuori[f"y_{linea}"] = np.asarray(ris.post_y_true, dtype=int)
    print(f"  {mercato:16} modello {nome:16} {len(fuori):6,} partite out-of-fold")
    return fuori.dropna(subset=["id_fixture"])


def main() -> int:
    print("Calcolo delle probabilita' out-of-fold calibrate:")
    uno = probabilita_oof("under_over_1_5", "1_5", None)          # champion: random_forest
    due = probabilita_oof("under_over_2_5", "2_5", "logistic")    # scelta dell'operatore

    unito = uno.merge(due, on="id_fixture", how="inner", suffixes=("", "_2"))
    print(f"\npartite in comune fra i due mercati: {len(unito):,}")
    if unito.empty:
        print("Nessuna partita in comune: confronto impossibile.")
        return 1

    p15, p25 = unito["p_1_5"].to_numpy(), unito["p_2_5"].to_numpy()
    violazione = p25 > p15
    scarto = p25 - p15

    print("\n" + "=" * 74)
    print("VIOLAZIONI  (P(Over 2.5) sopra P(Over 1.5): impossibile)")
    print("=" * 74)
    print(f"  partite in violazione : {violazione.sum():,}  ({violazione.mean():.2%})")
    if violazione.any():
        print(f"  scarto medio          : {scarto[violazione].mean():+.4f}")
        print(f"  scarto massimo        : {scarto[violazione].max():+.4f}")
        print("\n  distribuzione della gravita':")
        for lo, hi, lab in [(0, 0.01, "trascurabile (<0,01)"), (0.01, 0.05, "piccola (0,01-0,05)"),
                            (0.05, 0.15, "seria (0,05-0,15)"), (0.15, 9, "grave (>0,15)")]:
            m = violazione & (scarto >= lo) & (scarto < hi)
            print(f"     {lab:24} {m.sum():6,}  ({m.mean():6.2%} del totale)")

    print("\n" + "=" * 74)
    print("CONTROPROVA SUI FATTI  (le due linee sono coerenti nella realta'?)")
    print("=" * 74)
    incoerenti = (unito["y_2_5"] == 1) & (unito["y_1_5"] == 0)
    print(f"  partite Over 2.5 ma NON Over 1.5 : {incoerenti.sum()}  (devono essere ZERO)")
    if incoerenti.any():
        print("  ATTENZIONE: i target stessi sono incoerenti, il problema non e' nei modelli.")

    print("\n" + "=" * 74)
    print("EFFETTO SULLE DECISIONI")
    print("=" * 74)
    print("  Una violazione conta solo se cambia cosa si gioca. Quante volte il")
    print("  modello consiglierebbe Over 2.5 mentre sconsiglia Over 1.5?")
    for s15, s25 in ((0.50, 0.65), (0.65, 0.65), (0.75, 0.65)):
        assurde = (p25 >= s25) & (p15 < s15)
        print(f"     Over 2.5 sopra {s25:.2f} ma Over 1.5 sotto {s15:.2f}: {assurde.sum():5,}  ({assurde.mean():.2%})")

    print("\n  correlazione fra le due probabilita':", round(float(np.corrcoef(p15, p25)[0, 1]), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
