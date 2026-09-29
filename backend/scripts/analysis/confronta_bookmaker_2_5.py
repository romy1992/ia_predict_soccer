"""Under/Over 2.5: quanti bookmaker tenere nella media delle quote?

Dopo la riscrittura delle quote sulla linea corretta e' emerso che lo script
di riscrittura non applicava il filtro `if title in names_book` che la
pipeline originale aveva sul percorso `totals`: quel filtro teneva solo
1xBet, Pinnacle, Unibet e William Hill, mentre ora sono 19.

Qui si misura invece di decidere a giudizio. Tre varianti, stesso modello,
stesso split, stesso seed - cambia solo quali quote entrano nella media:

  A  19 bookmaker            stato attuale del DB
  B   4 bookmaker            la allowlist originale, rimessa sul percorso totals
  C  19 bookmaker + filtro   come A, ma senza le quote rilevate DOPO il
                             calcio d'inizio (45 quote su 52.436, 28 fixture):
                             sono prezzi live, informazione che al momento
                             della previsione non esisteva

Nota sull'asimmetria: il filtro ai 4 esisteva SOLO su `totals`. Le chiavi
`alternate_*` sono sempre state scritte per tutti i bookmaker
(`df_odds_service.py:726`, nessun controllo su names_book). La variante B
non e' quindi "il passato", e' solo il percorso totals ristretto.

Le quote si ricostruiscono in locale senza toccare il DB: il backup
`_backup_under_over_2_5.jsonl` contiene il bucket PRIMA della riscrittura
(quindi anche le chiavi `alternate_*`, che la riscrittura non ha toccato) e
il payload grezzo contiene le quote con il loro `point`.

Uso:
    python scripts/analysis/confronta_bookmaker_2_5.py
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.market_service.filter_market_service import FilterMarketService  # noqa: E402
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits  # noqa: E402

EXPORT = os.path.join("scripts", "analysis", "_export")
PAYLOAD = os.path.join("src", "service_ia", "dataset", "odds", "id_odds_h2h_totals.csv")
BACKUP = os.path.join("scripts", "maintenance", "_backup_under_over_2_5.jsonl")
MAPPA = os.path.join(EXPORT, "mappa_id_events.csv")
DATASET = os.path.join(EXPORT, "under_over_2_5_raw.csv")

ALLOWLIST = {"1xBet", "Pinnacle", "Unibet", "William Hill"}
LINEA = 2.5
SEED = 42
PREFISSI_TOTALS = ("over_2.5_", "under_2.5_")

QUOTE = [
    "prob_norm_over_2_5",
    "odds_mean_over_2_5",
    "odds_mean_under_2_5",
    "odds_count",
    "odds_std_over_2_5",
    "overround",
]
GRANDEZZE = [
    "shots_on_goal", "total_shots", "shots_insidebox", "shots_off_goal",
    "shots_outsidebox", "blocked_shots", "yellow_cards", "red_cards", "fouls",
    "corner_kicks", "offsides", "expected_goals", "ball_possession",
]


def leggi_payload() -> dict[str, list[dict]]:
    """{id_evento: [{book, over, under, minuti_dal_fischio}, ...]} sulla linea 2.5."""
    csv.field_size_limit(10**9)
    fuori: dict[str, list[dict]] = {}
    with open(PAYLOAD, newline="", encoding="utf-8") as handle:
        for riga in csv.DictReader(handle):
            try:
                books = ast.literal_eval(riga["bookmakers"])
                inizio = dt.datetime.fromisoformat(riga["commence_time"].replace("Z", "+00:00"))
            except (ValueError, SyntaxError, KeyError):
                continue
            voci = []
            for book in books:
                titolo = book.get("title")
                if not titolo:
                    continue
                minuti = None
                if book.get("last_update"):
                    try:
                        aggiornata = dt.datetime.fromisoformat(book["last_update"].replace("Z", "+00:00"))
                        minuti = (aggiornata - inizio).total_seconds() / 60
                    except ValueError:
                        pass
                for mercato in book.get("markets", []):
                    if mercato.get("key") != "totals":
                        continue
                    esiti = mercato.get("outcomes") or []
                    over = next((e for e in esiti if e.get("name") == "Over" and e.get("point") == LINEA), None)
                    under = next((e for e in esiti if e.get("name") == "Under" and e.get("point") == LINEA), None)
                    if over or under:
                        voci.append(
                            {
                                "book": titolo,
                                "over": (over or {}).get("price"),
                                "under": (under or {}).get("price"),
                                "minuti": minuti,
                            }
                        )
            if voci:
                fuori[riga["id"]] = voci
    return fuori


def bucket_variante(base: dict, voci: list[dict], solo_allowlist: bool, scarta_live: bool) -> dict:
    """Rimonta il bucket: le chiavi non-`totals` restano, le totals si riscrivono."""
    nuovo = {k: v for k, v in (base or {}).items() if not k.startswith(PREFISSI_TOTALS)}
    for voce in voci:
        if solo_allowlist and voce["book"] not in ALLOWLIST:
            continue
        if scarta_live and voce["minuti"] is not None and voce["minuti"] > 0:
            continue
        if voce["over"]:
            nuovo[f"over_2.5_{voce['book']}"] = voce["over"]
        if voce["under"]:
            nuovo[f"under_2.5_{voce['book']}"] = voce["under"]
    return nuovo


def feature_quote(bucket: dict) -> dict:
    f = dict(FilterMarketService._extract_market_odds_features(bucket) or {})
    f.update(FilterMarketService._extract_per_outcome_odds_features(bucket) or {})
    return f


def valuta(frame: pd.DataFrame, colonne: list[str]) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    frame = frame.sort_values("prediction_at").reset_index(drop=True)
    splits = _filter_valid_splits(frame["y"], _build_temporal_cv(frame) or [])
    y_all, p_all = [], []
    for tr, va in splits:
        modello = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1)),
            ]
        )
        modello.fit(frame.loc[tr, colonne], frame.loc[tr, "y"])
        p_all.extend(modello.predict_proba(frame.loc[va, colonne])[:, 1])
        y_all.extend(frame.loc[va, "y"])
    y, p = np.array(y_all), np.array(p_all)
    return roc_auc_score(y, p), log_loss(y, p), brier_score_loss(y, p), y, p


def main() -> int:
    for percorso in (PAYLOAD, BACKUP, MAPPA, DATASET):
        if not os.path.exists(percorso):
            print(f"Manca: {percorso}")
            return 1

    print("lettura del payload grezzo...")
    payload = leggi_payload()
    print(f"  eventi con quote 2.5: {len(payload):,}")

    print("lettura del backup (bucket prima della riscrittura)...")
    base_per_evento: dict[str, dict] = {}
    with open(BACKUP, encoding="utf-8") as handle:
        for riga in handle:
            r = json.loads(riga)
            base_per_evento[r["id_events"]] = r["prima"]
    print(f"  righe di backup: {len(base_per_evento):,}")

    mappa = pd.read_csv(MAPPA, dtype={"id_events": str, "id_alternate_events": str})
    mappa = mappa.dropna(subset=["id_fixture"])
    ev_to_fix = dict(zip(mappa["id_events"], mappa["id_fixture"].astype(int)))
    for alt, fix in zip(mappa["id_alternate_events"], mappa["id_fixture"].astype(int)):
        if isinstance(alt, str) and alt:
            ev_to_fix.setdefault(alt, fix)
    print(f"  mappa id_events -> id_fixture: {len(ev_to_fix):,}")

    df = pd.read_csv(DATASET)
    statistiche = [
        f"mean_{g}_{lato}_stat" for g in GRANDEZZE for lato in ("home", "away", "diff")
        if f"mean_{g}_{lato}_stat" in df.columns
    ]
    df = df.dropna(subset=["id_fixture"]).copy()
    df["id_fixture"] = df["id_fixture"].astype(int)
    df = df[df[statistiche].notna().all(axis=1)].copy()
    print(f"\ndataset di partenza: {len(df):,} righe con id_fixture e statistiche complete")

    varianti = {
        "A - 19 bookmaker (stato attuale)": dict(solo_allowlist=False, scarta_live=False),
        "B -  4 bookmaker (allowlist)": dict(solo_allowlist=True, scarta_live=False),
        "C - 19 bookmaker + no live": dict(solo_allowlist=False, scarta_live=True),
    }

    risultati = {}
    for nome, opzioni in varianti.items():
        righe = []
        for id_evento, voci in payload.items():
            fixture = ev_to_fix.get(id_evento)
            if fixture is None:
                continue
            bucket = bucket_variante(base_per_evento.get(id_evento, {}), voci, **opzioni)
            if not bucket:
                continue
            f = feature_quote(bucket)
            if not all(c in f for c in QUOTE):
                continue
            f["id_fixture"] = fixture
            righe.append(f)
        quote_df = pd.DataFrame(righe).drop_duplicates(subset="id_fixture", keep="last")
        unito = df.drop(columns=[c for c in df.columns if c in quote_df.columns and c != "id_fixture"]).merge(
            quote_df, on="id_fixture", how="inner"
        )
        unito = unito[unito[QUOTE].notna().all(axis=1)].copy()
        risultati[nome] = unito
        print(f"  {nome:34} {len(unito):6,} righe   odds_count medio {unito['odds_count'].mean():5.2f}")

    comuni = set.intersection(*(set(u["id_fixture"]) for u in risultati.values()))
    print(f"\nfixture presenti in TUTTE le varianti: {len(comuni):,}")
    print("Il confronto gira su quelle, cosi' l'unica variabile sono le quote.\n")

    print("=" * 76)
    print(f"{'variante':34} {'righe':>7} {'book':>6} {'AUC':>8} {'logloss':>9} {'brier':>8}")
    print("=" * 76)
    curve = {}
    for nome, unito in risultati.items():
        u = unito[unito["id_fixture"].isin(comuni)].copy()
        auc, ll, brier, y, p = valuta(u, QUOTE + statistiche)
        curve[nome] = (y, p)
        print(f"{nome:34} {len(u):7,} {u['odds_count'].mean():6.2f} {auc:8.4f} {ll:9.4f} {brier:8.4f}")

    print("\n" + "=" * 76)
    print("PRECISIONE / VOLUME sulla classe Over")
    print("=" * 76)
    for nome, (y, p) in curve.items():
        base = y.mean()
        print(f"\n{nome}   (dire sempre Over: {base:.1%})")
        print(f"   {'soglia':>7} {'precisione':>11} {'partite':>9} {'% tot':>7}")
        for soglia in (0.55, 0.60, 0.65, 0.70):
            scelte = p >= soglia
            n = int(scelte.sum())
            if n < 30:
                print(f"   {soglia:7.2f} {'-':>11} {n:9} {'poche':>7}")
                continue
            print(f"   {soglia:7.2f} {y[scelte].mean():10.1%} {n:9,} {n/len(y):6.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
