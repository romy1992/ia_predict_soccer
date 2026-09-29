"""Model Diagnostics (2026-09-12): metriche di classificazione dettagliate
(confusion matrix, precision/recall/F1 per classe, curva ROC+AUC) per
QUALUNQUE mercato BINARIO con un modello registrato - generalizza
`scripts/analysis/evaluate_champions_detailed.py` (limitato ai 4 Under/Over
e a un CSV di export che non esiste piu' nel repo) per leggere dal DB reale
e coprire ogni mercato di `FilterMarketService.SUPPORTED_MARKETS`.

Le metriche aggregate salvate in `ModelRegistry` (log_loss/brier/ece/auc/
selection_score) non includono queste - questa funzione le calcola via
walk-forward OOF (`temporal_oof_probabilities`, gia' in uso per
`train_multi_market.py`/`CalibrationService`): `clone()` del modello gia'
registrato (preserva l'intera architettura/iperparametri, incluso
l'eventuale wrapper di calibrazione, senza rifare la grid search) rifittato
fold per fold sullo stesso walk-forward espandente usato in produzione -
MAI valutato sui dati di training, stessa garanzia anti-leakage.

Esclude deliberatamente il mercato "1x2" (multiclasse HOME/DRAW/AWAY,
`src/ml/markets/market_1x2.py`) e i mercati specializzati a linea
configurabile (`corners_line_*`/`cards_line_*`, MARKET-05/06): la shape
qui (confusion matrix 2x2, ROC binaria) non si applica a un problema a 3
classi, e le linee configurabili non passano da
`FilterMarketService.build_dataset`. Entrambi fuori scope per questa prima
versione - da estendere con la propria logica quando/se avranno bisogno di
una vista diagnostica dedicata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

from src.ml.evaluation.probability_metrics import temporal_oof_probabilities
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_paths import resolve_model_path
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import _build_temporal_cv, _filter_valid_splits

_META_COLUMNS = ["y", "market", "id_fixture", "season", "league", "prediction_at"]


@dataclass
class MarketDiagnostics:
    """Esito per UN mercato. `status != "ok"` -> tutti i campi numerici
    restano `None` (mai un valore inventato), il chiamante (endpoint/CLI)
    decide come comunicarlo (es. omesso dalla lista principale, elencato a
    parte con il motivo)."""

    market: str
    status: str  # "ok" | "no_model" | "insufficient_data" | "single_class_oof"
    n_oof: Optional[int] = None
    champion: Optional[str] = None
    stage: Optional[str] = None
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    cm: Optional[dict[str, int]] = None
    class0: Optional[dict[str, float]] = None
    class1: Optional[dict[str, float]] = None
    weighted: Optional[dict[str, float]] = None
    roc_fpr: list[float] = field(default_factory=list)
    roc_tpr: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_diagnosable_markets(registry: Optional[ModelRegistry] = None) -> list[str]:
    """Mercati con ALMENO un modello registrato E costruibili via
    `FilterMarketService.build_dataset` - esclude quindi automaticamente
    "1x2" e i mercati a linea configurabile (nessuno dei due e' mai un
    membro di `SUPPORTED_MARKETS`), senza bisogno di una blocklist
    esplicita."""
    registry = registry or ModelRegistry()
    registered = set(registry.list_markets())
    return sorted(registered & FilterMarketService.SUPPORTED_MARKETS)


def evaluate_market_diagnostics(
    market: str,
    seasons: Optional[list[int]] = None,
    registry: Optional[ModelRegistry] = None,
) -> MarketDiagnostics:
    """Calcola le metriche OOF per `market`. Non solleva mai per un mercato
    senza modello/dati sufficienti: restituisce uno `status` esplicito
    invece (stesso principio "mai un'eccezione che interrompe l'intero
    giro" gia' seguito per i job schedulati di questo progetto)."""
    registry = registry or ModelRegistry()
    model_meta = registry.get_production(market=market) or registry.get_latest(market=market)
    if not model_meta:
        return MarketDiagnostics(market=market, status="no_model")

    # Percorso RISOLTO: dopo la riorganizzazione di `best_models/` il file
    # puo' stare in `under_over/<mercato>/` o in `archivio/`, e una riga di
    # registry rimasta indietro farebbe sparire il mercato dalla diagnostica
    # come se non avesse un modello.
    model_path = resolve_model_path(model_meta.get("model_path"))
    if not model_path:
        return MarketDiagnostics(market=market, status="no_model")

    df = FilterMarketService().build_dataset(market=market, seasons=seasons)
    if df.empty:
        return MarketDiagnostics(market=market, status="insufficient_data")

    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
    if df.empty:
        return MarketDiagnostics(market=market, status="insufficient_data")

    y = df["y"].astype(int)
    X = df.drop(columns=_META_COLUMNS, errors="ignore")

    raw_splits = _build_temporal_cv(df)
    if raw_splits is None:
        return MarketDiagnostics(market=market, status="insufficient_data")
    cv_splits = _filter_valid_splits(y=y, splits=raw_splits)
    if len(cv_splits) < 2:
        return MarketDiagnostics(market=market, status="insufficient_data")

    champion = joblib.load(model_path)
    oof = temporal_oof_probabilities(estimator=champion, X=X, y=y, cv_splits=cv_splits)
    if oof.empty or oof["y_true"].nunique() < 2:
        return MarketDiagnostics(market=market, status="single_class_oof")

    y_true = oof["y_true"].to_numpy()
    p1 = oof["probability"].to_numpy()
    y_pred = (p1 >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = float((y_true == y_pred).mean())
    fpr, tpr, _ = roc_curve(y_true, p1)
    auc = float(roc_auc_score(y_true, p1))

    return MarketDiagnostics(
        market=market,
        status="ok",
        n_oof=int(len(oof)),
        champion=type(champion).__name__,
        stage=model_meta.get("current_stage") or model_meta.get("stage"),
        accuracy=accuracy,
        auc=auc,
        cm={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        class0={
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        class1={
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        weighted={"precision": float(precision_w), "recall": float(recall_w), "f1": float(f1_w)},
        roc_fpr=fpr.tolist(),
        roc_tpr=tpr.tolist(),
    )


def evaluate_markets_diagnostics(
    markets: Optional[list[str]] = None,
    seasons: Optional[list[int]] = None,
    registry: Optional[ModelRegistry] = None,
) -> list[MarketDiagnostics]:
    """Variante multi-mercato. `markets=None` -> tutti quelli diagnosticabili
    (`list_diagnosable_markets`). Un fallimento isolato su un mercato non
    blocca gli altri (stesso principio "provider errors isolati" gia'
    applicato in LIVE-01/al job di snapshot predizioni)."""
    registry = registry or ModelRegistry()
    target_markets = markets if markets is not None else list_diagnosable_markets(registry=registry)

    results: list[MarketDiagnostics] = []
    for market in target_markets:
        try:
            results.append(evaluate_market_diagnostics(market=market, seasons=seasons, registry=registry))
        except Exception as exc:
            results.append(MarketDiagnostics(market=market, status=f"error: {exc}"))
    return results
