"""Direct Market Expert wrapper (EXP-05).

Standardizza l'interfaccia dei modelli "diretti" per mercato (h2h, dc,
goal_no_goal, corners, cards, under_over_*) gia' addestrati da
`train_multi_market.py`: training temporale (`expanding_window_splits`) e
calibrazione (`CalibrationService`) sono gia' presenti li' (ML-04/ML-06) e
NON vengono re-implementati qui. Questo modulo aggiunge solo:
- un'interfaccia comune `predict_proba` sopra al champion model salvato,
- metadati espliciti sul tipo di classificazione di ciascun mercato.

IMPORTANTE (acceptance criteria EXP-05): il mercato "h2h" qui e' e resta un
modello BINARIO (P(vittoria interna) vs resto). NON va MAI interpretato o
esposto come se fosse un 1X2 multiclasse (Home/Draw/Away): draw e away sono
aggregati nella classe negativa. Il vero 1X2 multiclasse e' pianificato nel
task MARKET-01, volutamente non anticipato qui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np

from src.service_ia.training.model_registry import ModelRegistry

# Specifica esplicita del tipo di classificazione per ciascun mercato "diretto"
# attualmente addestrato da train_multi_market.py. Impedisce che un modello
# binario venga scambiato/esposto come multiclasse (o viceversa).
DIRECT_MARKET_SPECS: dict[str, dict[str, Any]] = {
    "h2h": {
        "classification_type": "binary",
        "positive_label": "home_win",
        "outcome_semantics": (
            "P(vittoria squadra di casa) vs resto (pareggio+trasferta aggregati). "
            "NON e' un 1X2 multiclasse: vedi task MARKET-01 per la versione a 3 classi."
        ),
    },
    "dc": {
        "classification_type": "binary",
        "positive_label": "home_or_draw",
        "outcome_semantics": "P(1X, non-vittoria trasferta) vs P(2)",
    },
    "goal_no_goal": {
        "classification_type": "binary",
        "positive_label": "btts_yes",
        "outcome_semantics": "P(entrambe le squadre segnano) vs P(almeno una non segna)",
    },
    "under_over_1_5": {
        "classification_type": "binary",
        "positive_label": "over_1_5",
        "outcome_semantics": "P(Over 1.5) vs P(Under 1.5)",
    },
    "under_over_2_5": {
        "classification_type": "binary",
        "positive_label": "over_2_5",
        "outcome_semantics": "P(Over 2.5) vs P(Under 2.5)",
    },
    "under_over_3_5": {
        "classification_type": "binary",
        "positive_label": "over_3_5",
        "outcome_semantics": "P(Over 3.5) vs P(Under 3.5)",
    },
    "under_over_4_5": {
        "classification_type": "binary",
        "positive_label": "over_4_5",
        "outcome_semantics": "P(Over 4.5) vs P(Under 4.5)",
    },
    "corners": {
        "classification_type": "binary",
        "positive_label": "over_corners_threshold",
        "outcome_semantics": "P(corner totali >= soglia) vs resto",
    },
    "cards": {
        "classification_type": "binary",
        "positive_label": "over_cards_threshold",
        "outcome_semantics": "P(cartellini totali >= soglia) vs resto",
    },
}


def _class1_probability(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr[:, -1]


@dataclass
class DirectMarketExpert:
    """Interfaccia comune (`predict_proba`) sopra un champion model gia' addestrato."""

    market: str
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        if self.market not in DIRECT_MARKET_SPECS:
            raise ValueError(f"Mercato non riconosciuto come direct expert: {self.market}")
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("L'estimator caricato non espone predict_proba: interfaccia comune non rispettata")

    @property
    def classification_type(self) -> str:
        return DIRECT_MARKET_SPECS[self.market]["classification_type"]

    @property
    def positive_label(self) -> str:
        return DIRECT_MARKET_SPECS[self.market]["positive_label"]

    @property
    def outcome_semantics(self) -> str:
        return DIRECT_MARKET_SPECS[self.market]["outcome_semantics"]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Interfaccia comune: ritorna sempre P(classe positiva) come array 1D."""
        ordered = X[self.feature_names] if self.feature_names else X
        raw = self.estimator.predict_proba(ordered)
        return _class1_probability(raw)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @classmethod
    def load_production(cls, market: str, registry: Optional[Any] = None) -> "DirectMarketExpert":
        """Carica lo stage 'production' (vincolo generale: latest != production)."""
        registry = registry or ModelRegistry()
        run = registry.get_production(market=market)
        if run is None:
            raise LookupError(f"Nessun modello in stage 'production' per il mercato '{market}'")
        return cls._from_run(run)

    @classmethod
    def load_latest(cls, market: str, registry: Optional[Any] = None) -> "DirectMarketExpert":
        """ATTENZIONE: solo per debug/validazione, mai per servire predizioni reali."""
        registry = registry or ModelRegistry()
        run = registry.get_latest(market=market)
        if run is None:
            raise LookupError(f"Nessun modello registrato per il mercato '{market}'")
        return cls._from_run(run)

    @classmethod
    def _from_run(cls, run: dict[str, Any]) -> "DirectMarketExpert":
        model_path = run.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path non trovato per il run {run.get('run_id')}: {model_path}")

        estimator = joblib.load(model_path)
        return cls(
            market=run.get("market"),
            estimator=estimator,
            feature_names=list(run.get("feature_names") or []),
            run_id=run.get("run_id"),
            stage=run.get("current_stage") or run.get("stage"),
        )

    @classmethod
    def from_estimator(cls, market: str, estimator: Any, feature_names: Optional[list[str]] = None) -> "DirectMarketExpert":
        """Costruzione diretta (utile nei test, senza toccare registry/disco)."""
        return cls(market=market, estimator=estimator, feature_names=list(feature_names or []))
