"""Adapter da output NATIVO di ciascun Oracle Expert verso lo schema comune
`ExpertOutput` (ORACLE-01).

Design: ZERO modifiche agli esperti gia' esistenti ed ai loro test
(EXP-01..05, MARKET-05/06). Ogni funzione qui si limita a leggere l'output
GIA' prodotto da un esperto (dict / riga DataFrame / np.ndarray) e a
confezionarlo nello schema comune, cosi' da preservare al 100% compatibilita'
e test gia' esistenti (nessuna riscrittura degli esperti).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.ml.ensemble.expert_output import ExpertOutput


def _line_label(line: Any) -> str:
    return str(float(line)).replace(".", "_")


def from_team_strength(
    rating_row: Mapping[str, Any],
    *,
    expert_version: str = "",
    team_role: str = "home_team",
) -> ExpertOutput:
    """Team Strength Expert (EXP-01).

    Non produce probabilita' dirette di esito (espone SOLO rating/feature
    riusabili da altri esperti, es. `GoalDistributionExpert`): il
    `probability_vector` resta VUOTO per costruzione, il payload va
    consumato via `raw_output`/`metadata`.

    `rating_row` e' una riga (dict o `pd.Series`) prodotta da
    `TeamStrengthExpert.build_ratings_dataset()` (o un item di
    `current_ratings()`); `team_role` documenta quale prefisso
    ('home_team'/'away_team'/'team') e' stato usato per generarla.
    """
    payload = dict(rating_row)
    timestamp = payload.get("prediction_at")
    version = expert_version or str(payload.get("rating_version") or "")

    return ExpertOutput(
        expert_name="team_strength",
        expert_version=version,
        probability_vector={},
        feature_timestamp=str(timestamp) if timestamp is not None else None,
        metadata={"team_role": team_role, **{k: v for k, v in payload.items() if k != "prediction_at"}},
        raw_output=payload,
    )


def from_goal_distribution(
    expert_output: Mapping[str, Any],
    *,
    feature_timestamp: Optional[str] = None,
    use_score_matrix_probabilities: bool = False,
) -> ExpertOutput:
    """Goal Distribution Expert (EXP-02): converte il dict restituito da
    `GoalDistributionExpert.build_expert_output(home_lambda, away_lambda)`.

    Il `probability_vector` usa di default le probabilita' Over/Under dalla
    Poisson sulla lambda combinata (chiave 'over_under'); se
    `use_score_matrix_probabilities=True` usa invece quelle ricavate dalla
    convoluzione della score matrix ('over_under_from_score_matrix').
    L'esperto stesso non conosce il timestamp delle feature usate per
    stimare le lambda: va passato esplicitamente dal chiamante.
    """
    key = "over_under_from_score_matrix" if use_score_matrix_probabilities else "over_under"
    probability_vector = dict(expert_output.get(key) or {})

    metadata = {
        "home_lambda": expert_output.get("home_lambda"),
        "away_lambda": expert_output.get("away_lambda"),
        "total_lambda": expert_output.get("total_lambda"),
        "over_under_from_score_matrix": expert_output.get("over_under_from_score_matrix"),
        "score_matrix": expert_output.get("score_matrix"),
    }

    return ExpertOutput(
        expert_name="goal_distribution",
        expert_version=str(expert_output.get("expert_version") or ""),
        probability_vector=probability_vector,
        feature_timestamp=feature_timestamp,
        metadata=metadata,
        raw_output=dict(expert_output),
    )


def from_statistics(
    probability: float,
    *,
    outcome: str = "home_win",
    expert_version: str = "",
    feature_timestamp: Optional[str] = None,
    id_fixture: Optional[Any] = None,
) -> ExpertOutput:
    """Statistics Expert (EXP-03): converte una probabilita' calibrata
    singola (un valore estratto per riga da `embedding_features()`, che
    ritorna una `pd.Series` per l'intero batch) in un vettore binario
    {outcome: p, not_outcome: 1-p}. `StatisticsExpert` non espone una
    propria `VERSION` (a differenza di EXP-01/02/04): il chiamante puo'
    valorizzarla esplicitamente se serve tracciarla altrove.
    """
    p = float(probability)
    return ExpertOutput(
        expert_name="statistics",
        expert_version=expert_version,
        probability_vector={outcome: p, f"not_{outcome}": 1.0 - p},
        feature_timestamp=feature_timestamp,
        metadata={"outcome": outcome, "id_fixture": id_fixture},
    )


def from_market_odds(signal: Mapping[str, Any]) -> ExpertOutput:
    """Market/Odds Expert (EXP-04): converte il dict restituito da
    `MarketOddsExpert.build_market_signal(...)`.

    `probability_vector` viene popolato dalle fair probabilities gia'
    calcolate dall'esperto (`compute_market_baseline`), una entry per
    outcome del mercato richiesto. Dispersione/movement/opening-latest-
    closing restano disponibili in `metadata` (non sono probabilita' di
    esito, non appartengono al `probability_vector`).
    """
    fair = signal.get("fair_probabilities") or {}
    probability_vector = {
        str(item.get("outcome")): float(item.get("fair_probability"))
        for item in (fair.get("outcomes") or [])
        if item.get("fair_probability") is not None
    }

    metadata = {
        "fixture_id": signal.get("fixture_id"),
        "market": signal.get("market"),
        "is_exclusive": fair.get("is_exclusive"),
        "overround": fair.get("overround"),
        "dispersion": signal.get("dispersion"),
        "movement": signal.get("movement"),
        "opening_latest_closing": signal.get("opening_latest_closing"),
    }

    return ExpertOutput(
        expert_name="market_odds",
        expert_version=str(signal.get("signal_version") or ""),
        probability_vector=probability_vector,
        feature_timestamp=signal.get("as_of"),
        metadata=metadata,
        raw_output=dict(signal),
    )


def from_predict_proba_expert(
    expert: Any,
    X: Any,
    *,
    expert_name: Optional[str] = None,
    feature_timestamps: Optional[Sequence[Optional[str]]] = None,
) -> list[ExpertOutput]:
    """Adapter generico per QUALUNQUE esperto con l'interfaccia comune
    `predict_proba`/`predict_proba_dict' + attributi `run_id`/`stage`
    (EXP-05 `DirectMarketExpert`, MARKET-05 `CornersExpert`, MARKET-06
    `CardsExpert`, e futuri esperti con la stessa interfaccia): un solo
    adapter, zero duplicazione, perche' condividono esattamente lo stesso
    contratto.

    Batch-oriented: `predict_proba(X)` produce un array 1D con una
    probabilita' per riga di `X`, quindi qui viene restituita una LISTA di
    `ExpertOutput` (uno per riga/fixture), non un singolo output aggregato.
    """
    if hasattr(expert, "predict_proba_dict"):
        proba_dict = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in expert.predict_proba_dict(X).items()}
        n = len(next(iter(proba_dict.values())))
        vectors = [{k: float(v[i]) for k, v in proba_dict.items()} for i in range(n)]
    else:
        proba = np.asarray(expert.predict_proba(X), dtype=float).reshape(-1)
        n = proba.shape[0]
        positive_label = getattr(expert, "positive_label", None) or "positive"
        vectors = [{positive_label: float(proba[i]), f"not_{positive_label}": 1.0 - float(proba[i])} for i in range(n)]

    timestamps = list(feature_timestamps) if feature_timestamps is not None else [None] * n
    if len(timestamps) != n:
        raise ValueError("feature_timestamps deve avere la stessa lunghezza di X")

    market = getattr(expert, "market", None)
    line = getattr(expert, "line", None)
    if expert_name:
        resolved_name = expert_name
    elif market and line is not None:
        resolved_name = f"{market}_line_{_line_label(line)}"
    elif market:
        resolved_name = str(market)
    else:
        resolved_name = type(expert).__name__

    return [
        ExpertOutput(
            expert_name=resolved_name,
            probability_vector=vectors[i],
            model_run_id=getattr(expert, "run_id", None),
            stage=getattr(expert, "stage", None),
            feature_timestamp=timestamps[i],
            metadata={
                "market": market,
                "line": line,
                "classification_type": getattr(expert, "classification_type", None),
                "outcome_semantics": getattr(expert, "outcome_semantics", None),
            },
        )
        for i in range(n)
    ]
