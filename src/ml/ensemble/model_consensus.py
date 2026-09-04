"""Model Consensus per spiegabilita' (ORACLE-04, Fase ENSEMBLE).

Per una singola fixture/mercato, assembla:
1. l'output di ciascun Oracle Expert GENERICAMENTE applicabile a qualunque
   mercato di `FilterMarketService.SUPPORTED_MARKETS` (Direct Expert gia'
   allenato/registrato, EXP-05; Market/Odds Expert dalle quote gia' salvate,
   EXP-04);
2. l'Oracle finale: la predizione del meta-model calibrato (ORACLE-02/03)
   se ne esiste uno registrato per quel mercato, ALTRIMENTI (fallback
   esplicito, mai un errore silenzioso) la media semplice delle probabilita'
   dei singoli esperti disponibili — comunque un calcolo reale, mai un
   valore inventato;
3. la dispersione del consensus: quanto gli esperti disponibili sono
   d'accordo tra loro (deviazione standard delle probabilita' comparabili).

Esclusioni deliberate dal consensus (nessuna riscrittura non necessaria,
nessun mapping arbitrario):
- `TeamStrengthExpert` (EXP-01): non produce `probability_vector` per
  costruzione (ORACLE-01) — espone solo rating riusati da altri esperti,
  quindi non contribuisce a un consensus di probabilita'.
- `GoalDistributionExpert` (EXP-02): applicabile SOLO ai mercati "gol"
  (under_over_*, BTTS), non generico per TUTTI gli 8 mercati supportati:
  generalizzarlo qui richiederebbe logica specifica per mercato non ancora
  esistente in forma riusabile, fuori scope (priorita' P2, "non anticipare
  task successivi").
- `StatisticsExpert` (EXP-03): `embedding_features()` richiede un fit
  in-memory (nessun metodo `load_production`/persistenza su
  `ModelRegistry`), quindi non e' caricabile on-demand in un endpoint
  stateless senza reinventare una persistenza che oggi non esiste.

Vincolo "nessuna spiegazione inventata": ogni campo del report e' un numero
o un metadato realmente calcolato; se un valore non e' calcolabile
(esperto assente, meta-model non registrato, outcome non comparabile), il
campo resta `None`/l'esperto viene omesso dal calcolo comparabile — MAI un
valore fittizio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.ml.ensemble.adapters import from_market_odds, from_predict_proba_expert
from src.ml.ensemble.expert_output import ExpertOutput
from src.ml.ensemble.stacking import build_meta_features_from_expert_outputs
from src.ml.experts.direct.direct_market_expert import DirectMarketExpert
from src.ml.experts.market.market_odds_expert import MarketOddsExpert
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry

# Prefissi usati da ORACLE-02 (`stacking.py`) e ORACLE-03
# (`oracle_calibration.py`) per registrare un meta-model: la ricerca DEVE
# restare esplicita su questo prefisso (mai un `get_latest(market)`/
# `get_production(market)` generico), perche' un meta-model e un modello
# "diretto" (EXP-05, `train_multi_market.py`) possono condividere lo stesso
# `market` nel registry con `model_name` diverso.
_META_MODEL_NAME_PREFIXES = ("meta_", "oracle_ensemble_")

# Outcome "positivo" atteso nelle fair probabilities del Market/Odds Expert
# per ciascun mercato con soglia FISSA (stessa convenzione gia' in uso in
# `src/api/dashboard_service.py::_baseline_outcome_for_prediction`, qui
# esplicitata per rendere l'output del Market/Odds Expert comparabile con
# quello del Direct Expert). Corners/cards non compaiono: la soglia
# ("linea") e' un parametro del modello registrato, non deducibile in modo
# statico dal solo nome mercato -> trattati come "non comparabili".
_POSITIVE_OUTCOME_LABEL_FOR_MARKET: dict[str, str] = {
    "h2h": "Home",
    "dc": "Home/Draw",
    "goal_no_goal": "Yes",
    "under_over_1_5": "Over 1.5",
    "under_over_2_5": "Over 2.5",
    "under_over_3_5": "Over 3.5",
    "under_over_4_5": "Over 4.5",
}

_PREDICTION_FRAME_META_COLUMNS = ["market", "id_fixture", "season", "league", "prediction_at"]


def _agreement_level(dispersion: Optional[float]) -> Optional[str]:
    """Etichetta leggibile della dispersione (soglie euristiche, stesso
    ordine di grandezza di altre soglie gia' in uso nel progetto, es.
    `DEFAULT_MIN_CALIBRATION_SAMPLES`/`_value_decision`): MAI un giudizio
    "inventato", solo una fascia su un numero gia' calcolato."""
    if dispersion is None:
        return None
    if dispersion < 0.05:
        return "high"
    if dispersion < 0.15:
        return "medium"
    return "low"


@dataclass
class ModelConsensusReport:
    """Payload strutturato (acceptance criteria "API restituisce consensus
    strutturato"): solo numeri/metadati calcolati, nessun testo generato."""

    fixture_id: int
    market: str
    experts: list[dict[str, Any]]
    oracle_final: Optional[dict[str, Any]]
    consensus: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _expert_output_payload(
    output: ExpertOutput,
    comparable_probability: Optional[float],
    comparable: bool,
) -> dict[str, Any]:
    return {
        "expert_name": output.expert_name,
        "expert_version": output.expert_version,
        "probability_vector": output.probability_vector,
        "probability": comparable_probability,
        "comparable": comparable,
        "confidence": output.confidence,
        "model_run_id": output.model_run_id,
        "stage": output.stage,
        "feature_timestamp": output.feature_timestamp,
    }


def compute_consensus_from_expert_outputs(
    market: str,
    fixture_id: int,
    expert_outputs: list[ExpertOutput],
    positive_label: Optional[str] = None,
    meta_model: Optional[Any] = None,
    meta_model_run_id: Optional[str] = None,
    meta_model_feature_names: Optional[list[str]] = None,
) -> ModelConsensusReport:
    """Funzione pura (nessun DB/registry): dati gli `ExpertOutput` GIA'
    calcolati per questa fixture ed eventualmente un meta-model gia'
    caricato, produce il report di consensus strutturato.

    - `positive_label`: chiave del Direct Expert (`DirectMarketExpert.positive_label`)
      usata per estrarre una probabilita' "comparabile" da CIASCUN esperto
      (via `_POSITIVE_OUTCOME_LABEL_FOR_MARKET[market]` per il Market/Odds
      Expert). Se un esperto non espone quella chiave, viene comunque
      incluso in `experts` (con `probability_vector` completo) ma escluso
      dal calcolo di dispersione/oracle-finale-semplice (`comparable=False`):
      mai un valore stimato al posto di uno mancante.
    - Se `meta_model` e' fornito, l'Oracle finale e' la sua predizione sulle
      meta-feature costruite dagli STESSI `expert_outputs`
      (`build_meta_features_from_expert_outputs`, ORACLE-02).
    - Altrimenti (fallback esplicito, mai un'eccezione): l'Oracle finale e'
      la media semplice delle probabilita' comparabili disponibili.
    """
    warnings: list[str] = []
    comparable_probabilities: list[float] = []
    expert_payloads: list[dict[str, Any]] = []

    for output in expert_outputs:
        comparable = False
        comparable_probability: Optional[float] = None

        lookup_label = positive_label
        if output.expert_name == "market_odds":
            lookup_label = _POSITIVE_OUTCOME_LABEL_FOR_MARKET.get(market)

        if lookup_label and lookup_label in output.probability_vector:
            comparable_probability = float(output.probability_vector[lookup_label])
            comparable = True

        expert_payloads.append(
            _expert_output_payload(output, comparable_probability=comparable_probability, comparable=comparable)
        )
        if comparable and comparable_probability is not None:
            comparable_probabilities.append(comparable_probability)

    if not expert_payloads:
        warnings.append("no_expert_output_available")

    oracle_final: Optional[dict[str, Any]] = None
    if meta_model is not None and expert_outputs:
        try:
            meta_features = build_meta_features_from_expert_outputs([expert_outputs])
            if meta_model_feature_names:
                meta_features = meta_features.reindex(columns=meta_model_feature_names, fill_value=0.0)
            proba = np.asarray(meta_model.predict_proba(meta_features), dtype=float)
            p1 = float(proba[0, -1]) if proba.ndim == 2 else float(proba.reshape(-1)[0])
            oracle_final = {
                "probability": p1,
                "source": "meta_model",
                "model_run_id": meta_model_run_id,
            }
        except Exception as exc:
            warnings.append(f"meta_model_failed: {exc}")

    if oracle_final is None:
        if comparable_probabilities:
            oracle_final = {
                "probability": float(np.mean(comparable_probabilities)),
                "source": "simple_consensus_mean",
                "model_run_id": None,
            }
        else:
            warnings.append("oracle_final_not_available")

    dispersion = float(np.std(comparable_probabilities)) if len(comparable_probabilities) >= 1 else None

    return ModelConsensusReport(
        fixture_id=fixture_id,
        market=market,
        experts=expert_payloads,
        oracle_final=oracle_final,
        consensus={
            "expert_count": len(expert_payloads),
            "comparable_expert_count": len(comparable_probabilities),
            "dispersion_std": dispersion,
            "agreement_level": _agreement_level(dispersion),
        },
        warnings=warnings,
    )


def _find_meta_model_run(registry: ModelRegistry, market: str) -> Optional[dict[str, Any]]:
    """Cerca ESPLICITAMENTE un meta-model (ORACLE-02/03) registrato per
    `market`, distinguendolo da un eventuale modello "diretto" sullo stesso
    mercato tramite il prefisso di `model_name` (mai un `get_latest`/
    `get_production` generico, che mescolerebbe i due — vincolo generale
    "latest model non equivale automaticamente a production model")."""
    runs = registry.tail(limit=1000, market=market)
    meta_runs = [r for r in runs if str(r.get("model_name") or "").startswith(_META_MODEL_NAME_PREFIXES)]
    if not meta_runs:
        return None

    production = [r for r in meta_runs if r.get("current_stage") == "production"]
    candidates = production or meta_runs
    return max(candidates, key=lambda r: r.get("created_at", ""))


def build_model_consensus_for_fixture(
    market: str,
    fixture_id: int,
    as_of: Optional[datetime] = None,
    registry: Optional[ModelRegistry] = None,
    filter_service: Optional[FilterMarketService] = None,
    market_odds_expert: Optional[MarketOddsExpert] = None,
) -> ModelConsensusReport:
    """Orchestratore DB-aware: costruisce gli `ExpertOutput` REALI (Direct
    Expert gia' allenato EXP-05 + Market/Odds Expert dalle quote EXP-04,
    quando disponibili), cerca un eventuale meta-model registrato
    (ORACLE-02/03) per questo mercato, poi delega a
    `compute_consensus_from_expert_outputs`.

    Robusto per costruzione: se un singolo esperto non e' disponibile
    (nessun modello registrato, nessuna quota salvata, feature insufficienti
    per questa fixture) viene semplicemente omesso — MAI un'eccezione che
    blocca l'intero consensus.
    """
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise ValueError(f"Mercato non supportato: {market}")

    registry = registry or ModelRegistry()
    filter_service = filter_service or FilterMarketService()
    market_odds_expert = market_odds_expert or MarketOddsExpert()
    as_of = as_of or datetime.now(timezone.utc)

    expert_outputs: list[ExpertOutput] = []
    positive_label: Optional[str] = None

    # 1) Direct Expert (EXP-05): stage 'production', altrimenti 'candidate'
    #    piu' recente (mai silenziosamente spacciato per production: lo
    #    `stage` reale resta tracciato nel payload).
    direct_run = registry.get_production(market=market) or registry.get_latest(market=market)
    if direct_run and not str(direct_run.get("model_name") or "").startswith(_META_MODEL_NAME_PREFIXES):
        try:
            direct_expert = DirectMarketExpert._from_run(direct_run)
            positive_label = direct_expert.positive_label
            frame = filter_service.build_prediction_frame(market=market, fixture_id=fixture_id)
            if frame is not None and not frame.empty:
                X = frame.drop(columns=_PREDICTION_FRAME_META_COLUMNS, errors="ignore")
                selected_features = direct_expert.feature_names
                if selected_features:
                    for feature_name in selected_features:
                        if feature_name not in X.columns:
                            X[feature_name] = 0.0
                    X = X[selected_features]
                outputs = from_predict_proba_expert(direct_expert, X, feature_timestamps=[as_of.isoformat()])
                expert_outputs.extend(outputs)
        except Exception:
            pass  # esperto non disponibile per questa fixture: nessun crash del consensus

    # 2) Market/Odds Expert (EXP-04): fair probability dalle quote gia'
    #    salvate (`odds_snapshot`), se presenti per questa fixture/mercato.
    try:
        signal = market_odds_expert.build_market_signal(fixture_id=fixture_id, market=market, as_of=as_of)
        if (signal.get("fair_probabilities") or {}).get("outcomes"):
            expert_outputs.append(from_market_odds(signal))
    except Exception:
        pass

    meta_run = _find_meta_model_run(registry, market=market)
    meta_model = None
    meta_model_feature_names: Optional[list[str]] = None
    if meta_run:
        try:
            meta_model = joblib.load(meta_run["model_path"])
            meta_model_feature_names = list(meta_run.get("feature_names") or [])
        except Exception:
            meta_model = None

    return compute_consensus_from_expert_outputs(
        market=market,
        fixture_id=fixture_id,
        expert_outputs=expert_outputs,
        positive_label=positive_label,
        meta_model=meta_model,
        meta_model_run_id=meta_run.get("run_id") if meta_run else None,
        meta_model_feature_names=meta_model_feature_names,
    )
