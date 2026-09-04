"""Pick Pool per Schedina Oracle (SLIP-01, Fase SCHEDINA).

Seleziona, da un insieme di pick candidati GIA' valutati da BET-04
(`Decision`/`decision_cards` di `DashboardService`), quelli "giocabili"
per un'eventuale Schedina Oracle (le combinazioni vere e proprie sono
SLIP-02/03, task successivi - qui SOLO la selezione del pool):

1. Solo PLAY (e opzionalmente BORDERLINE, se esplicitamente configurato -
   default OFF, stesso principio "nessun cambio di comportamento non
   richiesto" gia' seguito da BET-02/04).
2. Vincoli quota/EV (soglie VERSIONATE, mai hardcoded nel corpo della
   funzione - stesso principio di `DecisionThresholds`/BET-04).
3. Una selezione per outcome/mercato: se per la STESSA fixture+market
   arrivano piu' candidati (difesa in profondita': l'attuale
   `decision_cards` produce gia' un solo pick per mercato, ma questo modulo
   non assume nulla sulla sorgente), viene scelto un solo vincitore
   (priorita' decisione, poi EV, poi prob_edge) - MAI due pick sullo stesso
   mercato della stessa fixture nello stesso pool.

Acceptance criteria "Pool deterministico e tracciabile":
- deterministico: nessuna randomicita', ordinamento stabile (EV
  decrescente, poi fixture_id/market per rompere i pareggi) - stesso input
  produce sempre lo stesso output, nello stesso ordine;
- tracciabile: ogni pick incluso/escluso riporta un motivo esplicito
  (`inclusion_reason`/`exclusion_reason`, mai uno scarto silenzioso), il
  pool espone un `pool_id` (hash deterministico dai pick inclusi: stesso
  insieme di pick -> stesso `pool_id` sempre) e la `policy_version`
  applicata.

Nessuna logica di betting duplicata: questo modulo NON ricalcola
edge/EV/decision (BET-01/02/04, gia' calcolati a monte) - si limita a
filtrare/selezionare/ordinare pick gia' pronti.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class PickPoolPolicy:
    """Soglie versionate (mai hardcoded inline) per la selezione del pool.

    Valori di default DELIBERATAMENTE prudenti (nessuna taratura ancora
    richiesta da questo task): `include_borderline=False` (solo PLAY),
    nessun vincolo aggiuntivo di quota/EV/edge/campione oltre a quanto gia'
    garantito da BET-04 (un PLAY ha gia' EV/probabilita' sopra le soglie di
    `DecisionPolicy`). Una nuova taratura richiede una nuova istanza con
    una nuova `version` (vedi `with_overrides`), mai un edit silenzioso di
    questa (stesso principio di `DecisionPolicy`/BET-04)."""

    version: str = "pick_pool_policy_v1"
    include_borderline: bool = False
    min_odd: Optional[float] = None
    max_odd: Optional[float] = None
    min_ev: Optional[float] = None
    min_prob_edge: Optional[float] = None
    min_samples: int = 0

    def __post_init__(self) -> None:
        # Fail-fast su una configurazione palesemente inconsistente (mai
        # silenziosa), stesso principio di `DecisionThresholds` (BET-04).
        if self.min_odd is not None and self.max_odd is not None and self.min_odd > self.max_odd:
            raise ValueError(f"min_odd ({self.min_odd}) non puo' essere maggiore di max_odd ({self.max_odd})")
        if self.min_samples < 0:
            raise ValueError(f"min_samples non puo' essere negativo: {self.min_samples}")

    @property
    def allowed_decisions(self) -> frozenset:
        return frozenset({"PLAY", "BORDERLINE"}) if self.include_borderline else frozenset({"PLAY"})

    @staticmethod
    def with_overrides(
        include_borderline: bool = False,
        min_odd: Optional[float] = None,
        max_odd: Optional[float] = None,
        min_ev: Optional[float] = None,
        min_prob_edge: Optional[float] = None,
        min_samples: int = 0,
    ) -> "PickPoolPolicy":
        """Costruisce una policy con eventuali override (usata dall'endpoint
        API per parametri custom). Se i parametri coincidono con i default,
        ritorna la STESSA istanza globale `DEFAULT_PICK_POOL_POLICY` (stessa
        `version`, nessun cambio di comportamento non richiesto). Altrimenti
        calcola una `version` derivata DETERMINISTICAMENTE dai parametri
        (hash breve, stesso principio di `TeamStrengthExpert.VERSION`): mai
        la stessa `policy_version` per soglie diverse (acceptance criteria
        "tracciabile"), ma parametri identici producono sempre la stessa
        versione derivata."""
        candidate_fields = {
            "include_borderline": include_borderline,
            "min_odd": min_odd,
            "max_odd": max_odd,
            "min_ev": min_ev,
            "min_prob_edge": min_prob_edge,
            "min_samples": min_samples,
        }
        default_fields = {
            "include_borderline": DEFAULT_PICK_POOL_POLICY.include_borderline,
            "min_odd": DEFAULT_PICK_POOL_POLICY.min_odd,
            "max_odd": DEFAULT_PICK_POOL_POLICY.max_odd,
            "min_ev": DEFAULT_PICK_POOL_POLICY.min_ev,
            "min_prob_edge": DEFAULT_PICK_POOL_POLICY.min_prob_edge,
            "min_samples": DEFAULT_PICK_POOL_POLICY.min_samples,
        }
        if candidate_fields == default_fields:
            return DEFAULT_PICK_POOL_POLICY

        fingerprint = hashlib.sha1(json.dumps(candidate_fields, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        return PickPoolPolicy(version=f"pick_pool_policy_custom_{fingerprint}", **candidate_fields)


DEFAULT_PICK_POOL_POLICY = PickPoolPolicy()


@dataclass(frozen=True)
class CandidatePick:
    """Un pick candidato grezzo IN INPUT alla selezione del pool: stesso
    shape di `Decision` (BET-04) + `fixture_id`/`kickoff_at` (necessari per
    raggruppare per fixture/mercato e ordinare in modo deterministico).
    Nessuna dipendenza da DB/dict esterni: la conversione da `decision_cards`
    e' responsabilita' del layer di servizio (`pick_pool_service.py`)."""

    fixture_id: int
    market: str
    outcome: str
    decision: str
    p_model: Optional[float] = None
    p_market_fair: Optional[float] = None
    odd: Optional[float] = None
    fair_odd: Optional[float] = None
    prob_edge: Optional[float] = None
    ev: Optional[float] = None
    samples: int = 0
    model_run_id: Optional[str] = None
    model_name: Optional[str] = None
    policy_version: Optional[str] = None
    kickoff_at: Optional[str] = None


@dataclass(frozen=True)
class PoolPick:
    """Un pick candidato che ha SUPERATO i filtri del pool (acceptance
    criteria "tracciabile"): il candidato originale + il motivo esplicito
    per cui e' stato incluso + la versione di policy applicata."""

    candidate: CandidatePick
    inclusion_reason: str
    pool_policy_version: str


@dataclass(frozen=True)
class ExcludedPick:
    """Un pick candidato SCARTATO dal pool, con il motivo esplicito
    (acceptance criteria "tracciabile" - mai uno scarto silenzioso)."""

    candidate: CandidatePick
    exclusion_reason: str


@dataclass
class PickPoolResult:
    """Output del pool (acceptance criteria "Pool deterministico e
    tracciabile"): `pool_id` e' un hash deterministico dai pick inclusi,
    `policy_version` la policy REALMENTE applicata, `excluded` rende
    ispezionabile ogni scarto."""

    pool_id: str
    generated_at: str
    policy_version: str
    picks: list = field(default_factory=list)
    excluded: list = field(default_factory=list)


def _ev_sort_key(ev: Optional[float]) -> float:
    # Attenzione: `ev or float("-inf")` tratterebbe erroneamente un EV=0.0
    # (valore legittimo) come "-inf" (0.0 e' falsy in Python) - controllo
    # esplicito su `is not None`.
    return float(ev) if ev is not None else float("-inf")


def _prob_edge_sort_key(prob_edge: Optional[float]) -> float:
    return float(prob_edge) if prob_edge is not None else float("-inf")


_DECISION_RANK = {"PLAY": 0, "BORDERLINE": 1, "NO BET": 2}


def _better_candidate(a: CandidatePick, b: CandidatePick) -> CandidatePick:
    """Vincitore tra due candidati sulla STESSA chiave (fixture_id, market)
    - "una selezione per outcome/mercato" (acceptance criteria): priorita'
    decisione (PLAY prima di BORDERLINE), poi EV decrescente, poi prob_edge
    decrescente. Confronto interamente deterministico, mai casuale."""
    key_a = (_DECISION_RANK.get(a.decision, 99), -_ev_sort_key(a.ev), -_prob_edge_sort_key(a.prob_edge))
    key_b = (_DECISION_RANK.get(b.decision, 99), -_ev_sort_key(b.ev), -_prob_edge_sort_key(b.prob_edge))
    return a if key_a <= key_b else b


def _fails_policy(candidate: CandidatePick, policy: PickPoolPolicy) -> Optional[str]:
    """Ritorna il motivo di esclusione (stringa leggibile) se il candidato
    NON supera i vincoli di `policy`, altrimenti `None`. Nessuno scarto
    silenzioso (acceptance criteria "tracciabile")."""
    if candidate.decision not in policy.allowed_decisions:
        return f"decision_not_allowed:{candidate.decision}"
    if candidate.odd is None or float(candidate.odd) <= 0.0:
        return "odd_missing"
    if policy.min_odd is not None and float(candidate.odd) < policy.min_odd:
        return f"odd_below_min:{candidate.odd}<{policy.min_odd}"
    if policy.max_odd is not None and float(candidate.odd) > policy.max_odd:
        return f"odd_above_max:{candidate.odd}>{policy.max_odd}"
    if policy.min_ev is not None and (candidate.ev is None or float(candidate.ev) < policy.min_ev):
        return f"ev_below_min:{candidate.ev}<{policy.min_ev}"
    if policy.min_prob_edge is not None and (
        candidate.prob_edge is None or float(candidate.prob_edge) < policy.min_prob_edge
    ):
        return f"prob_edge_below_min:{candidate.prob_edge}<{policy.min_prob_edge}"
    if candidate.samples < policy.min_samples:
        return f"samples_below_min:{candidate.samples}<{policy.min_samples}"
    return None


def _pool_id(picks: list[PoolPick]) -> str:
    """Hash deterministico dai pick inclusi (acceptance criteria
    "tracciabile"): stesso insieme di pick -> stesso `pool_id` sempre,
    indipendentemente dall'ordine di iterazione in input."""
    payload = sorted(
        (p.candidate.fixture_id, p.candidate.market, p.candidate.outcome, p.candidate.model_run_id or "")
        for p in picks
    )
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def build_pick_pool(
    candidates: list[CandidatePick],
    policy: PickPoolPolicy = DEFAULT_PICK_POOL_POLICY,
    generated_at: Optional[datetime] = None,
) -> PickPoolResult:
    """Funzione pura (nessun DB/IO): applica la policy, la regola "una
    selezione per outcome/mercato" e produce un ordinamento deterministico.
    """
    excluded: list[ExcludedPick] = []
    passed: dict[tuple, CandidatePick] = {}
    passed_reason: dict[tuple, str] = {}

    for candidate in candidates:
        reason = _fails_policy(candidate, policy)
        if reason is not None:
            excluded.append(ExcludedPick(candidate=candidate, exclusion_reason=reason))
            continue

        key = (candidate.fixture_id, candidate.market)
        current = passed.get(key)
        if current is None:
            passed[key] = candidate
            passed_reason[key] = f"passed_policy:{policy.version}"
            continue

        winner = _better_candidate(current, candidate)
        loser = candidate if winner is current else current
        excluded.append(
            ExcludedPick(candidate=loser, exclusion_reason=f"superseded_same_fixture_market:{policy.version}")
        )
        passed[key] = winner

    picks = [
        PoolPick(candidate=c, inclusion_reason=passed_reason[key], pool_policy_version=policy.version)
        for key, c in passed.items()
    ]
    # Ordinamento deterministico: EV decrescente, poi fixture_id/market per
    # rompere i pareggi (mai un ordine dipendente dall'iterazione in input).
    picks.sort(key=lambda p: (-_ev_sort_key(p.candidate.ev), p.candidate.fixture_id, p.candidate.market))
    excluded.sort(key=lambda e: (e.candidate.fixture_id, e.candidate.market, e.candidate.outcome))

    generated_at = generated_at or datetime.now(timezone.utc)
    return PickPoolResult(
        pool_id=_pool_id(picks),
        generated_at=generated_at.isoformat(),
        policy_version=policy.version,
        picks=picks,
        excluded=excluded,
    )


