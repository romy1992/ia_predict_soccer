"""Generatore Schedine per Schedina Oracle (SLIP-03, Fase SCHEDINA).

Le pick "giocabili" selezionate dal Pick Pool (SLIP-01, `pick_pool.py`) e
classificate dal Correlation Engine (SLIP-02, `correlation_engine.py`) sono
oggi solo un elenco piatto: questo modulo (PURO, nessun DB/IO, stesso
principio dei due moduli precedenti) le COMBINA in vere e proprie schedine
da 2/3/4 eventi, secondo tre profili di rischio distinti e VERSIONATI
(`SlipProfile`, stesso principio di `PickPoolPolicy`/`CorrelationRuleSet`):

- `SAFE`: solo pick a probabilita' modello alta e quota contenuta, ZERO
  coppie correlate (`PENALTY`) ammesse nella stessa schedina - ogni evento
  incluso resta statisticamente indipendente dagli altri.
- `BALANCED`: soglie intermedie, ammette al massimo 1 coppia `PENALTY`.
- `AGGRESSIVE`: soglie piu' permissive (probabilita' minima piu' bassa,
  quota massima per singolo evento piu' alta), ammette fino a 3 coppie
  `PENALTY`.

Una coppia `EXCLUDE` (contraddizione logica same-match, SLIP-02) blocca
SEMPRE la combinazione, per QUALSIASI profilo: non e' un parametro
tarabile, e' un vincolo assoluto (una schedina logicamente impossibile non
deve mai essere proposta, a prescindere da quanto aggressivo sia il
profilo di rischio).

Metodo esplicito per quota/probabilita' combinate (acceptance criteria
"Quote combinate e probabilita' dichiarate con metodo esplicito"):
- `combined_odd`: prodotto semplice delle quote dei singoli eventi (stesso
  metodo usato da qualunque bookmaker per una schedina multipla - un fatto
  aritmetico, non una stima).
- `naive_probability`: prodotto semplice delle probabilita' modello
  (l'assunzione "ingenua" di indipendenza, SLIP-02).
- `adjusted_probability`: probabilita' CORRETTA per la correlazione
  same-match (SLIP-02, `evaluate_combination` - raggruppamento in cluster
  via le coppie `PENALTY`, minimo per cluster), sempre <= `naive_probability`
  quando esistono correlazioni positive - MAI la probabilita' "ingenua"
  spacciata per corretta.
- `combined_ev`: SEMPRE calcolato con `adjusted_probability` (mai con
  quella naive, che sovrastimerebbe l'EV reale in presenza di correlazioni
  positive) - `adjusted_probability * combined_odd - 1`, stesso principio
  di `ev = p_model*odd-1` del Value Engine (BET-02) applicato all'intera
  combinazione.
- `risk_score`: `1 - adjusted_probability`, la probabilita' (secondo il
  modello Oracle, corretta per la correlazione) che la schedina NON si
  verifichi - una lettura diretta della stessa probabilita' gia'
  dichiarata, non un numero indipendente "inventato".

Ranking (acceptance criteria "Ranking probability/EV/risk", "Cosa deve
fare" punto 1): le schedine generate per ciascun profilo sono ordinate per
`combined_ev` decrescente, poi `adjusted_probability` decrescente, poi
`risk_score` crescente, poi `slip_id` per rompere i pareggi in modo
deterministico - mai un ordine dipendente dall'iterazione in input.

Limite pratico DOCUMENTATO (mai un comportamento nascosto): generare TUTTE
le combinazioni possibili da un pool potenzialmente grande (es. 40+ pick
in una giornata con molte partite) esploderebbe combinatoriamente per
schedine a 4 eventi. Il pool considerato per ciascun profilo viene quindi
troncato a `max_pool_size` elementi (default 14, ordinati in modo
deterministico per EV decrescente prima del troncamento): il troncamento e'
sempre tracciato in `BetslipGenerationResult.warnings`, mai silenzioso.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

from src.oracle.betslip.correlation_engine import (
    DEFAULT_CORRELATION_RULESET,
    PENALTY,
    CorrelationRuleSet,
    evaluate_combination,
)
from src.oracle.betslip.pick_pool import CandidatePick

# ---------------------------------------------------------------------------
# Profili di rischio versionati.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlipProfile:
    """Parametri versionati di un profilo di rischio (mai hardcoded inline
    nel corpo di `generate_betslips`): una nuova taratura richiede una
    nuova istanza con una nuova `version`, mai un edit silenzioso di questa
    (stesso principio di `PickPoolPolicy`/`CorrelationRuleSet`).

    Il vincolo "niente EXCLUDE" NON e' un campo di questa classe: e'
    assoluto, applicato incondizionatamente da `generate_betslips` per
    QUALSIASI profilo (vedi docstring di modulo)."""

    name: str
    version: str
    min_legs: int
    max_legs: int
    min_leg_probability: Optional[float] = None
    max_leg_odd: Optional[float] = None
    min_combined_odd: Optional[float] = None
    max_combined_odd: Optional[float] = None
    max_penalty_pairs: int = 0
    min_adjusted_probability: Optional[float] = None
    min_combined_ev: Optional[float] = None
    risk_label: str = ""

    def __post_init__(self) -> None:
        # Fail-fast su una configurazione palesemente inconsistente (mai
        # silenziosa), stesso principio di `PickPoolPolicy` (BET-04/SLIP-01).
        if self.min_legs < 2:
            raise ValueError(f"min_legs deve essere almeno 2 (una schedina e' una combinazione): {self.min_legs}")
        if self.max_legs < self.min_legs:
            raise ValueError(f"max_legs ({self.max_legs}) non puo' essere minore di min_legs ({self.min_legs})")
        if self.max_legs > 4:
            raise ValueError(f"SLIP-03 supporta schedine di massimo 4 eventi, richiesto max_legs={self.max_legs}")
        if self.max_penalty_pairs < 0:
            raise ValueError(f"max_penalty_pairs non puo' essere negativo: {self.max_penalty_pairs}")


# Valori di default DELIBERATAMENTE documentati (nessun numero "magico" non
# spiegato): SAFE privilegia probabilita' modello alta e zero correlazioni
# positive ammesse (eventi davvero indipendenti); BALANCED/AGGRESSIVE
# allargano progressivamente le soglie di quota/probabilita' per singolo
# evento e la tolleranza a correlazioni PENALTY - MAI la tolleranza a
# coppie EXCLUDE, sempre vietate (vedi docstring di modulo).
SAFE_PROFILE = SlipProfile(
    name="SAFE",
    version="slip_profile_safe_v3_diversified",
    min_legs=2,
    max_legs=2,
    min_leg_probability=0.55,
    max_leg_odd=2.50,
    max_penalty_pairs=0,
    min_adjusted_probability=0.25,
    risk_label="LOW",
)

BALANCED_PROFILE = SlipProfile(
    name="BALANCED",
    version="slip_profile_balanced_v3_diversified",
    min_legs=2,
    max_legs=3,
    min_leg_probability=0.40,
    max_leg_odd=4.50,
    max_penalty_pairs=1,
    min_adjusted_probability=0.10,
    risk_label="MEDIUM",
)

AGGRESSIVE_PROFILE = SlipProfile(
    name="AGGRESSIVE",
    version="slip_profile_aggressive_v3_diversified",
    min_legs=3,
    max_legs=4,
    min_leg_probability=0.25,
    max_leg_odd=8.00,
    max_penalty_pairs=3,
    min_adjusted_probability=0.03,
    risk_label="HIGH",
)

DEFAULT_SLIP_PROFILES: tuple[SlipProfile, ...] = (SAFE_PROFILE, BALANCED_PROFILE, AGGRESSIVE_PROFILE)


@dataclass(frozen=True)
class SlipDecisionPolicy:
    version: str = "slip_decision_policy_v1"
    min_edge_percent: float = 2.0

    def __post_init__(self) -> None:
        if self.min_edge_percent < 0:
            raise ValueError("min_edge_percent non puo' essere negativo")


DEFAULT_SLIP_DECISION_POLICY = SlipDecisionPolicy()


@dataclass(frozen=True)
class BetslipDiversificationPolicy:
    version: str = "betslip_diversification_v1"
    max_candidates_per_family: int = 4
    max_overlap_ratio: float = 0.5
    safe_max_legs_per_family: int = 1
    balanced_max_legs_per_family: int = 1
    aggressive_max_legs_per_family: int = 2

    def max_legs_for_family(self, profile_name: str) -> int:
        return {
            "SAFE": self.safe_max_legs_per_family,
            "BALANCED": self.balanced_max_legs_per_family,
            "AGGRESSIVE": self.aggressive_max_legs_per_family,
        }.get(profile_name, 1)


DEFAULT_DIVERSIFICATION_POLICY = BetslipDiversificationPolicy()


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


@dataclass
class GeneratedSlip:
    """Una schedina generata (acceptance criteria "Output spiegabile"):
    `legs` riusa direttamente `CandidatePick` (nessuna duplicazione dello
    shape gia' definito da SLIP-01), `findings` riporta le eventuali coppie
    `PENALTY` (mai una correlazione applicata silenziosamente), `explanation`
    e' una descrizione testuale del metodo di calcolo usato."""

    slip_id: str
    profile_name: str
    profile_version: str
    risk_label: str
    n_legs: int
    legs: list = field(default_factory=list)  # list[CandidatePick]
    combined_odd: float = 0.0
    naive_probability: Optional[float] = None
    adjusted_probability: Optional[float] = None
    combined_ev: Optional[float] = None
    combined_model_void_odd: Optional[float] = None
    combined_edge_absolute: Optional[float] = None
    combined_edge_percent: Optional[float] = None
    combined_expected_roi: Optional[float] = None
    combined_expected_roi_percent: Optional[float] = None
    combined_play_threshold: Optional[float] = None
    slip_min_edge_percent: float = 0.0
    decision_policy_version: str = ""
    situation: str = "NO BET"
    situation_reason: str = ""
    risk_score: Optional[float] = None
    penalty_pairs: int = 0
    correlation_ruleset_version: str = ""
    diversification_policy_version: str = ""
    findings: list = field(default_factory=list)  # list[CorrelationFinding], solo non-INDEPENDENT
    explanation: str = ""
    status: str = "PROPOSED"
    is_official: bool = False
    shadow_status: Optional[str] = None


@dataclass
class BetslipGenerationResult:
    """Output completo (tutti e tre i profili): `pool_considered` e'
    tracciabilita' su quante pick eligibili erano disponibili PRIMA del
    troncamento per profilo, `warnings` rende esplicito ogni troncamento
    del pool o combinazione impossibile per mancanza di pick (mai un
    comportamento silenzioso)."""

    generated_at: str
    correlation_ruleset_version: str
    pool_considered: int
    profiles: dict = field(default_factory=dict)  # name -> list[GeneratedSlip]
    decision_groups: dict = field(default_factory=dict)  # label -> profile -> slips
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Selezione/ordinamento del pool (deterministico).
# ---------------------------------------------------------------------------


def _ev_sort_key(value: Optional[float]) -> float:
    # Attenzione: `value or float("-inf")` tratterebbe erroneamente un
    # valore 0.0 (legittimo) come "-inf" (0.0 e' falsy in Python) -
    # controllo esplicito su `is not None`, stesso principio di `pick_pool.py`.
    return float(value) if value is not None else float("-inf")


def _market_family(market: str) -> str:
    value = str(market or "").lower()
    if value in {"h2h", "1x2", "dc", "double_chance"}:
        return "RESULT"
    if value.startswith("under_over_"):
        return "TOTALS"
    if value in {"goal_no_goal", "btts"}:
        return "BTTS"
    if value == "corners":
        return "CORNERS"
    if value == "cards":
        return "CARDS"
    return value.upper() or "UNKNOWN"


def _is_eligible(candidate: CandidatePick, allowed_decisions: frozenset[str]) -> bool:
    """Una pick e' utilizzabile in una schedina solo se ha sia una quota
    valida (per la quota combinata) sia una probabilita' modello (per la
    probabilita' dichiarata) - senza le quali "metodo esplicito" non
    sarebbe rispettabile."""
    return (
        candidate.decision in allowed_decisions
        and candidate.odd is not None
        and float(candidate.odd) > 0.0
        and candidate.p_model is not None
        and 0.0 < float(candidate.p_model) <= 1.0
    )


def _base_eligible_pool(
    candidates: list[CandidatePick],
    allowed_decisions: frozenset[str],
) -> list[CandidatePick]:
    eligible = [c for c in candidates if _is_eligible(c, allowed_decisions)]
    # Ordinamento deterministico (EV decrescente, poi fixture_id/market):
    # stesso principio di `build_pick_pool`, garantisce che il troncamento
    # successivo scelga sempre le stesse pick per lo stesso input.
    eligible.sort(key=lambda c: (-_ev_sort_key(c.ev), c.fixture_id, c.market))
    return eligible


def _profile_eligible_pool(
    pool: list[CandidatePick],
    profile: SlipProfile,
    max_pool_size: int,
    diversification_policy: BetslipDiversificationPolicy,
) -> tuple[list[CandidatePick], list[str]]:
    warnings: list[str] = []
    filtered = [
        c
        for c in pool
        if (profile.min_leg_probability is None or float(c.p_model) >= profile.min_leg_probability)
        and (profile.max_leg_odd is None or float(c.odd) <= profile.max_leg_odd)
    ]
    grouped: dict[str, list[CandidatePick]] = {}
    for candidate in filtered:
        grouped.setdefault(_market_family(candidate.market), []).append(candidate)
    for family in grouped:
        grouped[family] = grouped[family][: diversification_policy.max_candidates_per_family]

    family_order = sorted(
        grouped,
        key=lambda family: (
            -_ev_sort_key(grouped[family][0].ev),
            family,
        ),
    )
    diversified: list[CandidatePick] = []
    offset = 0
    while len(diversified) < max_pool_size:
        added = False
        for family in family_order:
            if offset < len(grouped[family]):
                diversified.append(grouped[family][offset])
                added = True
                if len(diversified) >= max_pool_size:
                    break
        if not added:
            break
        offset += 1
    if len(filtered) > len(diversified):
        warnings.append(
            f"pool_truncated:{profile.name}:{len(filtered)}_to_{len(diversified)}"
        )
        warnings.append(
            f"pool_diversified:{profile.name}:{len(filtered)}_to_{len(diversified)}:"
            f"{diversification_policy.version}"
        )
    return diversified, warnings


def _slip_id(
    profile: SlipProfile,
    legs: list[CandidatePick],
    policy: SlipDecisionPolicy,
    diversification_policy: BetslipDiversificationPolicy,
) -> str:
    """Hash deterministico dalle leg incluse (stesso principio di
    `pick_pool._pool_id`): stesso profilo + stesso insieme di leg -> stesso
    `slip_id` sempre, indipendentemente dall'ordine di iterazione."""
    payload = sorted(
        (
            leg.fixture_id,
            leg.market,
            leg.outcome,
            leg.model_run_id or "",
            leg.policy_version or "",
        )
        for leg in legs
    )
    serialized = json.dumps(
        {
            "profile": profile.name,
            "profile_version": profile.version,
            "decision_policy": policy.version,
            "diversification_policy": diversification_policy.version,
            "legs": payload,
        },
        sort_keys=True,
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def _combined_value_metrics(
    combined_odd: float,
    adjusted_probability: Optional[float],
    policy: SlipDecisionPolicy,
) -> dict[str, Optional[float]]:
    if adjusted_probability is None or not 0.0 < adjusted_probability <= 1.0:
        return {
            "model_void_odd": None,
            "edge_absolute": None,
            "edge_percent": None,
            "expected_roi": None,
            "expected_roi_percent": None,
            "play_threshold": None,
        }
    model_void = 1.0 / adjusted_probability
    expected_roi = adjusted_probability * combined_odd - 1.0
    return {
        "model_void_odd": model_void,
        "edge_absolute": combined_odd - model_void,
        "edge_percent": ((combined_odd / model_void) - 1.0) * 100.0,
        "expected_roi": expected_roi,
        "expected_roi_percent": expected_roi * 100.0,
        "play_threshold": model_void * (1.0 + policy.min_edge_percent / 100.0),
    }


def _classify_slip(
    legs: list[CandidatePick],
    metrics: dict[str, Optional[float]],
    evaluation_valid: bool,
) -> tuple[str, str]:
    if not evaluation_valid:
        return "NO BET", "Combinazione incompatibile"
    if any(leg.decision == "NO BET" for leg in legs):
        return "NO BET", "Almeno una selezione e' NO BET"
    if metrics["expected_roi"] is None or metrics["model_void_odd"] is None:
        return "NO BET", "Dati necessari non disponibili"
    if metrics["expected_roi"] <= 0.0:
        return "NO BET", "Expected ROI combinato non positivo"
    if any(leg.decision != "PLAY" for leg in legs):
        return "BORDERLINE", "Almeno una selezione non e' PLAY"
    if metrics["play_threshold"] is not None and metrics["edge_absolute"] is not None:
        combined_odd = metrics["model_void_odd"] + metrics["edge_absolute"]
        if combined_odd >= metrics["play_threshold"]:
            return "PLAY", "Tutte le selezioni sono PLAY e la soglia combinata e' superata"
    return "BORDERLINE", "Valore positivo, ma margine combinato insufficiente"


def _explanation(
    profile: SlipProfile,
    legs: list[CandidatePick],
    combined_odd: float,
    naive: Optional[float],
    adjusted: Optional[float],
    combined_ev: Optional[float],
    penalty_count: int,
) -> str:
    """Descrizione testuale del metodo di calcolo (acceptance criteria
    "Output spiegabile"): mai un numero senza il procedimento che lo
    giustifica."""
    parts = [
        f"Schedina a {len(legs)} eventi (profilo {profile.name}, rischio {profile.risk_label}).",
        f"Quota combinata {combined_odd:.2f} (prodotto delle {len(legs)} quote singole).",
    ]
    if adjusted is None or naive is None:
        parts.append("Probabilita' non disponibile (almeno una pick senza probabilita' modello).")
    elif penalty_count > 0:
        parts.append(
            f"Probabilita' stimata {adjusted:.3f}, corretta rispetto al prodotto semplice {naive:.3f} "
            f"per {penalty_count} coppia/e di pick correlate nella stessa partita (mai trattate come indipendenti)."
        )
    else:
        parts.append(f"Probabilita' stimata {adjusted:.3f} (prodotto delle probabilita' modello, eventi indipendenti).")
    if combined_ev is not None:
        parts.append(f"EV combinato {combined_ev:+.3f} (probabilita' corretta x quota combinata - 1).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Generazione.
# ---------------------------------------------------------------------------


def generate_betslips(
    candidates: list[CandidatePick],
    profiles: tuple[SlipProfile, ...] = DEFAULT_SLIP_PROFILES,
    ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
    max_pool_size: int = 14,
    max_slips_per_profile: int = 5,
    generated_at: Optional[datetime] = None,
    decision_policy: SlipDecisionPolicy = DEFAULT_SLIP_DECISION_POLICY,
    one_pick_per_fixture: bool = True,
    allowed_decisions: frozenset[str] = frozenset({"PLAY"}),
    diversification_policy: BetslipDiversificationPolicy = DEFAULT_DIVERSIFICATION_POLICY,
) -> BetslipGenerationResult:
    """Funzione pura (nessun DB/IO): per ciascun profilo, combina le pick
    eligibili in schedine da `profilo.min_legs` a `profilo.max_legs` eventi,
    scarta SEMPRE le combinazioni con almeno una coppia `EXCLUDE`
    (contraddizione logica same-match, SLIP-02), applica i limiti di
    correlazione/quota/probabilita' del profilo, poi ordina (ranking
    probability/EV/risk) e tronca a `max_slips_per_profile`.
    """
    base_pool = _base_eligible_pool(candidates, allowed_decisions)
    warnings: list[str] = []
    profiles_result: dict[str, list[GeneratedSlip]] = {}

    for profile in profiles:
        pool, pool_warnings = _profile_eligible_pool(
            base_pool,
            profile,
            max_pool_size,
            diversification_policy,
        )
        warnings.extend(pool_warnings)

        generated: list[GeneratedSlip] = []
        for size in range(profile.min_legs, profile.max_legs + 1):
            if size > len(pool):
                warnings.append(f"not_enough_picks:{profile.name}:size_{size}_pool_{len(pool)}")
                continue
            for combo in combinations(pool, size):
                legs = list(combo)
                if one_pick_per_fixture and len({leg.fixture_id for leg in legs}) != len(legs):
                    continue
                family_counts: dict[str, int] = {}
                for leg in legs:
                    family = _market_family(leg.market)
                    family_counts[family] = family_counts.get(family, 0) + 1
                if any(
                    count > diversification_policy.max_legs_for_family(profile.name)
                    for count in family_counts.values()
                ):
                    continue
                evaluation = evaluate_combination(legs, ruleset=ruleset)
                if not evaluation.is_valid:
                    continue  # almeno una coppia EXCLUDE: schedina logicamente impossibile, vietata per QUALSIASI profilo

                penalty_count = sum(1 for f in evaluation.findings if f.severity == PENALTY)
                if penalty_count > profile.max_penalty_pairs:
                    continue

                combined_odd = math.prod(float(leg.odd) for leg in legs)
                if profile.min_combined_odd is not None and combined_odd < profile.min_combined_odd:
                    continue
                if profile.max_combined_odd is not None and combined_odd > profile.max_combined_odd:
                    continue

                adjusted = evaluation.adjusted_probability
                if profile.min_adjusted_probability is not None and (
                    adjusted is None or adjusted < profile.min_adjusted_probability
                ):
                    continue

                combined_ev = (adjusted * combined_odd - 1.0) if adjusted is not None else None
                if profile.min_combined_ev is not None and (combined_ev is None or combined_ev < profile.min_combined_ev):
                    continue

                risk_score = (1.0 - adjusted) if adjusted is not None else None
                metrics = _combined_value_metrics(combined_odd, adjusted, decision_policy)
                situation, situation_reason = _classify_slip(legs, metrics, evaluation.is_valid)
                generated.append(
                    GeneratedSlip(
                        slip_id=_slip_id(profile, legs, decision_policy, diversification_policy),
                        profile_name=profile.name,
                        profile_version=profile.version,
                        risk_label=profile.risk_label,
                        n_legs=len(legs),
                        legs=legs,
                        combined_odd=combined_odd,
                        naive_probability=evaluation.naive_probability,
                        adjusted_probability=adjusted,
                        combined_ev=combined_ev,
                        combined_model_void_odd=metrics["model_void_odd"],
                        combined_edge_absolute=metrics["edge_absolute"],
                        combined_edge_percent=metrics["edge_percent"],
                        combined_expected_roi=metrics["expected_roi"],
                        combined_expected_roi_percent=metrics["expected_roi_percent"],
                        combined_play_threshold=metrics["play_threshold"],
                        slip_min_edge_percent=decision_policy.min_edge_percent,
                        decision_policy_version=decision_policy.version,
                        situation=situation,
                        situation_reason=situation_reason,
                        risk_score=risk_score,
                        penalty_pairs=penalty_count,
                        correlation_ruleset_version=evaluation.ruleset_version,
                        diversification_policy_version=diversification_policy.version,
                        findings=evaluation.findings,
                        explanation=_explanation(
                            profile, legs, combined_odd, evaluation.naive_probability, adjusted, combined_ev, penalty_count
                        ),
                    )
                )

        # Ranking (acceptance criteria "Ranking probability/EV/risk"): EV
        # decrescente, poi probabilita' aggiustata decrescente, poi rischio
        # crescente, poi slip_id per rompere i pareggi in modo deterministico.
        situation_rank = {"PLAY": 0, "BORDERLINE": 1, "NO BET": 2}
        generated.sort(
            key=lambda s: (
                situation_rank.get(s.situation, 99),
                -_ev_sort_key(s.combined_ev),
                -_ev_sort_key(s.adjusted_probability),
                s.risk_score if s.risk_score is not None else float("inf"),
                s.penalty_pairs,
                s.slip_id,
            )
        )
        selected: list[GeneratedSlip] = []
        for candidate in generated:
            candidate_picks = {
                (leg.fixture_id, leg.market, leg.outcome)
                for leg in candidate.legs
            }
            too_similar = False
            for previous in selected:
                previous_picks = {
                    (leg.fixture_id, leg.market, leg.outcome)
                    for leg in previous.legs
                }
                overlap = len(candidate_picks & previous_picks) / min(
                    len(candidate_picks),
                    len(previous_picks),
                )
                if overlap > diversification_policy.max_overlap_ratio:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(candidate)
            if len(selected) >= max_slips_per_profile:
                break
        profiles_result[profile.name] = selected

    generated_at = generated_at or datetime.now(timezone.utc)
    return BetslipGenerationResult(
        generated_at=generated_at.isoformat(),
        correlation_ruleset_version=ruleset.version,
        pool_considered=len(base_pool),
        profiles=profiles_result,
        warnings=warnings,
    )



