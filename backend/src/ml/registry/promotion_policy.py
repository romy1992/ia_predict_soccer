"""OPS-02: Candidate -> Champion -> Production promotion.

Modulo PURO (nessun accesso a disco/DB/registry), stesso pattern a due
livelli gia' consolidato nel progetto: policy versionata + funzioni di
valutazione qui, orchestrazione DB-aware in
`src/service_ia/training/model_registry.py::ModelRegistry`
(`evaluate_promotion`/`promote_with_policy`/`rollback`), stesso principio
di `DecisionPolicy` (BET-04), `PickPoolPolicy` (SLIP-01) e
`CorrelationRuleSet` (SLIP-02).

Due controlli ESPLICITI, mai bypassati silenziosamente:
1. Gate metriche: un candidate e' promuovibile SOLO se le sue metriche
   (quando disponibili) rispettano soglie minime di qualita' versionate.
2. Confronto production/candidate: quando il target e' 'production', il
   candidate deve reggere il confronto con l'attuale modello in
   produzione per lo stesso mercato (mai un semplice "e' piu' recente").

Le funzioni qui esposte producono un VERDETTO (`PromotionEvaluation`), mai
un'esecuzione diretta della promozione: l'ultimo training non diventa mai
automaticamente production (vincolo generale del progetto, gia' garantito
a monte da `SaveLoad`/`train_multi_market.py` che registrano sempre con
`stage='candidate'`) - eseguire la promozione resta un'azione esplicita di
`ModelRegistry.promote_with_policy`.

Schema `metrics` ETEROGENEO tra i mercati (verificato nel codice esistente):
- `train_multi_market.py` (h2h/under_over_2_5/goal_no_goal/corners/cards/dc
  quando lanciati dal training generico): chiavi dirette `log_loss`/
  `brier`/`ece`/`auc`/`sample_size` (assente, usa `rows`).
- `market_1x2.py`/`corners_market.py`/`cards_market.py`: chiavi
  `pre_log_loss`/`post_log_loss`/`pre_brier`/`post_brier` (pre/post
  calibrazione, nessun `ece`/`auc` diretto).
- `stacking.py`/`oracle_calibration.py`/`btts_market.py`/
  `totals_market.py`: `selection_score` (gia' calcolato da
  `champion_probability_score` o dall'equivalente multiclasse).

`_resolve_metric` legge in ordine di preferenza (es. 'post_log_loss' prima
di 'log_loss', la versione calibrata quando esiste) SENZA richiedere che
ogni trainer esponga lo stesso schema esatto - un controllo il cui dato
non e' disponibile viene esplicitamente SALTATO (mai un'assunzione
ottimistica nascosta), tracciato come tale nei `checks`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


def _resolve_metric(metrics: Optional[dict[str, Any]], *candidate_keys: str) -> Optional[float]:
    if not metrics:
        return None
    for key in candidate_keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def derive_comparable_score(metrics: Optional[dict[str, Any]]) -> tuple[Optional[float], str]:
    """Score singolo comparabile tra due run dello STESSO mercato.

    Preferisce `selection_score` (gia' calcolato in fase di training con lo
    stesso criterio ovunque nel progetto, `champion_probability_score` o
    l'equivalente multiclasse) quando disponibile; altrimenti deriva un
    composite PARZIALE dalle sole metriche probabilistiche effettivamente
    presenti (mai un'invenzione: il metodo usato e' sempre dichiarato nel
    secondo valore ritornato — 'selection_score' / 'derived_partial_composite'
    / 'unavailable' — mai silenzioso)."""
    selection_score = _resolve_metric(metrics, "selection_score", "champion_selection_score")
    if selection_score is not None:
        return selection_score, "selection_score"

    logloss = _resolve_metric(metrics, "post_log_loss", "log_loss")
    brier = _resolve_metric(metrics, "post_brier", "brier")
    ece = _resolve_metric(metrics, "post_ece", "ece")

    weighted: list[float] = []
    if logloss is not None:
        weighted.append(0.5 * (1.0 / (1.0 + logloss)))
    if brier is not None:
        weighted.append(0.3 * (1.0 - min(1.0, max(0.0, brier))))
    if ece is not None:
        weighted.append(0.2 * (1.0 - min(1.0, max(0.0, ece))))

    if not weighted:
        return None, "unavailable"
    return float(sum(weighted)), "derived_partial_composite"


@dataclass(frozen=True)
class PromotionGateThresholds:
    """Soglie minime di qualita' VERSIONATE (mai hardcoded inline nel
    codice di orchestrazione). Ogni soglia a `None` = controllo
    disattivato ESPLICITAMENTE (mai un valore magico "disattiva se zero").

    Default scelti per bocciare solo modelli chiaramente peggiori di un
    classificatore banale (log_loss/brier di un coin-flip su classi
    bilanciate sono rispettivamente ~0.693 e 0.25): soglie permissive ma
    non vuote, cosi' il gate e' operativo da subito senza taratura."""

    max_log_loss: Optional[float] = 0.75
    max_brier: Optional[float] = 0.30
    max_ece: Optional[float] = 0.25
    min_auc: Optional[float] = 0.50
    min_sample_size: int = 30
    require_at_least_one_metric: bool = True


@dataclass(frozen=True)
class PromotionPolicy:
    """Policy versionata di promozione. `min_improvement_over_production`:
    margine minimo (sullo score comparabile, vedi `derive_comparable_score`)
    che il candidate deve garantire rispetto all'attuale production per lo
    stesso mercato quando il target e' 'production' — `0.0` significa
    "basta non peggiorare" (candidate_score >= production_score),
    coerente col fatto che il progetto valuta i modelli con metriche di
    probabilita' dove un pareggio numerico non e' un vero peggioramento."""

    version: str = "promotion_policy_v1"
    gate: PromotionGateThresholds = field(default_factory=PromotionGateThresholds)
    min_improvement_over_production: float = 0.0
    allow_promotion_without_production_baseline: bool = True


DEFAULT_PROMOTION_POLICY = PromotionPolicy()


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str
    observed: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class MetricsGateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    method: str
    candidate_score: Optional[float]
    production_score: Optional[float]
    delta: Optional[float]
    passed: bool
    reason: str


@dataclass
class PromotionEvaluation:
    allowed: bool
    to_stage: str
    policy_version: str
    gate: MetricsGateResult
    comparison: Optional[ComparisonResult]
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_metrics_gate(
    metrics: Optional[dict[str, Any]],
    gate: PromotionGateThresholds = PromotionGateThresholds(),
) -> MetricsGateResult:
    """Valuta SOLO le metriche del candidate contro le soglie minime.
    Ogni soglia con valore `None` disattiva il controllo corrispondente;
    ogni metrica assente nel dict la salta con `metric_not_available`
    (mai bloccante da sola) — MA se `require_at_least_one_metric=True` e
    nessuna metrica e' stata valutabile, il gate fallisce esplicitamente
    (mai una promozione "alla cieca" di un run senza metriche)."""
    checks: list[GateCheck] = []
    blocking: list[str] = []
    evaluable = 0

    def _check(name: str, observed: Optional[float], threshold: Optional[float], comparator) -> None:
        nonlocal evaluable
        if threshold is None:
            return
        if observed is None:
            checks.append(GateCheck(name=name, passed=True, detail="metric_not_available", observed=None, threshold=threshold))
            return
        evaluable += 1
        ok = comparator(observed, threshold)
        checks.append(
            GateCheck(
                name=name,
                passed=ok,
                detail="ok" if ok else f"threshold_violation",
                observed=observed,
                threshold=threshold,
            )
        )
        if not ok:
            blocking.append(f"{name}={observed:.4f} viola la soglia {threshold:.4f}")

    _check(
        "log_loss",
        _resolve_metric(metrics, "post_log_loss", "log_loss"),
        gate.max_log_loss,
        lambda observed, threshold: observed <= threshold,
    )
    _check(
        "brier",
        _resolve_metric(metrics, "post_brier", "brier"),
        gate.max_brier,
        lambda observed, threshold: observed <= threshold,
    )
    _check(
        "ece",
        _resolve_metric(metrics, "post_ece", "ece"),
        gate.max_ece,
        lambda observed, threshold: observed <= threshold,
    )
    _check(
        "auc",
        _resolve_metric(metrics, "post_auc", "auc"),
        gate.min_auc,
        lambda observed, threshold: observed >= threshold,
    )

    sample_size = _resolve_metric(metrics, "sample_size", "rows")
    if gate.min_sample_size and gate.min_sample_size > 0:
        if sample_size is None:
            checks.append(
                GateCheck(
                    name="sample_size",
                    passed=False,
                    detail="metric_not_available",
                    observed=None,
                    threshold=float(gate.min_sample_size),
                )
            )
            blocking.append("sample_size non disponibile nelle metriche registrate")
        else:
            evaluable += 1
            ok = sample_size >= gate.min_sample_size
            checks.append(
                GateCheck(
                    name="sample_size",
                    passed=ok,
                    detail="ok" if ok else "threshold_violation",
                    observed=sample_size,
                    threshold=float(gate.min_sample_size),
                )
            )
            if not ok:
                blocking.append(f"sample_size={sample_size:.0f} sotto il minimo {gate.min_sample_size}")

    if gate.require_at_least_one_metric and evaluable == 0:
        blocking.insert(0, "nessuna metrica valutabile per il gate (metrics mancanti o incomplete)")

    return MetricsGateResult(passed=len(blocking) == 0, checks=checks, blocking_reasons=blocking)


def compare_candidate_to_production(
    candidate_metrics: Optional[dict[str, Any]],
    production_metrics: Optional[dict[str, Any]],
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> ComparisonResult:
    """Confronto ESPLICITO candidate vs attuale production dello STESSO
    mercato. Se non esiste ancora una production (bootstrap: primo modello
    mai promosso per quel mercato), il confronto passa o fallisce secondo
    `policy.allow_promotion_without_production_baseline` — mai un
    confronto "inventato" contro un valore neutro."""
    if production_metrics is None:
        allowed = policy.allow_promotion_without_production_baseline
        reason = (
            "nessuna production esistente per questo mercato: prima promozione consentita"
            if allowed
            else "nessuna production esistente e la policy non consente promozione senza baseline"
        )
        return ComparisonResult(
            method="no_production_baseline",
            candidate_score=None,
            production_score=None,
            delta=None,
            passed=allowed,
            reason=reason,
        )

    candidate_score, candidate_method = derive_comparable_score(candidate_metrics)
    production_score, production_method = derive_comparable_score(production_metrics)

    if candidate_score is None or production_score is None:
        return ComparisonResult(
            method="unavailable",
            candidate_score=candidate_score,
            production_score=production_score,
            delta=None,
            passed=False,
            reason="metriche insufficienti per confrontare candidate e production",
        )

    delta = candidate_score - production_score
    passed = delta >= policy.min_improvement_over_production
    method = candidate_method if candidate_method == production_method else f"{candidate_method}_vs_{production_method}"
    reason = (
        "candidate >= production (oltre il margine richiesto dalla policy)"
        if passed
        else "candidate non supera production (margine richiesto dalla policy non raggiunto)"
    )
    return ComparisonResult(
        method=method,
        candidate_score=candidate_score,
        production_score=production_score,
        delta=delta,
        passed=passed,
        reason=reason,
    )


def evaluate_promotion(
    candidate_metrics: Optional[dict[str, Any]],
    production_metrics: Optional[dict[str, Any]],
    to_stage: str,
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> PromotionEvaluation:
    """Verdetto completo: gate metriche SEMPRE valutato; confronto con
    production valutato SOLO quando `to_stage == 'production'` (per
    'champion'/'retired' il confronto con la production corrente non e'
    pertinente — 'champion' e' uno stadio di validazione intermedio,
    'retired' e' terminale)."""
    gate_result = evaluate_metrics_gate(candidate_metrics, policy.gate)
    blocking = list(gate_result.blocking_reasons)

    comparison_result: Optional[ComparisonResult] = None
    if to_stage == "production":
        comparison_result = compare_candidate_to_production(candidate_metrics, production_metrics, policy)
        if not comparison_result.passed:
            blocking.append(f"comparison_failed: {comparison_result.reason}")

    allowed = gate_result.passed and (comparison_result is None or comparison_result.passed)
    return PromotionEvaluation(
        allowed=allowed,
        to_stage=to_stage,
        policy_version=policy.version,
        gate=gate_result,
        comparison=comparison_result,
        blocking_reasons=blocking,
    )

