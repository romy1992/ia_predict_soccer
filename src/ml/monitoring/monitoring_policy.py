"""OPS-03: Monitoring performance modello e data drift — parte "pura".

Modulo PURO (nessun accesso a disco/DB), stesso pattern a due livelli gia'
consolidato nel progetto: policy versionata + funzioni di valutazione qui,
orchestrazione DB-aware in `src/ml/monitoring/monitoring_service.py`, stesso
principio di `PromotionPolicy` (OPS-02), `DecisionPolicy` (BET-04),
`PickPoolPolicy` (SLIP-01) e `CorrelationRuleSet` (SLIP-02).

Quattro segnali diagnostici (acceptance criteria), MAI un'azione automatica
- solo osservabilita':

1. **Prediction volume**: quante prediction sono state salvate (Prediction
   Ledger, BET-06) in una finestra recente - un drop drastico puo' indicare
   un problema a monte (import/scheduler fermo, endpoint non chiamato).
2. **Calibration drift**: differenza tra la calibrazione (ECE/Brier/LogLoss,
   framework ML-05 riusato SENZA ricalcolo qui) su una finestra RECENTE di
   prediction gia' settled vs una finestra BASELINE precedente.
3. **ROI rolling**: performance realizzata (ROI/hit-rate/drawdown, BET-03
   riusato via `compute_backtest_report`) su finestre mobili (es. 7/30/90
   giorni) - ESPLICITAMENTE SOLO diagnostico: la soglia e' disattivata di
   default (`min_roi_rolling=None`) e quando attivata produce al massimo un
   alert di severita' `info`, MAI un impatto sulla Decision Policy (BET-04)
   o sul gate di promozione (OPS-02), che restano invariati e basati solo
   su metriche probabilistiche out-of-sample calcolate in fase di training.
4. **Feature coverage**: quota di feature attese dal modello IN PRODUZIONE
   effettivamente presenti nei frame di prediction piu' recenti. NOTA
   ONESTA: `FilterMarketService.build_prediction_frame` applica gia' un
   `fillna(0)` a monte, quindi un valore "mancante" non e' distinguibile da
   un legittimo `0.0` da qui - la coverage misura percio' la PRESENZA
   STRUTTURALE della colonna (feature strutturalmente sparita, es.
   rinominata dal feature builder) e la quota di fixture per cui e' stato
   possibile costruire un frame, MAI una precisione "valore non-null" che
   non e' osservabile a questo livello (dichiarato esplicitamente nei nomi
   dei campi, mai un'invenzione silenziosa).

Le soglie sono VERSIONATE (mai hardcoded inline nel codice di
orchestrazione): ogni soglia a `None` disattiva ESPLICITAMENTE il controllo
corrispondente (mai un valore magico "disattiva se zero").
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

INFO = "info"
WARNING = "warning"
CRITICAL = "critical"


@dataclass(frozen=True)
class MonitoringThresholds:
    """Soglie diagnostiche VERSIONATE. Ogni soglia a `None` = controllo
    disattivato ESPLICITAMENTE. Default scelti per essere operativi da
    subito senza taratura, ma permissivi (mai un falso allarme su un
    sistema sano appena avviato con pochi dati)."""

    version: str = "monitoring_policy_v1"

    # Prediction volume: minimo di prediction salvate nella finestra
    # "recente" valutata da `evaluate_alerts` (di default 7 giorni, vedi
    # `MonitoringService.alerts_report`).
    min_prediction_volume_recent: Optional[int] = 1

    # Calibration drift: delta assoluto di ECE tra finestra recente e
    # baseline oltre il quale scatta l'alert; richiede un campione minimo
    # per finestra per evitare drift "rumorosi" su pochi dati.
    max_calibration_ece_drift: Optional[float] = 0.15
    min_samples_for_calibration: int = 20

    # ROI rolling - SOLO diagnostico: disattivato di default. Se attivato,
    # genera al massimo un alert `info` (mai bloccante), solo quando il
    # numero di bet e' sufficiente per essere un segnale (non rumore).
    min_roi_rolling: Optional[float] = None
    min_bets_for_roi_alert: int = 10

    # Feature coverage: quota minima di presenza strutturale delle feature
    # attese (vedi nota ONESTA sopra).
    min_feature_coverage_ratio: Optional[float] = 0.90

    # Job falliti di recente (Job History, DATA-08): soglia sul conteggio
    # nella finestra valutata da `MonitoringService._recent_failed_jobs_count`.
    max_recent_failed_jobs: Optional[int] = 0


DEFAULT_MONITORING_THRESHOLDS = MonitoringThresholds()


@dataclass
class MonitoringAlert:
    """Un singolo alert (VERDETTO, mai un'azione automatica): `severity`
    e' puramente informativa per l'operatore/il frontend, non pilota MAI
    alcuna logica di betting/promotion."""

    code: str
    severity: str
    message: str
    metric: Optional[str] = None
    observed: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_calibration_drift(
    recent_metrics: Optional[dict[str, Any]],
    baseline_metrics: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Confronta due dict GIA' calcolati da `compute_probability_metrics`
    (ML-05, mai ricalcolato qui): `None` se uno dei due manca (drift non
    calcolabile - mai un'assunzione ottimistica). Un delta POSITIVO di
    ECE/Brier/LogLoss (`recent - baseline`) indica un PEGGIORAMENTO; un
    delta negativo indica un miglioramento (mai confuso col caso opposto)."""
    if not recent_metrics or not baseline_metrics:
        return None

    def _delta(key: str) -> Optional[float]:
        recent_value = recent_metrics.get(key)
        baseline_value = baseline_metrics.get(key)
        if recent_value is None or baseline_value is None:
            return None
        return float(recent_value) - float(baseline_value)

    return {
        "recent_sample_size": recent_metrics.get("sample_size"),
        "baseline_sample_size": baseline_metrics.get("sample_size"),
        "recent_ece": recent_metrics.get("ece"),
        "baseline_ece": baseline_metrics.get("ece"),
        "ece_drift": _delta("ece"),
        "recent_brier": recent_metrics.get("brier"),
        "baseline_brier": baseline_metrics.get("brier"),
        "brier_drift": _delta("brier"),
        "recent_log_loss": recent_metrics.get("log_loss"),
        "baseline_log_loss": baseline_metrics.get("log_loss"),
        "log_loss_drift": _delta("log_loss"),
    }


def compute_feature_coverage(
    expected_features: list[str],
    column_presence_counts: dict[str, int],
    sampled_frames: int,
    requested_fixtures: int,
) -> dict[str, Any]:
    """Coverage STRUTTURALE delle feature attese (vedi nota ONESTA nel
    docstring del modulo su `fillna(0)`): per ciascuna feature, quota dei
    frame COSTRUITI CON SUCCESSO in cui la colonna e' presente
    (`column_presence_ratio`); `fixture_coverage_ratio` = quota di fixture
    RICHIESTE per cui e' stato possibile costruire un frame."""
    sampled = max(0, int(sampled_frames))
    requested = max(0, int(requested_fixtures))

    per_feature: list[dict[str, Any]] = []
    fully_missing: list[str] = []
    for feature in expected_features:
        count = int(column_presence_counts.get(feature, 0))
        ratio = (count / sampled) if sampled > 0 else None
        per_feature.append({"feature": feature, "present_count": count, "column_presence_ratio": ratio})
        if sampled > 0 and count == 0:
            fully_missing.append(feature)

    ratios = [row["column_presence_ratio"] for row in per_feature if row["column_presence_ratio"] is not None]
    overall_ratio = (sum(ratios) / len(ratios)) if ratios else None

    return {
        "requested_fixtures": requested,
        "sampled_frames": sampled,
        "fixture_coverage_ratio": (sampled / requested) if requested > 0 else None,
        "expected_feature_count": len(expected_features),
        "overall_column_presence_ratio": overall_ratio,
        "fully_missing_features": fully_missing,
        "per_feature": per_feature,
    }


def evaluate_alerts(
    *,
    prediction_volume_recent: int,
    calibration_drift: Optional[dict[str, Any]] = None,
    roi_rolling: Optional[dict[str, Any]] = None,
    feature_coverage: Optional[dict[str, Any]] = None,
    recent_failed_jobs: int = 0,
    thresholds: MonitoringThresholds = DEFAULT_MONITORING_THRESHOLDS,
) -> list[MonitoringAlert]:
    """Valuta TUTTI i segnali disponibili e produce un VERDETTO (lista di
    alert), MAI un'eccezione: un segnale non fornito (`None`) viene
    SALTATO esplicitamente (mai un'assunzione ottimistica nascosta)."""
    alerts: list[MonitoringAlert] = []

    if thresholds.min_prediction_volume_recent is not None and prediction_volume_recent < thresholds.min_prediction_volume_recent:
        alerts.append(
            MonitoringAlert(
                code="low_prediction_volume",
                severity=WARNING,
                message=(
                    f"Volume prediction recenti ({prediction_volume_recent}) sotto la soglia minima "
                    f"({thresholds.min_prediction_volume_recent})."
                ),
                metric="prediction_volume_recent",
                observed=float(prediction_volume_recent),
                threshold=float(thresholds.min_prediction_volume_recent),
            )
        )

    if calibration_drift and thresholds.max_calibration_ece_drift is not None:
        ece_drift = calibration_drift.get("ece_drift")
        recent_n = calibration_drift.get("recent_sample_size") or 0
        baseline_n = calibration_drift.get("baseline_sample_size") or 0
        if (
            ece_drift is not None
            and recent_n >= thresholds.min_samples_for_calibration
            and baseline_n >= thresholds.min_samples_for_calibration
            and ece_drift > thresholds.max_calibration_ece_drift
        ):
            alerts.append(
                MonitoringAlert(
                    code="calibration_drift",
                    severity=CRITICAL,
                    message=(
                        f"ECE peggiorato di {ece_drift:.4f} rispetto alla baseline "
                        f"(soglia {thresholds.max_calibration_ece_drift:.4f})."
                    ),
                    metric="ece_drift",
                    observed=float(ece_drift),
                    threshold=float(thresholds.max_calibration_ece_drift),
                )
            )

    if roi_rolling and thresholds.min_roi_rolling is not None:
        roi = roi_rolling.get("roi")
        bets = roi_rolling.get("bets") or 0
        if roi is not None and bets >= thresholds.min_bets_for_roi_alert and roi < thresholds.min_roi_rolling:
            alerts.append(
                MonitoringAlert(
                    code="roi_rolling_below_threshold",
                    severity=INFO,
                    message=(
                        f"ROI rolling ({roi:.4f}) sotto la soglia diagnostica "
                        f"({thresholds.min_roi_rolling:.4f}) - SOLO informativo, nessun impatto sulla "
                        f"Decision Policy."
                    ),
                    metric="roi_rolling",
                    observed=float(roi),
                    threshold=float(thresholds.min_roi_rolling),
                )
            )

    if feature_coverage and thresholds.min_feature_coverage_ratio is not None:
        overall = feature_coverage.get("overall_column_presence_ratio")
        if overall is not None and overall < thresholds.min_feature_coverage_ratio:
            alerts.append(
                MonitoringAlert(
                    code="low_feature_coverage",
                    severity=WARNING,
                    message=(
                        f"Coverage feature medio ({overall:.4f}) sotto la soglia minima "
                        f"({thresholds.min_feature_coverage_ratio:.4f})."
                    ),
                    metric="feature_coverage_ratio",
                    observed=float(overall),
                    threshold=float(thresholds.min_feature_coverage_ratio),
                )
            )
        fully_missing = feature_coverage.get("fully_missing_features") or []
        if fully_missing:
            preview = ", ".join(fully_missing[:5]) + ("..." if len(fully_missing) > 5 else "")
            alerts.append(
                MonitoringAlert(
                    code="feature_fully_missing",
                    severity=CRITICAL,
                    message=f"{len(fully_missing)} feature attese risultano SEMPRE assenti: {preview}",
                    metric="fully_missing_features_count",
                    observed=float(len(fully_missing)),
                    threshold=0.0,
                )
            )

    if thresholds.max_recent_failed_jobs is not None and recent_failed_jobs > thresholds.max_recent_failed_jobs:
        alerts.append(
            MonitoringAlert(
                code="recent_failed_jobs",
                severity=CRITICAL,
                message=f"{recent_failed_jobs} job falliti di recente (soglia {thresholds.max_recent_failed_jobs}).",
                metric="recent_failed_jobs",
                observed=float(recent_failed_jobs),
                threshold=float(thresholds.max_recent_failed_jobs),
            )
        )

    return alerts

