"""OPS-03: Monitoring performance modello e data drift — orchestrazione
DB-aware.

Collega il modulo PURO (`monitoring_policy.py`) ai dati GIA' persistiti da
task precedenti, SENZA duplicare alcuna logica:

- **Prediction Ledger** (BET-06, `PredictionLedgerRepository`) per volume/
  ROI rolling/calibration drift (stessi campi `p_model`/`won`/`created_at`
  gia' validati da BET-06).
- **Betting Backtester** (BET-03, `compute_backtest_report`) per il report
  ROI/hit-rate/drawdown per finestra mobile.
- **Probability Metrics** (ML-05, `compute_probability_metrics`) per la
  calibrazione recent-vs-baseline.
- **Model Registry** (OPS-02) per le `feature_names` del modello IN
  PRODUZIONE (fallback a `latest` se nessuna production esiste ancora,
  STESSO fallback gia' usato da `/predict/{market}`).
- **Prediction Logger** per le fixture piu' recenti su cui il modello ha
  EFFETTIVAMENTE predetto (stesso flusso di `/predict/{market}`), usate
  come campione per la feature coverage.
- **Job History** (DATA-08) per i job falliti di recente.

Nessuna nuova tabella/migration: tutti i dati sono gia' persistiti da task
precedenti (BET-06/OPS-02/DATA-08). L'unica estensione a codice esistente e'
un parametro opzionale `since` su `PredictionLedgerRepository.list_all`
(retro-compatibile, default `None` = comportamento invariato) per filtrare
per finestra temporale lato query invece di caricare l'intera tabella.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Optional

from src.ml.evaluation.probability_metrics import compute_probability_metrics
from src.ml.monitoring.monitoring_policy import (
    DEFAULT_MONITORING_THRESHOLDS,
    MonitoringThresholds,
    compute_calibration_drift,
    compute_feature_coverage,
    evaluate_alerts,
)
from src.oracle.backtest.betting_backtester import (
    DEFAULT_PLACED_DECISIONS,
    STAKE_DEFAULT,
    BacktestBet,
    compute_backtest_report,
)
from src.jobs.job_history import JobHistory
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.prediction_logger import PredictionLogger

DEFAULT_ROI_WINDOWS_DAYS: tuple[int, ...] = (7, 30, 90)
DEFAULT_CALIBRATION_RECENT_DAYS = 30
DEFAULT_PREDICTION_VOLUME_DAYS = 30
DEFAULT_ALERT_VOLUME_DAYS = 7
DEFAULT_FEATURE_COVERAGE_SAMPLE = 50
DEFAULT_FAILED_JOBS_WINDOW_DAYS = 7


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _as_aware(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


class MonitoringService:
    def __init__(
        self,
        ledger_repo: Optional[PredictionLedgerRepository] = None,
        job_history: Optional[JobHistory] = None,
        registry: Optional[ModelRegistry] = None,
        prediction_logger: Optional[PredictionLogger] = None,
        filter_service: Optional[FilterMarketService] = None,
    ):
        self.ledger_repo = ledger_repo or PredictionLedgerRepository()
        self.job_history = job_history or JobHistory()
        self.registry = registry or ModelRegistry()
        self.prediction_logger = prediction_logger or PredictionLogger()
        self.filter_service = filter_service or FilterMarketService()

    # ------------------------------------------------------------------
    # 1. Prediction volume
    # ------------------------------------------------------------------
    def prediction_volume_report(
        self, market: Optional[str] = None, days: int = DEFAULT_PREDICTION_VOLUME_DAYS
    ) -> dict[str, Any]:
        """Conteggio prediction (Prediction Ledger, BET-06) salvate negli
        ultimi `days` giorni, in bucket giornalieri (ordine cronologico
        crescente, SEMPRE tutti i giorni della finestra anche a conteggio
        zero - mai un buco silenzioso nella serie)."""
        since = _utc_now() - dt.timedelta(days=days)
        rows = self.ledger_repo.list_all(market=market, since=since, limit=1_000_000)

        buckets: dict[str, int] = {}
        for row in rows:
            created_at = _as_aware(row.created_at)
            if created_at is None:
                continue
            key = created_at.date().isoformat()
            buckets[key] = buckets.get(key, 0) + 1

        today = _utc_now().date()
        series = []
        for offset in range(days - 1, -1, -1):
            day = today - dt.timedelta(days=offset)
            key = day.isoformat()
            series.append({"date": key, "count": buckets.get(key, 0)})

        total = len(rows)
        return {
            "market": market,
            "days": days,
            "total": total,
            "average_per_day": (total / days) if days > 0 else None,
            "series": series,
        }

    # ------------------------------------------------------------------
    # 2. Calibration drift
    # ------------------------------------------------------------------
    def _settled_rows_with_probability(self, market: Optional[str]) -> list[Any]:
        rows = self.ledger_repo.list_all(market=market, is_settled=True, limit=1_000_000)
        # Solo righe realmente confrontabili: VOID (odd/esito mancante,
        # `settle_prediction_record`) esclude gia' `won`, qui filtriamo
        # anche eventuali `p_model` mancanti (mai un ECE su dati parziali).
        return [row for row in rows if row.p_model is not None and row.won is not None]

    def calibration_drift_report(
        self,
        market: str,
        recent_days: int = DEFAULT_CALIBRATION_RECENT_DAYS,
        thresholds: MonitoringThresholds = DEFAULT_MONITORING_THRESHOLDS,
    ) -> Optional[dict[str, Any]]:
        """Calibrazione (ECE/Brier/LogLoss, ML-05 riusato) su prediction
        GIA' settled: finestra RECENTE (ultimi `recent_days` giorni) vs
        BASELINE (tutto il resto dello storico, PRIMA della finestra
        recente). `available=False` (con `reason` esplicito) se non c'e'
        abbastanza storico per un confronto onesto - mai un drift
        calcolato su un campione troppo piccolo. `None` solo se non
        esiste ALCUNA prediction settled con probabilita' per il mercato."""
        all_rows = self._settled_rows_with_probability(market=market)
        if not all_rows:
            return None

        cutoff = _utc_now() - dt.timedelta(days=recent_days)
        recent_rows = [row for row in all_rows if (_as_aware(row.created_at) or cutoff) >= cutoff]
        baseline_rows = [row for row in all_rows if (_as_aware(row.created_at) or cutoff) < cutoff]

        if (
            len(recent_rows) < thresholds.min_samples_for_calibration
            or len(baseline_rows) < thresholds.min_samples_for_calibration
        ):
            return {
                "market": market,
                "recent_days": recent_days,
                "available": False,
                "reason": (
                    f"Campione insufficiente per un confronto affidabile "
                    f"(recent={len(recent_rows)}, baseline={len(baseline_rows)}, minimo richiesto "
                    f"{thresholds.min_samples_for_calibration} per finestra)."
                ),
                "recent_metrics": None,
                "baseline_metrics": None,
                "drift": None,
            }

        recent_metrics = compute_probability_metrics(
            y_true=[1 if row.won else 0 for row in recent_rows],
            probabilities=[float(row.p_model) for row in recent_rows],
        )
        baseline_metrics = compute_probability_metrics(
            y_true=[1 if row.won else 0 for row in baseline_rows],
            probabilities=[float(row.p_model) for row in baseline_rows],
        )
        drift = compute_calibration_drift(recent_metrics=recent_metrics, baseline_metrics=baseline_metrics)

        return {
            "market": market,
            "recent_days": recent_days,
            "available": True,
            "reason": None,
            "recent_metrics": recent_metrics,
            "baseline_metrics": baseline_metrics,
            "drift": drift,
        }

    # ------------------------------------------------------------------
    # 3. ROI rolling (SOLO diagnostico)
    # ------------------------------------------------------------------
    def roi_rolling_report(
        self,
        market: Optional[str] = None,
        windows_days: tuple[int, ...] = DEFAULT_ROI_WINDOWS_DAYS,
        stake: float = STAKE_DEFAULT,
    ) -> dict[str, Any]:
        """ROI/hit-rate/drawdown (BET-03, riusato via `compute_backtest_
        report`) su finestre mobili, ESPLICITAMENTE SOLO diagnostico
        (`diagnostic_only=True`): non deve MAI essere usato per alimentare
        la Decision Policy (BET-04) o il gate di promozione (OPS-02), che
        restano invariati e basati solo su metriche out-of-sample di
        training."""
        windows: dict[str, Any] = {}
        for days in windows_days:
            since = _utc_now() - dt.timedelta(days=days)
            rows = self.ledger_repo.list_all(market=market, is_settled=True, since=since, limit=1_000_000)
            bets = [
                BacktestBet(
                    market=row.market,
                    outcome=row.outcome,
                    p_model=row.p_model,
                    p_market_fair=row.p_market_fair,
                    odd=row.odd,
                    prob_edge=row.prob_edge,
                    ev=row.ev,
                    decision=row.decision,
                    policy_version=row.policy_version or "",
                    won=row.won,
                    fixture_id=row.fixture_id,
                    kickoff_at=row.kickoff_at.isoformat() if row.kickoff_at else None,
                )
                for row in rows
            ]
            report = compute_backtest_report(bets=bets, stake=stake, include_decisions=DEFAULT_PLACED_DECISIONS)
            windows[str(days)] = dataclasses.asdict(report)

        return {"market": market, "diagnostic_only": True, "windows": windows}

    # ------------------------------------------------------------------
    # 4. Feature coverage
    # ------------------------------------------------------------------
    def feature_coverage_report(
        self, market: str, sample_limit: int = DEFAULT_FEATURE_COVERAGE_SAMPLE
    ) -> Optional[dict[str, Any]]:
        """Coverage strutturale delle `feature_names` del modello IN
        PRODUZIONE (o `latest` se nessuna production esiste ancora, STESSO
        fallback di `/predict/{market}`) sui frame di prediction PIU'
        RECENTI (stesso `FilterMarketService.build_prediction_frame` usato
        in produzione - nessuna duplicazione della feature-building
        logic). `None` se nessun modello o nessuna feature registrata per
        il mercato (mai una coverage calcolata su un modello inesistente)."""
        active = self.registry.get_production(market=market) or self.registry.get_latest(market=market)
        if not active:
            return None

        expected_features = active.get("feature_names") or []
        if not expected_features:
            return None

        logged_rows = self.prediction_logger.tail(limit=sample_limit * 3, market=market)
        fixture_ids: list[int] = []
        seen: set[int] = set()
        for row in reversed(logged_rows):  # piu' recenti prima (tail e' cronologico crescente)
            fixture_id = row.get("fixture_id")
            if fixture_id is None or fixture_id in seen:
                continue
            seen.add(fixture_id)
            fixture_ids.append(int(fixture_id))
            if len(fixture_ids) >= sample_limit:
                break

        presence_counts: dict[str, int] = {feature: 0 for feature in expected_features}
        sampled_frames = 0
        for fixture_id in fixture_ids:
            frame = self.filter_service.build_prediction_frame(market=market, fixture_id=fixture_id)
            if frame is None or frame.empty:
                continue
            sampled_frames += 1
            for feature in expected_features:
                if feature in frame.columns:
                    presence_counts[feature] += 1

        coverage = compute_feature_coverage(
            expected_features=expected_features,
            column_presence_counts=presence_counts,
            sampled_frames=sampled_frames,
            requested_fixtures=len(fixture_ids),
        )
        coverage["market"] = market
        coverage["model_run_id"] = active.get("run_id")
        coverage["model_stage"] = active.get("current_stage") or active.get("stage")
        return coverage

    # ------------------------------------------------------------------
    # 5. Alert base
    # ------------------------------------------------------------------
    def _recent_failed_jobs_count(self, since_days: int = DEFAULT_FAILED_JOBS_WINDOW_DAYS) -> int:
        """Job falliti (Job History, DATA-08) con timestamp nella finestra
        recente. Un timestamp mancante/non parsabile viene comunque
        contato (mai un job fallito nascosto per un dettaglio di
        formato)."""
        cutoff = _utc_now() - dt.timedelta(days=since_days)
        failed = self.job_history.tail(limit=500, status="failed")
        count = 0
        for row in failed:
            ts = row.get("timestamp")
            if not ts:
                count += 1
                continue
            try:
                parsed = _as_aware(dt.datetime.fromisoformat(ts))
            except ValueError:
                count += 1
                continue
            if parsed is not None and parsed >= cutoff:
                count += 1
        return count

    def alerts_report(
        self, market: Optional[str] = None, thresholds: MonitoringThresholds = DEFAULT_MONITORING_THRESHOLDS
    ) -> dict[str, Any]:
        """Aggrega tutti i segnali disponibili e li valuta con
        `evaluate_alerts` (modulo puro). Senza `market`, calibration
        drift/feature coverage sono `None` (richiedono un modello/mercato
        specifico - non c'e' un "modello aggregato")."""
        volume = self.prediction_volume_report(market=market, days=DEFAULT_ALERT_VOLUME_DAYS)
        roi = self.roi_rolling_report(market=market, windows_days=(30,))
        roi_30 = (roi.get("windows") or {}).get("30", {}).get("overall")

        calibration = self.calibration_drift_report(market=market) if market else None
        calibration_drift_payload = calibration.get("drift") if calibration and calibration.get("available") else None

        coverage = self.feature_coverage_report(market=market) if market else None

        recent_failed_jobs = self._recent_failed_jobs_count()

        alerts = evaluate_alerts(
            prediction_volume_recent=volume["total"],
            calibration_drift=calibration_drift_payload,
            roi_rolling=roi_30,
            feature_coverage=coverage,
            recent_failed_jobs=recent_failed_jobs,
            thresholds=thresholds,
        )

        return {
            "generated_at": _utc_now().isoformat(),
            "market": market,
            "thresholds_version": thresholds.version,
            "alerts": [alert.to_dict() for alert in alerts],
        }

    # ------------------------------------------------------------------
    # Overview aggregata
    # ------------------------------------------------------------------
    def full_report(
        self, market: Optional[str] = None, thresholds: MonitoringThresholds = DEFAULT_MONITORING_THRESHOLDS
    ) -> dict[str, Any]:
        """Risposta unica per la dashboard di monitoring (acceptance
        criteria "Dashboard/endpoint monitoring disponibile"): tutti e
        quattro i segnali diagnostici piu' l'elenco alert, in un'unica
        chiamata."""
        prediction_volume = self.prediction_volume_report(market=market, days=DEFAULT_PREDICTION_VOLUME_DAYS)
        roi_rolling = self.roi_rolling_report(market=market)
        calibration_drift = self.calibration_drift_report(market=market) if market else None
        feature_coverage = self.feature_coverage_report(market=market) if market else None
        alerts_payload = self.alerts_report(market=market, thresholds=thresholds)

        return {
            "generated_at": _utc_now().isoformat(),
            "market": market,
            "thresholds_version": thresholds.version,
            "prediction_volume": prediction_volume,
            "calibration_drift": calibration_drift,
            "roi_rolling": roi_rolling,
            "feature_coverage": feature_coverage,
            "alerts": alerts_payload["alerts"],
        }

