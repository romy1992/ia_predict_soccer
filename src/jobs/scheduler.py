from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import selectinload

from src.data.live.live_sync_job import run_manual_live_sync
from src.data.quality_report_service import DataQualityService
from src.jobs.job_history import JobHistory
from src.jobs.job_lock import LOCK_IMPORT_MATCH, job_lock
from src.jobs.job_settings import JOB_DEFINITIONS, is_job_enabled, resolve_job_schedule
from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService
from src.oracle.betslip.betslip_service import BetslipService
from src.oracle.betslip.official_betslip_service import OfficialBetslipService
from src.oracle.betslip.proposal_snapshot_service import BetslipProposalSnapshotService
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.official_capture_service import OfficialPredictionCaptureService
from src.repository.base.repository_db import SessionLocal
from src.service_ia.config.app_config import AppConfig, load_app_config
from src.service_ia.model.match import Match
from src.service_ia.pre_processing.download_match_service import calculate_mean, download_import_matches
from src.service_ia.pre_processing.settlement_service import SettlementService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import train_all_markets

logging.basicConfig(level=logging.INFO)


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _esito_job_saltato(history: JobHistory, job_id: str, start: float, nome_lock: str = LOCK_IMPORT_MATCH) -> dict:
    """Chiude nello storico un job che NON e' partito perche' un altro
    processo stava gia' facendo lo stesso lavoro (vedi `src/jobs/job_lock.py`).

    Viene registrato come `success` e non come `failed`: non e' andato storto
    niente, semplicemente non c'era nulla da fare. Marcarlo `failed`
    sporcherebbe lo storico di errori finti e - peggio - `_is_job_due` legge
    proprio da qui per decidere quando rieseguire, quindi un finto fallimento
    resterebbe indistinguibile da un problema vero. Il flag `skipped_locked`
    nel summary lascia comunque la traccia esplicita del motivo."""
    summary = {
        "skipped_locked": True,
        "motivo": (
            "Un altro processo (container api/scheduler) stava gia' eseguendo un import "
            "sulle stesse tabelle: esecuzione saltata per non duplicare chiamate API-Sports "
            "e scritture a DB."
        ),
        "duration_seconds": time.perf_counter() - start,
    }
    history.mark_success(job_id=job_id, summary=summary)
    logging.warning("Job %s saltato: lock '%s' già preso.", job_id, nome_lock)
    return {"job_id": job_id, **summary}


def run_manual_import(
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    fixture_date: Optional[str] = None,
    statuses: Optional[str] = None,
    days_ahead: Optional[int] = None,
    is_next: bool = False,
    job_id: Optional[str] = None,
    job_type: str = "import",
) -> dict:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    leagues = leagues or cfg.leagues
    history = JobHistory()
    params = {
        "seasons": seasons,
        "leagues": leagues,
        "from_date": from_date,
        "to_date": to_date,
        "fixture_date": fixture_date,
        "statuses": statuses,
        "days_ahead": days_ahead,
        "is_next": is_next,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type=job_type, status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    with job_lock(LOCK_IMPORT_MATCH, obbligatorio=False) as lock_preso:
        if not lock_preso:
            return _esito_job_saltato(history=history, job_id=job_id, start=start)
        try:
            report = download_import_matches(
                seasons=seasons,
                leagues=leagues,
                is_next=is_next,
                from_date=from_date,
                to_date=to_date,
                fixture_date=fixture_date,
                statuses=statuses,
                days_ahead=days_ahead,
            )
            for season in seasons:
                calculate_mean(with_season=season)
            summary = {
                "report": report,
                "duration_seconds": time.perf_counter() - start,
            }
            history.mark_success(job_id=job_id, summary=summary)
            report["job_id"] = job_id
            return report
        except Exception as exc:
            history.mark_failed(
                job_id=job_id,
                error={
                    "message": str(exc),
                    "duration_seconds": time.perf_counter() - start,
                    "params": params,
                },
            )
            raise


def run_manual_retrain(
    markets: Optional[list[str]] = None,
    seasons: Optional[list[int]] = None,
    selection_method: str = "kbest",
    job_id: Optional[str] = None,
) -> dict:
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    history = JobHistory()
    params = {
        "markets": markets,
        "seasons": seasons,
        "selection_method": selection_method,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="retrain", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        results = train_all_markets(
            markets=markets,
            seasons=seasons,
            selection_method=selection_method,
            save_model=True,
        )
        summary = {
            "markets": markets,
            "seasons": seasons,
            "selection_method": selection_method,
            "results": [r.__dict__ for r in results],
            "duration_seconds": time.perf_counter() - start,
        }
        history.mark_success(job_id=job_id, summary=summary)
        return {"job_id": job_id, **summary}
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def run_daily_pipeline() -> None:
    """DEPRECATO (OPS-01): incatenava update+settlement+retrain in un unico
    job, causando un retrain automatico ad OGNI ciclo di import - esattamente
    l'anti-pattern che OPS-01 rimuove dallo scheduler (vedi `build_scheduler`).
    Non piu' registrata come job; lasciata SOLO per compatibilita' di eventuali
    chiamate manuali/script esterni gia' esistenti (mai una funzione rimossa
    silenziosamente se ancora invocabile)."""
    logging.warning(
        "run_daily_pipeline e' deprecata (OPS-01): usa i job separati "
        "(data_sync_today/data_settlement/data_future_sync/ml_training) "
        "registrati da build_scheduler()."
    )
    run_manual_today_update()
    run_manual_settlement()
    run_manual_retrain()


def run_manual_today_update(
    target_date: Optional[str] = None,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Sincronizza fixture odierne (NS/live/final) con persistenza idempotente."""
    date_value = target_date or _today_iso()
    return run_manual_import(
        seasons=seasons,
        leagues=leagues,
        fixture_date=date_value,
        statuses="NS-1H-HT-2H-FT-AET-PEN-ABD",
        is_next=False,
        job_id=job_id,
        job_type="today_update",
    )


def run_manual_future_sync(
    days_ahead: int = 7,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Importa fixture future in una finestra configurabile, mantenendo solo status NS."""
    if days_ahead < 1:
        raise ValueError("days_ahead deve essere >= 1")

    from_day = datetime.now(timezone.utc).date().isoformat()
    to_day = (datetime.now(timezone.utc).date() + timedelta(days=days_ahead)).isoformat()
    return run_manual_import(
        seasons=seasons,
        leagues=leagues,
        from_date=from_day,
        to_date=to_day,
        statuses="NS",
        days_ahead=days_ahead,
        is_next=True,
        job_id=job_id,
        job_type="future_sync",
    )


def run_daily_refresh(
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    days_ahead: int = 7,
    job_id: Optional[str] = None,
) -> dict:
    """Azione combinata per il bottone "Aggiorna tutto" del Data Center:

    1) importa i risultati (con statistiche+quote) delle partite giocate
       IERI per TUTTI i campionati censiti (`cfg.leagues` se non filtrati
       esplicitamente da `leagues`);
    2) sincronizza il calendario delle partite PROSSIME in una finestra di
       `days_ahead` giorni (default 7), che si arricchira' via via di quote
       man mano che le partite si avvicinano (le quote dell'API sports
       durano solo una settimana - vedi `download_match_service.py`).

    Le due sotto-fasi restano loggate anche singolarmente in
    `best_models/jobs_history.jsonl` (job_type `daily_refresh_played` /
    `future_sync`, chiamando le funzioni manuali gia' esistenti) cosi' da
    non perdere granularita' di debug in caso di fallimento parziale, oltre
    alla riga "ombrello" `daily_refresh` con il riepilogo di entrambe.
    """
    cfg = load_app_config()
    seasons = seasons or cfg.seasons
    leagues = leagues or cfg.leagues
    if days_ahead < 1:
        raise ValueError("days_ahead deve essere >= 1")

    history = JobHistory()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    params = {
        "seasons": seasons,
        "leagues": leagues,
        "days_ahead": days_ahead,
        "played_date": yesterday,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="daily_refresh", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    # Il lock e' preso anche QUI, oltre che dentro le due sotto-fasi (che
    # passano entrambe da `run_manual_import`): `job_lock` e' rientrante
    # nello stesso thread, quindi le sotto-fasi lo ri-acquisiscono senza
    # bloccarsi, ma nessun altro processo puo' piu' infilarsi NEL MEZZO tra
    # la fase "ieri" e la fase "prossimi giorni" - che e' esattamente la
    # finestra in cui il 2026-09-15 due giri si sono sovrapposti.
    with job_lock(LOCK_IMPORT_MATCH, obbligatorio=False) as lock_preso:
        if not lock_preso:
            return _esito_job_saltato(history=history, job_id=job_id, start=start)
        try:
            played_report = run_manual_import(
                seasons=seasons,
                leagues=leagues,
                fixture_date=yesterday,
                statuses="FT-AET-PEN-ABD",
                is_next=False,
                job_type="daily_refresh_played",
            )
            upcoming_report = run_manual_future_sync(
                days_ahead=days_ahead,
                seasons=seasons,
                leagues=leagues,
            )
            summary = {
                "played_date": yesterday,
                "played_matches": played_report,
                "upcoming_matches": upcoming_report,
                "duration_seconds": time.perf_counter() - start,
            }
            history.mark_success(job_id=job_id, summary=summary)
            return {"job_id": job_id, **summary}
        except Exception as exc:
            history.mark_failed(
                job_id=job_id,
                error={
                    "message": str(exc),
                    "duration_seconds": time.perf_counter() - start,
                    "params": params,
                },
            )
            raise


def run_manual_settlement(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Riconcilia partite finali e marca completezza dati per settlement prediction."""
    history = JobHistory()
    params = {
        "from_date": from_date,
        "to_date": to_date,
        "seasons": seasons,
        "leagues": leagues,
    }
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(job_type="settlement", status="running", params=params, started_at=JobHistory._now_iso())
        job_id = started["job_id"]

    start = time.perf_counter()
    service = SettlementService()

    # `run_settlement` inizia richiamando `download_import_matches` (vedi
    # `SettlementService.import_runner`), quindi scrive sulle stesse tabelle
    # degli altri import e va sotto lo stesso lock.
    with job_lock(LOCK_IMPORT_MATCH, obbligatorio=False) as lock_preso:
        if not lock_preso:
            return _esito_job_saltato(history=history, job_id=job_id, start=start)
        try:
            report = service.run_settlement(
                from_date=from_date,
                to_date=to_date,
                seasons=seasons,
                leagues=leagues,
            )
            data_phase_duration = time.perf_counter() - start
            ledger_start = time.perf_counter()
            ledger_report = PredictionLedgerService().settle_pending()
            ledger_duration = time.perf_counter() - ledger_start
            betslip_start = time.perf_counter()
            betslip_report = OfficialBetslipService().settle_pending()
            betslip_duration = time.perf_counter() - betslip_start
            shadow_start = time.perf_counter()
            shadow_report = BetslipProposalSnapshotService().settle_pending()
            shadow_duration = time.perf_counter() - shadow_start
            report["matches_updated"] = report.get("updated", 0)
            report["matches_complete"] = report.get("complete", 0)
            report["matches_incomplete"] = report.get("incomplete", 0)
            report.update(ledger_report)
            report.update(betslip_report)
            report["shadow_betslips"] = shadow_report
            report["phase_durations"] = {
                "match_settlement_seconds": data_phase_duration,
                "ledger_settlement_seconds": ledger_duration,
                "betslip_settlement_seconds": betslip_duration,
                "shadow_betslip_settlement_seconds": shadow_duration,
            }
            report["duration_seconds"] = time.perf_counter() - start
            history.mark_success(job_id=job_id, summary=report)
            report["job_id"] = job_id
            return report
        except Exception as exc:
            history.mark_failed(
                job_id=job_id,
                error={
                    "message": str(exc),
                    "duration_seconds": time.perf_counter() - start,
                    "params": params,
                },
            )
            raise


def run_official_prediction_capture(job_id: Optional[str] = None) -> dict:
    """Cattura deterministica delle PLAY ufficiali, indipendente dal frontend."""
    cfg = load_app_config()
    history = JobHistory()
    params = {"cutoff_minutes": cfg.official_capture_minutes_before_kickoff}
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(
            job_type="official_prediction_capture",
            status="running",
            params=params,
            started_at=JobHistory._now_iso(),
        )
        job_id = started["job_id"]
    start = time.perf_counter()
    try:
        report = OfficialPredictionCaptureService().capture(
            cutoff_minutes=cfg.official_capture_minutes_before_kickoff
        )
        report.update(OfficialBetslipService().capture_from_official_ledger())
        report["duration_seconds"] = time.perf_counter() - start
        history.mark_success(job_id=job_id, summary=report)
        return {"job_id": job_id, **report}
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={"message": str(exc), "duration_seconds": time.perf_counter() - start, "params": params},
        )
        raise


def run_manual_refresh_for_next_round() -> None:
    """Compatibilità legacy: alias del nuovo future sync."""
    cfg = load_app_config()
    run_manual_future_sync(days_ahead=7, seasons=cfg.seasons, leagues=cfg.leagues)


def run_data_quality_report(
    top_n: int = 20,
    seasons: Optional[list[int]] = None,
    leagues: Optional[list[int]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Ricalcola il report Data Quality (coverage odds/anomalie/distribuzione
    - vedi `DataQualityService.build_report`) e lo logga come job (job_type
    "data_quality_report"), sia per il bottone "Aggiorna report" della
    pagina Data Quality sia per il job schedulato omonimo (`IntervalTrigger`,
    vedi `build_scheduler`) - stesso principio "una sola funzione, riusata
    da manuale e schedulato" gia' applicato a import/settlement/retrain/
    future_sync. Non chiama alcun provider esterno (SOLO dati gia' a DB),
    quindi mai coinvolto dall'auto-pausa per quota API-Sports esaurita."""
    history = JobHistory()
    params = {"top_n": top_n, "seasons": seasons, "leagues": leagues}
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(
            job_type="data_quality_report", status="running", params=params, started_at=JobHistory._now_iso()
        )
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        report = DataQualityService().build_report(top_n=top_n, seasons=seasons, leagues=leagues)
        report["duration_seconds"] = time.perf_counter() - start
        history.mark_success(job_id=job_id, summary=report)
        report["job_id"] = job_id
        return report
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


# Stesso insieme di `PredictionSnapshotService._FINAL_STATUSES` - duplicato
# qui per lo stesso motivo li' documentato (niente dipendenza a ritroso tra
# package, e scelta esplicita gia' presa in questa sessione di NON
# consolidare le copie sparse nel progetto).
_FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}

# Finestra "recenti" per le partite appena concluse (2026-09-10, punto 2/4
# di `PROMPT_fast_historical_predictions.md`): piccola apposta - lo storico
# gia' passato oltre questa finestra e' compito dello script di backfill
# una tantum (punto 3 dello stesso piano), MAI di questo job ricorrente.
_RECENTLY_FINISHED_WINDOW_DAYS = 3


def run_prediction_snapshot_refresh(
    days_ahead: Optional[int] = None,
    recently_finished_days: Optional[int] = None,
    job_id: Optional[str] = None,
    target_date: Optional[str] = None,
) -> dict:
    """Ricalcola/popola in BACKGROUND la banca dati predizioni
    (`match_prediction_snapshot`) per due categorie di fixture, riusando
    `PredictionSnapshotService` (2026-09-09, richiesto esplicitamente
    dall'operatore: "salvare le predizioni... per le partite di oggi o
    future, solo se cambia una delle feature"):

    1. **NON ANCORA disputate** (status NS) nella finestra oggi ->
       oggi+`days_ahead` giorni (default `cfg.daily_refresh_days_ahead`,
       la STESSA finestra gia' tenuta sincronizzata da
       `data_daily_refresh`/`data_future_sync`) - una riga viene
       RICALCOLATA solo se la fingerprint delle feature o il modello in
       produzione sono cambiati dall'ultimo giro.
    2. **APPENA concluse** (status finale, `_FINAL_STATUSES`) negli ultimi
       `recently_finished_days` giorni (default `_RECENTLY_FINISHED_WINDOW_DAYS`,
       2026-09-10, punto 2/4 di `PROMPT_fast_historical_predictions.md`) -
       CHIUSURA del buco di copertura per le partite che finiscono senza
       mai essere state aperte in Dashboard ne' intercettate mentre erano
       ancora NS: una volta congelata (`PredictionSnapshotService`, regime
       partita conclusa) una riga per fixture+mercato non serve MAI piu'
       essere ricalcolata, quindi qui viene fatto un ANTI-JOIN preventivo
       (`MatchPredictionSnapshotRepository.get_latest_bulk`, una query sola
       per l'intero batch) per scartare le fixture GIA' completamente
       coperte - evita di richiamare `resolve_predictions` (che farebbe
       comunque una query di verifica per mercato) per fixture che non ne
       hanno bisogno. Lo storico OLTRE questa piccola finestra resta scoperto
       da questo job apposta - e' lo scope dello script di backfill una
       tantum (punto 3 dello stesso piano).

    Popola `match_prediction_snapshot` PRIMA che un utente apra la
    Dashboard, cosi' il percorso di serving resta una pura lettura da DB
    (mai un caricamento modello nel path della richiesta - la causa
    principale della lentezza percepita indipendentemente dalla data,
    diagnosticata il 2026-09-09).

    `target_date` (2026-09-13, bottone "Ricalcola previsioni del giorno")
    sostituisce ENTRAMBE le finestre sopra con un solo giorno: tutte le
    fixture di quella data, qualunque sia lo status, invece di "prossimi N
    giorni NS" + "ultimi N giorni conclusi". Nasce dal caso reale di un
    mercato appena promosso a production (Corners/Cards a linea
    configurabile): le fixture gia' a DB restano senza riga per quel
    mercato nuovo finche' il giro schedulato non le ripassa, e l'operatore
    vuole popolare SUBITO il giorno che sta guardando in Dashboard senza
    aspettare. Resta comunque il regime normale di `resolve_predictions`
    (nessun `force`): le partite concluse gia' congelate non vengono
    ricalcolate, i mercati senza riga si'.

    Un fallimento su una SINGOLA fixture non blocca le altre (stesso
    principio "provider errors isolati" gia' applicato in LIVE-01) - finisce
    in `errors`, mai un'eccezione che interrompe l'intero giro."""
    cfg = load_app_config()
    days_ahead = days_ahead if days_ahead is not None else cfg.daily_refresh_days_ahead
    recently_finished_days = (
        recently_finished_days if recently_finished_days is not None else _RECENTLY_FINISHED_WINDOW_DAYS
    )
    if target_date:
        try:
            datetime.fromisoformat(target_date).date()
        except ValueError as exc:
            raise ValueError(f"target_date non valida (atteso YYYY-MM-DD): {target_date}") from exc

    history = JobHistory()
    params = {"days_ahead": days_ahead, "recently_finished_days": recently_finished_days, "target_date": target_date}
    if job_id:
        history.mark_running(job_id=job_id, params=params)
    else:
        started = history.create_job(
            job_type="prediction_snapshot_refresh", status="running", params=params, started_at=JobHistory._now_iso()
        )
        job_id = started["job_id"]

    start = time.perf_counter()
    try:
        today = datetime.now(timezone.utc).date()
        if target_date:
            # Un solo giorno esplicito: la distinzione NS/concluse non serve
            # (le si processano tutte, e' `resolve_predictions` a decidere
            # cosa e' gia' congelato), quindi tutta la giornata finisce nella
            # lista processata incondizionatamente e l'anti-join sulle
            # concluse resta vuoto.
            window_start_iso = target_date
            window_end_iso = (datetime.fromisoformat(target_date).date() + timedelta(days=1)).isoformat()
        else:
            window_start_iso = today.isoformat()
            window_end_iso = (today + timedelta(days=days_ahead + 1)).isoformat()
        finished_window_start_iso = (today - timedelta(days=recently_finished_days)).isoformat()
        finished_window_end_iso = (today + timedelta(days=1)).isoformat()

        try:
            with SessionLocal() as session:
                upcoming_query = (
                    session.query(Match)
                    .options(selectinload(Match.statistics), selectinload(Match.odds))
                    .filter(Match.id_fixture.is_not(None))
                    .filter(Match.date_match >= window_start_iso)
                    .filter(Match.date_match < window_end_iso)
                )
                if not target_date:
                    upcoming_query = upcoming_query.filter(Match.status == "NS")
                upcoming_matches = upcoming_query.all()
                finished_matches = (
                    []
                    if target_date
                    else (
                        session.query(Match)
                        .options(selectinload(Match.statistics), selectinload(Match.odds))
                        .filter(Match.id_fixture.is_not(None))
                        .filter(Match.status.in_(_FINAL_STATUSES))
                        .filter(Match.date_match >= finished_window_start_iso)
                        .filter(Match.date_match < finished_window_end_iso)
                        .all()
                    )
                )
        except (OperationalError, ProgrammingError):
            upcoming_matches = []
            finished_matches = []

        markets = ModelRegistry().list_markets()
        service = PredictionSnapshotService()

        existing_snapshots = service.repo.get_latest_bulk([m.id_fixture for m in finished_matches])
        finished_matches_needing_snapshot = [
            match
            for match in finished_matches
            if any((match.id_fixture, market) not in existing_snapshots for market in markets)
        ]

        fixtures_considered = 0
        predictions_resolved = 0
        errors: list[dict] = []

        for match in [*upcoming_matches, *finished_matches_needing_snapshot]:
            fixtures_considered += 1
            try:
                payload = service.resolve_predictions(
                    fixture_id=match.id_fixture, markets=markets, db_match=match, status=match.status
                )
                predictions_resolved += len(payload)
            except Exception as exc:
                errors.append({"fixture_id": match.id_fixture, "message": str(exc)})

        # Dopo l'aggiornamento delle predizioni salva automaticamente le
        # proposte per ogni giornata futura. Lo snapshot è idempotente:
        # nessuna nuova riga se quote/probabilità/combinazioni sono immutate.
        proposal_report = {
            "dates_considered": 0,
            "proposals_seen": 0,
            "proposals_created": 0,
            "proposals_unchanged": 0,
            "errors": [],
        }
        for proposal_date in sorted(
            {
                datetime.fromisoformat(match.date_match).date()
                for match in upcoming_matches
                if match.date_match
            }
        ):
            # Le proposte esistono solo da oggi in avanti (il generatore
            # rifiuta per policy le giornate passate): con `target_date` su
            # una data storica la lista qui sopra ne conterrebbe una, e
            # tentarla produrrebbe solo un errore atteso nel report.
            if proposal_date < today:
                continue
            proposal_report["dates_considered"] += 1
            try:
                _, _, saved = BetslipService().generate_and_snapshot_for_day(proposal_date)
                for key in ("proposals_seen", "proposals_created", "proposals_unchanged"):
                    proposal_report[key] += saved[key]
            except Exception as exc:
                proposal_report["errors"].append(
                    {"reference_date": proposal_date.isoformat(), "message": str(exc)}
                )

        summary = {
            "days_ahead": days_ahead,
            "recently_finished_days": recently_finished_days,
            "target_date": target_date,
            "fixtures_considered": fixtures_considered,
            "fixtures_upcoming": len(upcoming_matches),
            "fixtures_recently_finished": len(finished_matches_needing_snapshot),
            "predictions_resolved": predictions_resolved,
            "betslip_proposals": proposal_report,
            "errors": errors,
            "duration_seconds": time.perf_counter() - start,
        }
        history.mark_success(job_id=job_id, summary=summary)
        return {"job_id": job_id, **summary}
    except Exception as exc:
        history.mark_failed(
            job_id=job_id,
            error={
                "message": str(exc),
                "duration_seconds": time.perf_counter() - start,
                "params": params,
            },
        )
        raise


def _add_job(
    scheduler: BlockingScheduler,
    func,
    trigger,
    job_id: str,
    misfire_grace_time: int,
) -> None:
    """Registra un job con `max_instances=1`/`coalesce=True` SEMPRE
    espliciti (acceptance criteria "max_instances/coalesce corretti"),
    centralizzati qui una volta sola invece che ripetuti (e potenzialmente
    disallineati) ad ogni chiamata di `add_job` - stesso principio "niente
    duplicazione" gia' seguito nel resto del progetto.

    - `max_instances=1`: mai due esecuzioni sovrapposte dello STESSO job
      (es. un data sync che dura piu' del previsto non ne fa partire un
      secondo in parallelo).
    - `coalesce=True`: se il trigger "perde" piu' esecuzioni per qualunque
      motivo (container fermo, scheduler in pausa), ne recupera SOLO UNA
      al riavvio - mai un burst di esecuzioni accumulate in coda.
    """
    scheduler.add_job(
        func,
        trigger=trigger,
        id=job_id,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=misfire_grace_time,
    )


# Fuso orario SOLO per decidere "e' l'ora X locale?"/"e' passato abbastanza
# tempo?" nel self-gating sotto - MAI usato per i timestamp persistiti
# nello storico job (`JobHistory`, sempre UTC), stesso principio gia'
# applicato in `DashboardService` per la finestra oraria del pool API.
_SCHEDULER_TIMEZONE = ZoneInfo("Europe/Rome")

# Cadenza dell'heartbeat unico che rimpiazza i vecchi trigger per-job
# (`IntervalTrigger`/`CronTrigger` con orario fissato una volta sola
# all'avvio): ad OGNI tick, ogni job si auto-valuta (`_is_job_due`) contro
# lo `schedule` CORRENTE (`resolve_job_schedule`, riletto da disco ad ogni
# chiamata) - un cambio di orario/intervallo da Impostazioni ha quindi
# effetto immediato, senza restart del container `scheduler` (2026-09-09,
# richiesto esplicitamente dall'operatore per "maggiore controllo").
# 30s e' abbastanza fine da non introdurre ritardi percepibili nemmeno sul
# job piu' frequente (`data_sync_live`, minimo 30s per via di
# `_SCHEDULE_FIELD_BOUNDS`), restando comunque leggero (nessuna query DB,
# solo lettura di due JSON piccoli + una lista in memoria).
_HEARTBEAT_SECONDS = 30
_MISFIRE_GRACE_SECONDS = 300


def _last_completed_run(job_history_type: str) -> Optional[dict]:
    """Ultima esecuzione TERMINATA (success o failed) di un job, riusando
    lo storico JSONL gia' esistente (`JobHistory`, la stessa fonte
    mostrata in Data Center/ML Lab) come unica fonte di verita' per "quando
    e' girato l'ultima volta" - nessun nuovo tracking parallelo introdotto
    apposta per lo scheduling."""
    history = JobHistory()
    rows = [row for row in history.tail(limit=200, job_type=job_history_type) if row.get("status") in ("success", "failed")]
    return rows[-1] if rows else None


def _parse_history_timestamp(row: dict) -> datetime:
    raw = row.get("finished_at") or row.get("timestamp") or row.get("started_at")
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_job_due(job_id: str, cfg: AppConfig) -> bool:
    """Vero se, secondo lo `schedule` EFFETTIVO corrente (override salvato
    da Impostazioni se presente, altrimenti default `AppConfig` - vedi
    `resolve_job_schedule`), e' il momento di eseguire `job_id`:

    - `schedule_kind == "daily"`: vero se l'ora locale (Europe/Rome) e'
      gia' oltre `hour:minute` di OGGI, e l'ultima esecuzione riuscita/
      fallita risale a un giorno locale precedente (mai due volte lo
      stesso giorno, anche con molti tick di heartbeat).
    - `schedule_kind in ("interval_minutes", "interval_seconds")`: vero se
      e' trascorso almeno l'intervallo configurato dall'ultima esecuzione
      (confronto in UTC, fuso irrilevante per una durata relativa).

    Nessuna esecuzione precedente in `JobHistory` → sempre dovuto (un job
    appena abilitato/promosso parte al primo tick utile, mai in attesa di
    un giro storico che non esiste)."""
    definition = JOB_DEFINITIONS[job_id]
    schedule_kind = definition["schedule_kind"]
    schedule = resolve_job_schedule(job_id, cfg=cfg)
    last_run = _last_completed_run(definition["job_history_type"])

    if schedule_kind == "daily":
        now_local = datetime.now(_SCHEDULER_TIMEZONE)
        target_today = now_local.replace(hour=schedule["hour"], minute=schedule["minute"], second=0, microsecond=0)
        if now_local < target_today:
            return False
        if last_run is None:
            return True
        last_run_local = _parse_history_timestamp(last_run).astimezone(_SCHEDULER_TIMEZONE)
        return last_run_local.date() < now_local.date()

    if schedule_kind == "interval_minutes":
        interval_seconds = schedule["interval_minutes"] * 60
    elif schedule_kind == "interval_seconds":
        interval_seconds = schedule["interval_seconds"]
    else:
        raise ValueError(f"schedule_kind sconosciuto per '{job_id}': {schedule_kind}")

    if last_run is None:
        return True
    elapsed = (datetime.now(timezone.utc) - _parse_history_timestamp(last_run)).total_seconds()
    return elapsed >= interval_seconds


def _run_if_due(job_id: str, func, *, cfg: AppConfig, **kwargs) -> Optional[dict]:
    """Esegue `func` SOLO se il job e' (1) abilitato in `job_settings.json`
    (pagina Impostazioni) E (2) dovuto secondo lo `schedule` corrente
    (`_is_job_due`) - entrambi controllati ad OGNI tick dell'heartbeat, non
    solo alla registrazione: un toggle o un cambio orario da frontend ha
    quindi effetto immediato, senza richiedere il restart del container
    `scheduler`."""
    if not is_job_enabled(job_id):
        logging.debug("Job '%s' disabilitato da Impostazioni: skip.", job_id)
        return None
    if not _is_job_due(job_id, cfg):
        return None
    logging.info("Job '%s': schedulato ed abilitato, esecuzione in corso.", job_id)
    return func(**kwargs)


def build_scheduler(cfg: Optional[AppConfig] = None) -> BlockingScheduler:
    """Costruisce lo scheduler con TUTTI i job registrati, SENZA avviarlo
    (`.start()` bloccherebbe il processo): separato da `start_scheduler`
    apposta per restare ispezionabile/testabile (`scheduler.get_jobs()`)
    senza dover mockare un loop bloccante.

    OPS-01 (REFACTOR) + orario editabile (2026-09-09): ogni job resta
    COMPLETAMENTE separato e indipendente (nessun retrain automatico
    innescato dal completamento di un data job - acceptance criteria "No
    retrain automatico ad ogni import"), ma NESSUNO ha piu' un trigger
    APScheduler con orario fissato all'avvio (`CronTrigger`/`IntervalTrigger`
    calcolato una volta da `cfg.*`): tutti condividono un unico heartbeat
    (`IntervalTrigger(seconds=_HEARTBEAT_SECONDS)`) e si auto-valutano ad
    ogni tick tramite `_run_if_due`/`_is_job_due` contro lo `schedule`
    EFFETTIVO corrente (`resolve_job_schedule`, che riflette un eventuale
    override salvato da Impostazioni) - un cambio di orario/intervallo da
    frontend ha quindi effetto immediato, senza richiedere il restart del
    container `scheduler` (vedi `job_settings.py::update_job_schedule`).

    - `data_sync_today` (`schedule_kind="interval_minutes"`): sincronizza
      le fixture odierne (NS/live/final) — l'unico job che deve girare
      spesso, per tenere aggiornati punteggi/stati in tempo quasi reale.
    - `data_settlement` (`schedule_kind="interval_minutes"`): riconcilia le
      partite concluse per il settlement (BET-06/dashboard), indipendente
      dal training.
    - `data_quality_report` (`schedule_kind="interval_minutes"`, default 60):
      ricalcola il report Data Quality (coverage/anomalie/distribuzione) e
      lo logga - STESSA funzione (`run_data_quality_report`) invocata dal
      bottone "Aggiorna report" della pagina Data Quality. Nessuna chiamata
      al provider esterno (solo dati gia' a DB).
    - `data_future_sync` (`schedule_kind="daily"`): importa le fixture
      future in una finestra di N giorni — non richiede la frequenza dei
      due job precedenti (le partite future non cambiano stato spesso).
    - `data_daily_refresh` (`schedule_kind="daily"`): STESSO job invocato
      dal bottone "Aggiorna tutto" della Sidebar (sempre visibile, in ogni
      pagina) - chiama `run_daily_refresh` per importare le partite di IERI
      (tutti i campionati censiti) + sincronizzare il calendario prossimo
      (`cfg.daily_refresh_days_ahead` giorni, default 7 — NON editabile da
      Impostazioni, solo orario/intervallo lo sono). Si sovrappone
      volutamente alla finestra futura di `data_future_sync`: se entrambi
      abilitati il calendario prossimo viene risincronizzato due volte al
      giorno (quote piu' fresche, ma doppio consumo quota API-Sports) - chi
      preferisce un solo giro puo' disattivare uno dei due da Impostazioni.
    - `ml_training` (`schedule_kind="daily"`, orario INDIPENDENTE dai data
      job): l'UNICO job che fa retrain — MAI innescato dal completamento di
      un data job, gira col proprio orario/frequenza configurabile
      separatamente (acceptance criteria "No retrain automatico ad ogni
      import").

    LIVE-01 aggiunge un job aggiuntivo, anch'esso indipendente:
    - `data_sync_live` (`schedule_kind="interval_seconds"`): sincronizza il
      dataset LIVE distinto (`src/data/live/`, tabelle `live_*`) - MAI le
      tabelle pre-match `match`/`statistics`/`odds` (nessun impatto sul
      training).

    2026-09-09 aggiunge un ultimo job, indipendente anch'esso:
    - `prediction_snapshot_refresh` (`schedule_kind="interval_minutes"`):
      ricalcola in background le predizioni delle fixture NS nella stessa
      finestra di `data_daily_refresh`/`data_future_sync` (solo dove la
      fingerprint delle feature o il modello sono cambiati dall'ultimo
      giro), E (2026-09-10, punto 2/4 di
      `PROMPT_fast_historical_predictions.md`) chiude in automatico il
      buco di copertura per le fixture APPENA concluse (finestra piccola,
      `_RECENTLY_FINISHED_WINDOW_DAYS` giorni) che non hanno ancora nessuna
      riga salvata - riusando `PredictionSnapshotService` (vedi
      `run_prediction_snapshot_refresh`). Non chiama alcun provider
      esterno (solo dati gia' a DB + inferenza ML locale).
    """
    cfg = cfg or load_app_config()
    scheduler = BlockingScheduler(timezone="Europe/Rome")
    heartbeat = IntervalTrigger(seconds=_HEARTBEAT_SECONDS)

    # --- Data jobs (frequenti, indipendenti dal training) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "data_sync_today", run_manual_today_update, cfg=cfg),
        trigger=heartbeat,
        job_id="data_sync_today",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "data_settlement", run_manual_settlement, cfg=cfg),
        trigger=heartbeat,
        job_id="data_settlement",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "data_quality_report", run_data_quality_report, cfg=cfg),
        trigger=heartbeat,
        job_id="data_quality_report",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "data_future_sync", run_manual_future_sync, cfg=cfg),
        trigger=heartbeat,
        job_id="data_future_sync",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _add_job(
        scheduler,
        functools.partial(
            _run_if_due,
            "data_daily_refresh",
            run_daily_refresh,
            cfg=cfg,
            days_ahead=cfg.daily_refresh_days_ahead,
        ),
        trigger=heartbeat,
        job_id="data_daily_refresh",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )

    # --- Training job (indipendente, orario SEPARATO dai data job) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "ml_training", run_manual_retrain, cfg=cfg),
        trigger=heartbeat,
        job_id="ml_training",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )

    # --- Live job (LIVE-01, dataset SEPARATO, polling frequente in secondi) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "data_sync_live", run_manual_live_sync, cfg=cfg),
        trigger=heartbeat,
        job_id="data_sync_live",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )

    # --- Banca dati predizioni (2026-09-09, background, indipendente) ---
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "prediction_snapshot_refresh", run_prediction_snapshot_refresh, cfg=cfg),
        trigger=heartbeat,
        job_id="prediction_snapshot_refresh",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _add_job(
        scheduler,
        functools.partial(_run_if_due, "official_prediction_capture", run_official_prediction_capture, cfg=cfg),
        trigger=heartbeat,
        job_id="official_prediction_capture",
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )

    return scheduler


def start_scheduler() -> None:
    cfg = load_app_config()
    scheduler = build_scheduler(cfg)

    schedule_summary = ", ".join(
        f"{job_id}={resolve_job_schedule(job_id, cfg=cfg)}" for job_id in JOB_DEFINITIONS
    )
    logging.info(
        "Scheduler started: heartbeat ogni %d sec, ogni job si auto-valuta contro il proprio "
        "schedule effettivo (editabile da Impostazioni senza restart) - %s",
        _HEARTBEAT_SECONDS,
        schedule_summary,
    )
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()





