import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE_URL,
  getApiQuota,
  getDashboardAvailableDates,
  getDashboardBundle,
  getDashboardMatchDetail,
  getBettingStatistics,
  getBetslipGenerate,
  getSavedBetslipProposals,
  getOfficialBetslips,
  getOfficialBetslipStatistics,
  getHealth,
  getJob,
  getJobs,
  getJobSettings,
  getMarkets,
  getModelDiagnostics,
  getMonitoringAlerts,
  getMonitoringOverview,
  getOfficialPerformance,
  getPredictions,
  predict,
  recomputeMatchPredictions,
  refreshDayPredictions,
  refreshApiQuota,
  runJobNow,
  saveBetslipGeneration,
  triggerDailyRefresh,
  triggerDataQualityReport,
  triggerFutureSync,
  triggerImport,
  triggerSettlement,
  triggerTodayUpdate,
  triggerRetrain,
  updateJobSchedule,
  updateJobSettings,
  resetJobSchedule,
} from "./api";
import AppRouter from "./features/layout/AppRouter";
import Sidebar from "./features/layout/Sidebar";
import TopFilters from "./features/layout/TopFilters";
import {
  dayPredictionsProgressFromJob,
  jobSnapshotFromStorage,
  persistDayPredictionsJob,
  readDayPredictionsJob,
  writeDayPredictionsJob,
} from "./features/dashboard/dayPredictionsJobStorage";
import MatchDetailPanel from "./features/matches/components/MatchDetailPanel";
import { filterRowByMarket, todayIso } from "./features/shared/formatters";
export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [selectedDate, setSelectedDate] = useState(todayIso());
  const [availableDates, setAvailableDates] = useState([todayIso()]);
  const [searchInput, setSearchInput] = useState("");
  const [searchFilter, setSearchFilter] = useState("");
  const [phaseFilter, setPhaseFilter] = useState("all");
  const [markets, setMarkets] = useState(["all"]);
  const [selectedMarket, setSelectedMarket] = useState("all");
  const [asyncRun, setAsyncRun] = useState(true);
  const [manualFixtureId, setManualFixtureId] = useState("");
  const [manualMarket, setManualMarket] = useState("under_over_2_5");
  const [health, setHealth] = useState({ status: "loading" });
  const [overview, setOverview] = useState(null);
  const [liveData, setLiveData] = useState({ rows: [], returned: 0, total: 0 });
  const [dayData, setDayData] = useState({ rows: [], returned: 0, total: 0, model_markets: [] });
  const [jobsRows, setJobsRows] = useState([]);
  const [predictionRows, setPredictionRows] = useState([]);
  const [qualityReport, setQualityReport] = useState(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState("");
  const [betslipDate, setBetslipDate] = useState(todayIso());
  const [betslipReport, setBetslipReport] = useState(null);
  const [betslipLoading, setBetslipLoading] = useState(false);
  const [betslipError, setBetslipError] = useState("");
  const [bettingStatistics, setBettingStatistics] = useState(null);
  const [bettingStatsDays, setBettingStatsDays] = useState(30);
  const [monitoringMarket, setMonitoringMarket] = useState("all");
  const [monitoringReport, setMonitoringReport] = useState(null);
  const [monitoringAlerts, setMonitoringAlerts] = useState([]);
  const [officialPerformance, setOfficialPerformance] = useState(null);
  const [officialDays, setOfficialDays] = useState(30);
  const [monitoringLoading, setMonitoringLoading] = useState(false);
  const [monitoringError, setMonitoringError] = useState("");
  const [modelDiagnosticsReport, setModelDiagnosticsReport] = useState(null);
  const [modelDiagnosticsLoading, setModelDiagnosticsLoading] = useState(false);
  const [modelDiagnosticsError, setModelDiagnosticsError] = useState("");
  const [predictOutput, setPredictOutput] = useState("");
  const [opsMessage, setOpsMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isFilterLoading, setIsFilterLoading] = useState(false);
  const initialLoadStartedRef = useRef(false);
  const lastFilterQueryRef = useRef(`${todayIso()}|`);
  const [jobSettingsRows, setJobSettingsRows] = useState([]);
  const [jobSettingsLoading, setJobSettingsLoading] = useState(false);
  const [jobSettingsError, setJobSettingsError] = useState("");
  const [jobSettingsSavingId, setJobSettingsSavingId] = useState(null);
  const [jobScheduleSavingId, setJobScheduleSavingId] = useState(null);
  const [jobRunningId, setJobRunningId] = useState(null);
  const [jobRunFeedback, setJobRunFeedback] = useState({});
  // Avanzamento dei job lanciati con "Esegui ora" (2026-09-21): mappa
  // job_id dello scheduler -> riga di `/jobs/{id}`. Piu' job possono girare
  // insieme, quindi e' una mappa e non un singolo id come `jobRunningId`
  // (che resta per il solo stato "sto inviando la POST").
  const [jobRunRows, setJobRunRows] = useState({});
  const [quotaPaused, setQuotaPaused] = useState(false);
  const [quotaPausedSince, setQuotaPausedSince] = useState(null);
  const [apiQuota, setApiQuota] = useState(null);
  const [apiQuotaLoading, setApiQuotaLoading] = useState(false);
  const [apiQuotaError, setApiQuotaError] = useState("");
  const [lastRefresh, setLastRefresh] = useState("");
  const [selectedFixtureId, setSelectedFixtureId] = useState(null);
  const [matchDetail, setMatchDetail] = useState(null);
  const [matchDetailLoading, setMatchDetailLoading] = useState(false);
  const [matchDetailError, setMatchDetailError] = useState("");
  const [recomputingPredictions, setRecomputingPredictions] = useState(false);
  const [recomputePredictionsError, setRecomputePredictionsError] = useState("");
  const [refreshDayPredictionsError, setRefreshDayPredictionsError] = useState("");
  const [dayPredictionsJobId, setDayPredictionsJobId] = useState(() => readDayPredictionsJob()?.jobId || null);
  const [dayPredictionsJob, setDayPredictionsJob] = useState(() => jobSnapshotFromStorage(readDayPredictionsJob()));
  const handledDayPredictionsJobRef = useRef(null);
  const [oracleFixtureId, setOracleFixtureId] = useState(null);
  const [previousPage, setPreviousPage] = useState("dashboard");
  const marketsQuery = useMemo(() => {
    if (selectedMarket === "all") {
      return undefined;
    }
    return [selectedMarket];
  }, [selectedMarket]);
  const safeRows = dayData?.rows || [];
  // Filtro client-side (fase + mercato) sui dati GIA' scaricati da
  // `loadDashboardData` (che ora richiede sempre tutte le fasi/mercati in
  // un colpo solo): cambiare tab in Dashboard e' quindi istantaneo, nessuna
  // nuova richiesta al backend ne' ricalcolo delle predizioni ML.
  const dashboardDayData = useMemo(() => {
    const rows = dayData?.rows || [];
    const phaseRows = phaseFilter === "all" ? rows : rows.filter((row) => row.phase === phaseFilter);
    const marketRows =
      selectedMarket === "all" ? phaseRows : phaseRows.map((row) => filterRowByMarket(row, selectedMarket));
    return { ...dayData, rows: marketRows, returned: marketRows.length };
  }, [dayData, phaseFilter, selectedMarket]);
  const showMatchFilters = ["dashboard", "predictions"].includes(activePage);
  // Segnale globale (non solo pagina Impostazioni) per disabilitare TUTTI i
  // bottoni che richiamano il provider esterno API-Sports quando la quota
  // giornaliera e' al 100% - stessa percentuale mostrata in Impostazioni
  // (`apiQuota.daily_used_percentage`, che sia "live" o "estimated": qui
  // conta cosa e' VISIBILE all'utente, non solo il check autoritativo usato
  // lato backend per l'auto-pausa dei job schedulati). Un click su un
  // bottone disabilitato fallirebbe comunque lato server con
  // "quota_exceeded": disabilitarlo qui evita solo l'attesa inutile.
  const isQuotaExhausted = Boolean(apiQuota?.available && (apiQuota?.daily_used_percentage ?? 0) >= 100);
  const refreshJobs = useCallback(async (options = {}) => {
    const rows = await getJobs(80, options);
    setJobsRows(rows?.rows || []);
    return rows?.rows || [];
  }, []);
  const refreshPredictions = useCallback(async () => {
    const rows = await getPredictions(80);
    setPredictionRows(rows?.rows || []);
    return rows?.rows || [];
  }, []);
  const loadMetaData = useCallback(async () => {
    const [healthData, marketsData, jobsData, predData, datesData, quotaData] = await Promise.all([
      getHealth(),
      getMarkets(),
      getJobs(80),
      getPredictions(80),
      getDashboardAvailableDates(),
      // Fetch quota GLOBALE (non solo mentre la pagina Impostazioni e'
      // aperta): serve a `isQuotaExhausted` sopra per disabilitare i
      // bottoni ovunque. Rilegge solo la cache locale (nessuna chiamata
      // reale consumata) - se fallisce non deve bloccare il resto del
      // caricamento iniziale.
      getApiQuota().catch(() => null),
    ]);
    setHealth({ ...healthData, apiBaseUrl: API_BASE_URL });
    const apiMarkets = marketsData?.markets || [];
    const mergedMarkets = ["all", ...apiMarkets];
    setMarkets(mergedMarkets);
    setSelectedMarket((prev) => (mergedMarkets.includes(prev) ? prev : "all"));
    setManualMarket((prev) => (apiMarkets.includes(prev) ? prev : apiMarkets[0] || "under_over_2_5"));
    setJobsRows(jobsData?.rows || []);
    setPredictionRows(predData?.rows || []);
    const dates = datesData?.dates || [];
    setAvailableDates(dates.length > 0 ? dates : [todayIso()]);
    if (quotaData) {
      setApiQuota(quotaData);
    }
  }, []);
  const loadDashboardData = useCallback(
    async (mode = "full", { forceRefresh = false } = {}) => {
      const queryKey = `${selectedDate}|${searchFilter}`;
      if (mode === "filter" && !forceRefresh && lastFilterQueryRef.current === queryKey) {
        return null;
      }
      lastFilterQueryRef.current = queryKey;
      if (mode === "full") {
        setIsLoading(true);
      } else if (mode === "filter") {
        setIsFilterLoading(true);
      }
      setError("");
      try {
        // NIENTE piu' `phase`/`markets` qui: si scarica SEMPRE il giorno
        // intero con TUTTE le fasi e TUTTI i mercati (una sola volta per
        // data/ricerca) - i tab Fase/Mercato filtrano poi istantaneamente
        // in memoria (vedi `dashboardDayData` sotto), senza rifare la
        // fetch/ricalcolare le predizioni ML ad ogni click sul tab.
        const payload = await getDashboardBundle({
          targetDate: selectedDate,
          limit: 400,
          search: searchFilter || undefined,
          forceRefresh,
        });
        setOverview(payload.overview);
        setLiveData(payload.live);
        setDayData(payload.day);
        setLastRefresh(new Date().toLocaleString("it-IT"));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        if (mode === "full") {
          setIsLoading(false);
        } else if (mode === "filter") {
          setIsFilterLoading(false);
        }
      }
    },
    [selectedDate, searchFilter]
  );
  // Bottone "Forza aggiornamento" (TopFilters): richiama il provider
  // esterno anche per una data storica gia' sincronizzata a DB (skip
  // DB-first bypassato SOLO qui, mai il guard "quota esaurita" - vedi
  // `is_quota_exhausted_today` in `dashboard_service.py`). Disabilitato a
  // quota esaurita come tutti gli altri bottoni che chiamano API-Sports.
  const handleForceRefreshDay = useCallback(() => {
    loadDashboardData("filter", { forceRefresh: true });
  }, [loadDashboardData]);
  const loadDataQuality = useCallback(async ({ topN = 20, seasons, leagues } = {}) => {
    // Bottone "Aggiorna report": passa dal job "data_quality_report" (stessa
    // funzione del job schedulato omonimo, loggato in storico job - vedi
    // POST /jobs/data-quality-report) invece di una semplice GET, cosi'
    // anche l'esecuzione manuale risulta tracciata come le altre pagine
    // operative. Sincrono (async_run: false): il report calcolato torna
    // direttamente in `details`, nessuna seconda chiamata necessaria.
    setQualityLoading(true);
    setQualityError("");
    try {
      const data = await triggerDataQualityReport({ async_run: false, top_n: topN, seasons, leagues });
      const payload = data?.details || null;
      setQualityReport(payload);
      refreshJobs().catch(() => {});
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setQualityError(message);
      throw err;
    } finally {
      setQualityLoading(false);
    }
  }, [refreshJobs]);
  const loadBetslip = useCallback(
    async (overrides = {}) => {
      setBetslipLoading(true);
      setBetslipError("");
      try {
        const { persist = false, ...generationOverrides } = overrides;
        const isPastDate = betslipDate < todayIso();
        let payload;
        if (isPastDate) {
          const saved = await getSavedBetslipProposals({ targetDate: betslipDate });
          const decisionGroups = {
            PLAY: { SAFE: [], BALANCED: [], AGGRESSIVE: [] },
            BORDERLINE: { SAFE: [], BALANCED: [], AGGRESSIVE: [] },
            "NO BET": { SAFE: [], BALANCED: [], AGGRESSIVE: [] },
          };
          (saved.rows || []).forEach((slip) => {
            const profile = slip.profile_name || "BALANCED";
            const decision = slip.situation || "NO BET";
            decisionGroups[decision][profile] = [
              ...(decisionGroups[decision][profile] || []),
              slip,
            ];
          });
          payload = {
            generated_at: null,
            correlation_ruleset_version: null,
            pool_considered: 0,
            profiles: decisionGroups.PLAY,
            decision_groups: decisionGroups,
            warnings: [],
            historical_snapshot: true,
          };
        } else {
          payload = persist
            ? await saveBetslipGeneration({ targetDate: betslipDate, ...generationOverrides })
            : await getBetslipGenerate({ targetDate: betslipDate, ...generationOverrides });
        }
        const [officialResult, statisticsResult, unifiedResult] = await Promise.allSettled([
          getOfficialBetslips({ targetDate: betslipDate }),
          getOfficialBetslipStatistics(),
          getBettingStatistics({ days: bettingStatsDays }),
        ]);
        const official = officialResult.status === "fulfilled" ? officialResult.value : { rows: [] };
        const officialStatistics = statisticsResult.status === "fulfilled" ? statisticsResult.value : { statistics: null };
        const unifiedStatistics = unifiedResult.status === "fulfilled" ? unifiedResult.value : null;
        const enriched = {
          ...payload,
          official_slips: official.rows || [],
          official_statistics: officialStatistics.statistics || null,
        };
        if (unifiedStatistics) setBettingStatistics(unifiedStatistics);
        setBetslipReport(enriched);
        return enriched;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setBetslipError(message);
        throw err;
      } finally {
        setBetslipLoading(false);
      }
    },
    [betslipDate, bettingStatsDays]
  );
  const loadBettingStatistics = useCallback(async (days) => {
    const selectedDays = Number(days || bettingStatsDays);
    setBettingStatsDays(selectedDays);
    const payload = await getBettingStatistics({ days: selectedDays });
    setBettingStatistics(payload);
    return payload;
  }, [bettingStatsDays]);
  const loadMonitoring = useCallback(async () => {
    setMonitoringLoading(true);
    setMonitoringError("");
    try {
      const marketFilter = monitoringMarket === "all" ? undefined : monitoringMarket;
      const [overviewPayload, alertsPayload, officialPayload] = await Promise.all([
        getMonitoringOverview({ market: marketFilter }),
        getMonitoringAlerts({ market: marketFilter }),
        getOfficialPerformance({ market: marketFilter, days: officialDays }),
      ]);
      setMonitoringReport(overviewPayload);
      setMonitoringAlerts(alertsPayload?.alerts || []);
      setOfficialPerformance(officialPayload);
      return overviewPayload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMonitoringError(message);
      throw err;
    } finally {
      setMonitoringLoading(false);
    }
  }, [monitoringMarket, officialDays]);
  const loadModelDiagnostics = useCallback(async ({ forceRefresh = false } = {}) => {
    // GET /models/diagnostics: walk-forward OOF ricalcolato server-side
    // (cache TTL 15 min li' - vedi `ModelDiagnosticsService`), qui solo
    // fetch + stato, nessun calcolo di metriche lato client.
    setModelDiagnosticsLoading(true);
    setModelDiagnosticsError("");
    try {
      const payload = await getModelDiagnostics({ forceRefresh });
      setModelDiagnosticsReport(payload);
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setModelDiagnosticsError(message);
      throw err;
    } finally {
      setModelDiagnosticsLoading(false);
    }
  }, []);
  const loadEverything = useCallback(async (manual = true) => {
    // Ricarica SOLO dati gia' presenti a DB (health/markets/jobs/dashboard) -
    // NESSUNA sync col provider esterno API-Sports qui (per quello vedi
    // `handleRefreshAll`/`run_daily_refresh` sotto). `manual=true` mostra lo
    // stato "Aggiornamento in corso..." sul bottone Sidebar SOLO se
    // esplicitamente richiesto: il caricamento al mount/reload pagina usa
    // invece `isInitialLoading`, che non tocca il bottone.
    if (manual) {
      setIsLoading(true);
    }
    setError("");
    try {
      await loadMetaData();
      await loadDashboardData("silent");
      setLastRefresh(new Date().toLocaleString("it-IT"));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      if (manual) {
        setIsLoading(false);
      }
      setIsInitialLoading(false);
    }
  }, [loadMetaData, loadDashboardData]);
  const [fullRefreshJobId, setFullRefreshJobId] = useState(null);
  const handledFullRefreshJobRef = useRef(null);
  const fullRefreshJobRow = useMemo(
    () => jobsRows.find((row) => row.job_id === fullRefreshJobId) || null,
    [jobsRows, fullRefreshJobId]
  );
  const fullRefreshRunning =
    Boolean(fullRefreshJobId) && (!fullRefreshJobRow || ["queued", "running"].includes(fullRefreshJobRow.status));
  // Bottone "Aggiorna tutto" (Sidebar, principale, sempre visibile): DEVE
  // fare la STESSA cosa che faceva "Aggiorna tutto ora" nel Data Center -
  // import partite di IERI (tutti i campionati censiti) + sync calendario
  // prossimo (default 7 giorni, con upsert sulle partite gia' presenti) -
  // vedi POST /jobs/daily-refresh -> run_daily_refresh. Consolidato QUI
  // (invece che nel solo Data Center) cosi' funziona/e' visibile da
  // qualunque pagina, non solo restando su Data Center.
  const handleRefreshAll = useCallback(async () => {
    setError("");
    setFullRefreshJobId(null);
    try {
      const data = await triggerDailyRefresh({ async_run: true, days_ahead: 7 });
      const jobId = data?.details?.job_id;
      if (jobId) {
        setFullRefreshJobId(jobId);
      }
      await refreshJobs();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    }
  }, [refreshJobs]);
  // Finche' il job non risulta success/failed nello storico, continua a
  // interrogare /jobs/history cosi' spinner e banner sulla Sidebar restano
  // vivi ovunque l'utente navighi (a differenza della vecchia pagina Data
  // Center, la Sidebar resta sempre montata).
  useEffect(() => {
    if (!fullRefreshRunning) {
      return undefined;
    }
    const timer = setInterval(() => {
      refreshJobs().catch(() => {});
    }, 4000);
    return () => clearInterval(timer);
  }, [fullRefreshRunning, refreshJobs]);
  // Al termine del job (success O failed) ricarica UNA SOLA VOLTA i dati
  // dashboard (il ref evita loop se l'effect si riesegue per altri motivi
  // con lo stesso job_id gia' gestito).
  useEffect(() => {
    if (!fullRefreshJobRow || !["success", "failed"].includes(fullRefreshJobRow.status)) {
      return;
    }
    if (handledFullRefreshJobRef.current === fullRefreshJobRow.job_id) {
      return;
    }
    handledFullRefreshJobRef.current = fullRefreshJobRow.job_id;
    loadEverything(false).catch(() => {});
  }, [fullRefreshJobRow, loadEverything]);
  const loadJobSettings = useCallback(async () => {
    setJobSettingsLoading(true);
    setJobSettingsError("");
    try {
      const payload = await getJobSettings();
      setJobSettingsRows(payload?.jobs || []);
      setQuotaPaused(Boolean(payload?.quota_paused));
      setQuotaPausedSince(payload?.quota_paused_since || null);
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setJobSettingsError(message);
      throw err;
    } finally {
      setJobSettingsLoading(false);
    }
  }, []);
  const toggleJobSetting = useCallback(
    async (jobId, enabled) => {
      setJobSettingsSavingId(jobId);
      setJobSettingsError("");
      // Optimistic update: il toggle sembra istantaneo, poi si riallinea
      // con la risposta reale del server (o torna indietro in caso di errore).
      setJobSettingsRows((rows) => rows.map((row) => (row.job_id === jobId ? { ...row, enabled } : row)));
      try {
        const payload = await updateJobSettings({ [jobId]: enabled });
        setJobSettingsRows(payload?.jobs || []);
        setQuotaPaused(Boolean(payload?.quota_paused));
        setQuotaPausedSince(payload?.quota_paused_since || null);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setJobSettingsError(message);
        await loadJobSettings().catch(() => {});
      } finally {
        setJobSettingsSavingId(null);
      }
    },
    [loadJobSettings]
  );
  const saveJobSchedule = useCallback(async (jobId, schedule) => {
    setJobScheduleSavingId(jobId);
    setJobSettingsError("");
    try {
      const payload = await updateJobSchedule(jobId, schedule);
      setJobSettingsRows((rows) =>
        rows.map((row) =>
          row.job_id === jobId
            ? { ...row, schedule: payload.schedule, schedule_is_default: payload.schedule_is_default }
            : row
        )
      );
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setJobSettingsError(message);
      throw err;
    } finally {
      setJobScheduleSavingId(null);
    }
  }, []);
  const resetJobScheduleToDefault = useCallback(async (jobId) => {
    setJobScheduleSavingId(jobId);
    setJobSettingsError("");
    try {
      const payload = await resetJobSchedule(jobId);
      setJobSettingsRows((rows) =>
        rows.map((row) =>
          row.job_id === jobId
            ? { ...row, schedule: payload.schedule, schedule_is_default: payload.schedule_is_default }
            : row
        )
      );
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setJobSettingsError(message);
      throw err;
    } finally {
      setJobScheduleSavingId(null);
    }
  }, []);
  const runJobNowHandler = useCallback(async (jobId) => {
    setJobRunningId(jobId);
    setJobRunFeedback((prev) => ({ ...prev, [jobId]: { status: "running", message: "Job avviato..." } }));
    try {
      const result = await runJobNow(jobId);
      const runId = result?.details?.job_id;
      setJobRunFeedback((prev) => ({
        ...prev,
        [jobId]: { status: "queued", message: result?.message || "Job accodato con successo." },
      }));
      // Senza `job_id` il job non e' pollabile (job sincroni o risposte
      // legacy): resta il messaggio testuale, nessuna barra inventata.
      if (runId) {
        setJobRunRows((prev) => ({ ...prev, [jobId]: { job_id: runId, status: "queued", summary: {} } }));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setJobRunFeedback((prev) => ({ ...prev, [jobId]: { status: "error", message } }));
    } finally {
      setJobRunningId(null);
    }
  }, []);
  // Chiave STABILE dei job da pollare: cambia solo quando un job parte o
  // finisce, non ad ogni aggiornamento di avanzamento. Serve come dipendenza
  // dell'effect sotto: usare direttamente `jobRunRows` creerebbe un ciclo
  // infinito (l'effect scrive `jobRunRows`, che lo farebbe ripartire ad ogni
  // tick, riazzerando l'intervallo e ripollando subito).
  const jobRunActiveKey = useMemo(
    () =>
      Object.entries(jobRunRows)
        .filter(([, row]) => row?.job_id && ["queued", "running"].includes(row.status))
        .map(([jobId, row]) => `${jobId}|${row.job_id}`)
        .sort()
        .join(","),
    [jobRunRows]
  );
  // Polling dei job "Esegui ora" ancora in corso: un solo timer per tutti,
  // si spegne da solo quando nessun job e' piu' queued/running.
  useEffect(() => {
    const attivi = jobRunActiveKey
      ? jobRunActiveKey.split(",").map((voce) => {
        const [jobId, runId] = voce.split("|");
        return [jobId, { job_id: runId }];
      })
      : [];
    if (attivi.length === 0) {
      return undefined;
    }
    let cancelled = false;
    const tick = async () => {
      const esiti = await Promise.all(
        attivi.map(async ([jobId, row]) => {
          try {
            return [jobId, await getJob(row.job_id)];
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return [jobId, { ...row, status: "failed", error: { message } }];
          }
        })
      );
      if (cancelled) {
        return;
      }
      setJobRunRows((prev) => {
        const next = { ...prev };
        for (const [jobId, riga] of esiti) {
          if (riga) {
            next[jobId] = riga;
          }
        }
        return next;
      });
      setJobRunFeedback((prev) => {
        const next = { ...prev };
        for (const [jobId, riga] of esiti) {
          if (riga?.status === "success") {
            next[jobId] = { status: "success", message: "Job completato." };
          } else if (riga?.status === "failed") {
            next[jobId] = {
              status: "error",
              message: riga.error?.message || "Job fallito.",
            };
          }
        }
        return next;
      });
    };
    tick();
    const timer = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobRunActiveKey]);
  const loadApiQuota = useCallback(async () => {
    setApiQuotaLoading(true);
    setApiQuotaError("");
    try {
      const payload = await getApiQuota();
      setApiQuota(payload);
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setApiQuotaError(message);
      throw err;
    } finally {
      setApiQuotaLoading(false);
    }
  }, []);
  const refreshApiQuotaLive = useCallback(async () => {
    // A differenza di loadApiQuota() (rilegge solo la cache locale), questa
    // interroga DAVVERO API-Sports (bottone "Aggiorna" - vedi
    // POST /settings/quota/refresh): l'unica azione di questa pagina che
    // consuma 1 chiamata reale, per questo NON e' nel polling automatico.
    setApiQuotaLoading(true);
    setApiQuotaError("");
    try {
      const payload = await refreshApiQuota();
      setApiQuota(payload);
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setApiQuotaError(message);
      throw err;
    } finally {
      setApiQuotaLoading(false);
    }
  }, []);
  const loadMatchDetail = useCallback(
    async (fixtureId, silent = false) => {
      if (!fixtureId) {
        setMatchDetail(null);
        return;
      }
      if (!silent) {
        setMatchDetailLoading(true);
      }
      setMatchDetailError("");
      try {
        const payload = await getDashboardMatchDetail(fixtureId, {
          withPredictions: true,
          markets: marketsQuery,
        });
        setMatchDetail(payload);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setMatchDetailError(message);
      } finally {
        if (!silent) {
          setMatchDetailLoading(false);
        }
      }
    },
    [marketsQuery]
  );
  const openMatchDetail = useCallback(
    async (fixtureId) => {
      setSelectedFixtureId(fixtureId);
      await loadMatchDetail(fixtureId);
    },
    [loadMatchDetail]
  );
  const recomputePredictions = useCallback(
    async (fixtureId) => {
      if (!fixtureId) {
        return;
      }
      setRecomputingPredictions(true);
      setRecomputePredictionsError("");
      try {
        await recomputeMatchPredictions(fixtureId, { markets: marketsQuery });
        await loadMatchDetail(fixtureId, true);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setRecomputePredictionsError(message);
      } finally {
        setRecomputingPredictions(false);
      }
    },
    [marketsQuery, loadMatchDetail]
  );
  // "Ricalcola previsioni del giorno": job ASINCRONO con barra. Il POST
  // sincrono restava appeso (timeout) e `refreshing` non tornava mai false
  // finche' non si ricaricava la pagina. Ora si accoda, si polla
  // `/jobs/{id}`, e `job_id` sta in localStorage cosi' un refresh riprende
  // la barra fino alla fine.
  const refreshingDayPredictions =
    Boolean(dayPredictionsJobId) &&
    (!dayPredictionsJob || ["queued", "running"].includes(dayPredictionsJob.status));
  const dayPredictionsProgress = dayPredictionsJob ? dayPredictionsProgressFromJob(dayPredictionsJob) : null;
  const refreshDayPredictionsNow = useCallback(async () => {
    if (!selectedDate || refreshingDayPredictions) {
      return;
    }
    setRefreshDayPredictionsError("");
    try {
      const data = await refreshDayPredictions(selectedDate);
      const jobId = data?.details?.job_id;
      if (!jobId) {
        throw new Error("Job di ricalcolo non accodato");
      }
      handledDayPredictionsJobRef.current = null;
      const queuedJob = {
        job_id: jobId,
        status: "queued",
        params: { target_date: selectedDate },
        summary: { target_date: selectedDate, fixtures_total: 0, fixtures_done: 0, percent: 0 },
      };
      persistDayPredictionsJob(queuedJob);
      setDayPredictionsJob(queuedJob);
      setDayPredictionsJobId(jobId);
    } catch (err) {
      writeDayPredictionsJob(null);
      setDayPredictionsJobId(null);
      setDayPredictionsJob(null);
      setRefreshDayPredictionsError(err instanceof Error ? err.message : String(err));
    }
  }, [selectedDate, refreshingDayPredictions]);
  useEffect(() => {
    if (!dayPredictionsJobId) {
      return undefined;
    }
    let cancelled = false;
    const tick = async () => {
      try {
        const row = await getJob(dayPredictionsJobId);
        if (!cancelled) {
          setDayPredictionsJob(row);
          if (row && ["queued", "running"].includes(row.status)) {
            persistDayPredictionsJob(row);
          }
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err);
          if (message.includes("Job non trovato")) {
            writeDayPredictionsJob(null);
            setDayPredictionsJobId(null);
            setDayPredictionsJob(null);
          }
          setRefreshDayPredictionsError(message);
        }
      }
    };
    tick();
    const timer = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [dayPredictionsJobId]);
  useEffect(() => {
    if (!dayPredictionsJob || !["success", "failed"].includes(dayPredictionsJob.status)) {
      return;
    }
    if (handledDayPredictionsJobRef.current === dayPredictionsJob.job_id) {
      return;
    }
    handledDayPredictionsJobRef.current = dayPredictionsJob.job_id;
    writeDayPredictionsJob(null);
    if (dayPredictionsJob.status === "failed") {
      const message = dayPredictionsJob.error?.message || "Ricalcolo previsioni fallito";
      setRefreshDayPredictionsError(message);
    }
    loadDashboardData("full", { forceRefresh: true }).catch(() => {});
    setDayPredictionsJobId(null);
  }, [dayPredictionsJob, loadDashboardData]);
  const openOracleDetail = useCallback(
    (fixtureId) => {
      setPreviousPage((current) => (activePage === "oracle-detail" ? current : activePage));
      setOracleFixtureId(fixtureId);
      setActivePage("oracle-detail");
    },
    [activePage]
  );
  const closeOracleDetail = useCallback(() => {
    setActivePage(previousPage || "dashboard");
  }, [previousPage]);
  useEffect(() => {
    if (initialLoadStartedRef.current) {
      return;
    }
    initialLoadStartedRef.current = true;
    loadEverything(false);
    // Solo al mount, con manual=false: NON deve comparire "Aggiornamento in
    // corso..." sul bottone Sidebar al semplice reload della pagina (quello
    // deve partire SOLO se l'utente clicca esplicitamente - richiesto
    // 2026-09-05). NON dipendere da `loadEverything` (la sua reference
    // cambia ad ogni cambio di selectedDate/searchFilter tramite
    // loadDashboardData) altrimenti ogni cambio data/ricerca rilancerebbe
    // l'intero refresh pesante. phaseFilter/selectedMarket NON toccano piu'
    // `loadDashboardData` (filtrati client-side, vedi `dashboardDayData`).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (activePage !== "dashboard") {
      return undefined;
    }
    const timer = setInterval(() => {
      // Fix (2026-09-07): se la quota e' gia' segnalata esaurita, richiamare
      // l'intera dashboard ogni 60s non serve a nulla (il backend rifiuta
      // comunque le fetch esterne, vedi `is_quota_exhausted_today` in
      // `dashboard_service.py`) - continua SOLO a rileggere `apiQuota`
      // (cache locale, nessuna chiamata reale) per accorgersi del reset
      // appena avviene, senza sprecare un giro di rete/query DB a vuoto.
      if (!isQuotaExhausted) {
        loadDashboardData("silent");
        if (selectedFixtureId) {
          loadMatchDetail(selectedFixtureId, true);
        }
      }
      // Tiene fresco `apiQuota` (e quindi `isQuotaExhausted`) ANCHE quando
      // l'utente non e' sulla pagina Impostazioni - es. se la quota si
      // esaurisce mentre si naviga altrove, i bottoni si disabilitano
      // entro 60s senza dover aprire Impostazioni. Sempre solo cache
      // locale, nessuna chiamata reale consumata da questo polling.
      getApiQuota().then(setApiQuota).catch(() => {});
    }, 60000);
    return () => clearInterval(timer);
  }, [activePage, isQuotaExhausted, loadDashboardData, loadMatchDetail, selectedFixtureId]);
  useEffect(() => {
    // Ricarica dal backend SOLO quando cambiano data o testo di ricerca
    // (esplicito click "Cerca") - NON piu' su phaseFilter/selectedMarket,
    // che ora filtrano istantaneamente in memoria i dati gia' scaricati
    // (vedi `dashboardDayData` sotto): cambiare tab non deve mai piu'
    // rifare un giro di rete che ricalcola le predizioni ML da capo.
    loadDashboardData("filter");
  }, [selectedDate, searchFilter, loadDashboardData]);
  useEffect(() => {
    if (!selectedFixtureId) {
      return;
    }
    loadMatchDetail(selectedFixtureId, true);
  }, [selectedFixtureId, selectedMarket, loadMatchDetail]);
  useEffect(() => {
    if (activePage !== "data-quality") {
      return;
    }
    if (qualityReport) {
      return;
    }
    loadDataQuality({ topN: 20 }).catch(() => {});
  }, [activePage, qualityReport, loadDataQuality]);
  useEffect(() => {
    if (activePage !== "betslip") {
      return;
    }
    if (betslipReport) {
      return;
    }
    loadBetslip().catch(() => {});
  }, [activePage, betslipReport, loadBetslip]);
  useEffect(() => {
    if (activePage !== "monitoring") {
      return;
    }
    loadMonitoring().catch(() => {});
  }, [activePage, monitoringMarket, loadMonitoring]);
  useEffect(() => {
    if (activePage !== "model-diagnostics") {
      return;
    }
    if (modelDiagnosticsReport) {
      return;
    }
    loadModelDiagnostics().catch(() => {});
  }, [activePage, modelDiagnosticsReport, loadModelDiagnostics]);
  useEffect(() => {
    if (activePage !== "settings") {
      return undefined;
    }
    loadJobSettings().catch(() => {});
    loadApiQuota().catch(() => {});
    // Polling leggero SOLO mentre la pagina Impostazioni e' aperta: rilegge
    // SEMPRE la cache locale (loadApiQuota -> GET /settings/quota), MAI un
    // controllo live su API-Sports - nessuna chiamata reale viene consumata
    // da questo timer. Il controllo live (che consuma 1 chiamata reale)
    // parte SOLO dal click esplicito sul bottone "Aggiorna"
    // (refreshApiQuotaLive -> POST /settings/quota/refresh).
    const timer = setInterval(() => {
      loadApiQuota().catch(() => {});
    }, 30000);
    return () => clearInterval(timer);
  }, [activePage, loadJobSettings, loadApiQuota]);
  async function handleImport() {
    try {
      const data = await triggerImport(asyncRun);
      setOpsMessage(JSON.stringify(data, null, 2));
      await refreshJobs();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setOpsMessage(message);
    }
  }
  async function handleHistoricalImport(payload) {
    const data = await triggerImport(payload);
    setOpsMessage(JSON.stringify(data, null, 2));
    return data;
  }
  async function handleTodayUpdate(payload) {
    const data = await triggerTodayUpdate(payload);
    setOpsMessage(JSON.stringify(data, null, 2));
    return data;
  }
  async function handleFutureSync(payload) {
    const data = await triggerFutureSync(payload);
    setOpsMessage(JSON.stringify(data, null, 2));
    return data;
  }
  async function handleSettlement(payload) {
    const data = await triggerSettlement(payload);
    setOpsMessage(JSON.stringify(data, null, 2));
    return data;
  }
  async function handleRetrain() {
    try {
      const data = await triggerRetrain(asyncRun);
      setOpsMessage(JSON.stringify(data, null, 2));
      await refreshJobs();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setOpsMessage(message);
    }
  }
  async function handleManualPredict() {
    if (!manualFixtureId) {
      setPredictOutput("Inserisci fixture id");
      return;
    }
    try {
      const result = await predict(manualMarket, manualFixtureId);
      setPredictOutput(JSON.stringify(result, null, 2));
      await refreshPredictions();
      await loadDashboardData("silent");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setPredictOutput(message);
    }
  }
  const pageProps = {
    dashboard: {
      overview,
      liveData,
      dayData: dashboardDayData,
      onOpenMatch: openMatchDetail,
      onOpenOracleDetail: openOracleDetail,
      selectedFixtureId,
      phaseFilter,
      onChangePhaseFilter: setPhaseFilter,
      markets,
      selectedMarket,
      onChangeSelectedMarket: setSelectedMarket,
      isFilterLoading,
    },
    oracleDetail: {
      fixtureId: oracleFixtureId,
      onBack: closeOracleDetail,
    },
    predictions: {
      markets,
      manualFixtureId,
      onChangeFixtureId: setManualFixtureId,
      manualMarket,
      onChangeManualMarket: setManualMarket,
      onManualPredict: handleManualPredict,
      predictOutput,
      predictionRows,
      onRefreshPredictions: refreshPredictions,
    },
    dataCenter: {
      asyncRun,
      onChangeAsyncRun: setAsyncRun,
      onRunHistoricalImport: handleHistoricalImport,
      onRunTodayUpdate: handleTodayUpdate,
      onRunFutureSync: handleFutureSync,
      onRunSettlement: handleSettlement,
      opsMessage,
      health,
      jobsRows,
      onRefreshJobs: refreshJobs,
      quotaExhausted: isQuotaExhausted,
    },
    mlLab: {
      asyncRun,
      onChangeAsyncRun: setAsyncRun,
      onImport: handleImport,
      onRetrain: handleRetrain,
      opsMessage,
      health,
      jobsRows,
      onRefreshJobs: refreshJobs,
      quotaExhausted: isQuotaExhausted,
    },
    dataQuality: {
      report: qualityReport,
      isLoading: qualityLoading,
      error: qualityError,
      onLoadReport: loadDataQuality,
    },
    betslip: {
      targetDate: betslipDate,
      onChangeTargetDate: (value) => {
        setBetslipDate(value);
        setBetslipReport(null);
      },
      report: betslipReport,
      isLoading: betslipLoading,
      error: betslipError,
      onLoadReport: (overrides = {}) => loadBetslip({ ...overrides, persist: true }),
      dayData,
      bettingStatistics,
      bettingStatsDays,
      onLoadStatistics: loadBettingStatistics,
    },
    monitoring: {
      market: monitoringMarket,
      onChangeMarket: setMonitoringMarket,
      markets,
      report: monitoringReport,
      officialPerformance,
      officialDays,
      onChangeOfficialDays: setOfficialDays,
      alerts: monitoringAlerts,
      isLoading: monitoringLoading,
      error: monitoringError,
      onLoadReport: loadMonitoring,
    },
    modelDiagnostics: {
      report: modelDiagnosticsReport,
      isLoading: modelDiagnosticsLoading,
      error: modelDiagnosticsError,
      onRefresh: () => loadModelDiagnostics({ forceRefresh: true }).catch(() => {}),
    },
    settings: {
      jobs: jobSettingsRows,
      isLoading: jobSettingsLoading,
      error: jobSettingsError,
      savingJobId: jobSettingsSavingId,
      onToggleJob: toggleJobSetting,
      onRefreshJobs: loadJobSettings,
      scheduleSavingJobId: jobScheduleSavingId,
      onSaveSchedule: saveJobSchedule,
      onResetSchedule: resetJobScheduleToDefault,
      runningJobId: jobRunningId,
      runFeedback: jobRunFeedback,
      runRows: jobRunRows,
      onRunJob: runJobNowHandler,
      quota: apiQuota,
      quotaLoading: apiQuotaLoading,
      quotaError: apiQuotaError,
      onRefreshQuota: refreshApiQuotaLive,
      quotaPaused,
      quotaPausedSince,
    },
  };
  return (
    <div className="layout">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        onRefreshAll={handleRefreshAll}
        lastRefresh={lastRefresh}
        healthStatus={health.status}
        isRefreshing={fullRefreshRunning || isLoading}
        refreshJobRow={fullRefreshJobRow}
        quotaExhausted={isQuotaExhausted}
      />

      <main className="content">
        {showMatchFilters && (
          <TopFilters
            selectedDate={selectedDate}
            onChangeDate={setSelectedDate}
            availableDates={availableDates}
            markets={markets}
            selectedMarket={selectedMarket}
            onChangeSelectedMarket={setSelectedMarket}
            searchInput={searchInput}
            onChangeSearchInput={setSearchInput}
            onApplySearch={() => setSearchFilter(searchInput.trim())}
            onForceRefresh={handleForceRefreshDay}
            forceRefreshDisabled={isQuotaExhausted || isFilterLoading || isLoading}
            onRefreshDayPredictions={refreshDayPredictionsNow}
            refreshingDayPredictions={refreshingDayPredictions}
            dayPredictionsProgress={refreshingDayPredictions ? dayPredictionsProgress : null}
            refreshDayPredictionsError={refreshDayPredictionsError}
          />
        )}
        {error && <div className="error-box">Errore: {error}</div>}
        {(isInitialLoading || isLoading) && <div className="info-box">Caricamento dashboard...</div>}
        <AppRouter activePage={activePage} props={pageProps} />
        {showMatchFilters && (
          <MatchDetailPanel
            selectedFixtureId={selectedFixtureId}
            matchDetail={matchDetail}
            matchDetailLoading={matchDetailLoading}
            matchDetailError={matchDetailError}
            onOpenOracleDetail={openOracleDetail}
            onClose={() => {
              setSelectedFixtureId(null);
              setMatchDetail(null);
            }}
            onRecomputePredictions={recomputePredictions}
            recomputingPredictions={recomputingPredictions}
            recomputePredictionsError={recomputePredictionsError}
          />
        )}
      </main>
    </div>
  );
}