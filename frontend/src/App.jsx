import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE_URL,
  getApiQuota,
  getDataQuality,
  getDashboardAvailableDates,
  getDashboardDay,
  getDashboardLive,
  getDashboardMatchDetail,
  getDashboardOverview,
  getBetslipGenerate,
  getHealth,
  getJobs,
  getJobSettings,
  getMarkets,
  getMonitoringAlerts,
  getMonitoringOverview,
  getPredictions,
  predict,
  refreshApiQuota,
  triggerDailyRefresh,
  triggerFutureSync,
  triggerImport,
  triggerSettlement,
  triggerTodayUpdate,
  triggerRetrain,
  updateJobSettings,
} from "./api";
import AppRouter from "./features/layout/AppRouter";
import Sidebar from "./features/layout/Sidebar";
import TopFilters from "./features/layout/TopFilters";
import MatchDetailPanel from "./features/matches/components/MatchDetailPanel";
import { todayIso } from "./features/shared/formatters";
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
  const [monitoringMarket, setMonitoringMarket] = useState("all");
  const [monitoringReport, setMonitoringReport] = useState(null);
  const [monitoringAlerts, setMonitoringAlerts] = useState([]);
  const [monitoringLoading, setMonitoringLoading] = useState(false);
  const [monitoringError, setMonitoringError] = useState("");
  const [predictOutput, setPredictOutput] = useState("");
  const [opsMessage, setOpsMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isFilterLoading, setIsFilterLoading] = useState(false);
  const [jobSettingsRows, setJobSettingsRows] = useState([]);
  const [jobSettingsLoading, setJobSettingsLoading] = useState(false);
  const [jobSettingsError, setJobSettingsError] = useState("");
  const [jobSettingsSavingId, setJobSettingsSavingId] = useState(null);
  const [apiQuota, setApiQuota] = useState(null);
  const [apiQuotaLoading, setApiQuotaLoading] = useState(false);
  const [apiQuotaError, setApiQuotaError] = useState("");
  const [lastRefresh, setLastRefresh] = useState("");
  const [selectedFixtureId, setSelectedFixtureId] = useState(null);
  const [matchDetail, setMatchDetail] = useState(null);
  const [matchDetailLoading, setMatchDetailLoading] = useState(false);
  const [matchDetailError, setMatchDetailError] = useState("");
  const [oracleFixtureId, setOracleFixtureId] = useState(null);
  const [previousPage, setPreviousPage] = useState("dashboard");
  const marketsQuery = useMemo(() => {
    if (selectedMarket === "all") {
      return undefined;
    }
    return [selectedMarket];
  }, [selectedMarket]);
  const safeRows = dayData?.rows || [];
  const showMatchFilters = ["dashboard", "predictions"].includes(activePage);
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
    const [healthData, marketsData, jobsData, predData, datesData] = await Promise.all([
      getHealth(),
      getMarkets(),
      getJobs(80),
      getPredictions(80),
      getDashboardAvailableDates(),
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
  }, []);
  const loadDashboardData = useCallback(
    async (mode = "full") => {
      if (mode === "full") {
        setIsLoading(true);
      } else if (mode === "filter") {
        setIsFilterLoading(true);
      }
      setError("");
      try {
        const phase = phaseFilter === "all" ? undefined : phaseFilter;
        const [overviewData, livePayload, dayPayload] = await Promise.all([
          getDashboardOverview(selectedDate),
          getDashboardLive({
            targetDate: selectedDate,
            limit: 30,
            withPredictions: true,
            markets: marketsQuery,
          }),
          getDashboardDay({
            targetDate: selectedDate,
            limit: 400,
            withPredictions: true,
            markets: marketsQuery,
            phase,
            search: searchFilter || undefined,
          }),
        ]);
        setOverview(overviewData);
        setLiveData(livePayload);
        setDayData(dayPayload);
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
    [selectedDate, marketsQuery, phaseFilter, searchFilter]
  );
  const loadDataQuality = useCallback(async ({ topN = 20, seasons, leagues } = {}) => {
    setQualityLoading(true);
    setQualityError("");
    try {
      const payload = await getDataQuality({ topN, seasons, leagues });
      setQualityReport(payload);
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setQualityError(message);
      throw err;
    } finally {
      setQualityLoading(false);
    }
  }, []);
  const loadBetslip = useCallback(
    async (overrides = {}) => {
      setBetslipLoading(true);
      setBetslipError("");
      try {
        const payload = await getBetslipGenerate({ targetDate: betslipDate, ...overrides });
        setBetslipReport(payload);
        return payload;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setBetslipError(message);
        throw err;
      } finally {
        setBetslipLoading(false);
      }
    },
    [betslipDate]
  );
  const loadMonitoring = useCallback(async () => {
    setMonitoringLoading(true);
    setMonitoringError("");
    try {
      const marketFilter = monitoringMarket === "all" ? undefined : monitoringMarket;
      const [overviewPayload, alertsPayload] = await Promise.all([
        getMonitoringOverview({ market: marketFilter }),
        getMonitoringAlerts({ market: marketFilter }),
      ]);
      setMonitoringReport(overviewPayload);
      setMonitoringAlerts(alertsPayload?.alerts || []);
      return overviewPayload;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMonitoringError(message);
      throw err;
    } finally {
      setMonitoringLoading(false);
    }
  }, [monitoringMarket]);
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
    loadEverything(false);
    // Solo al mount, con manual=false: NON deve comparire "Aggiornamento in
    // corso..." sul bottone Sidebar al semplice reload della pagina (quello
    // deve partire SOLO se l'utente clicca esplicitamente - richiesto
    // 2026-09-05). NON dipendere da `loadEverything` (la sua reference
    // cambia ad ogni cambio di phaseFilter/selectedMarket/selectedDate/
    // searchFilter tramite loadDashboardData) altrimenti ogni click su un
    // tab/filtro rilancerebbe l'intero refresh pesante (bug "Aggiornamento
    // in corso" ad ogni cambio tab).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const timer = setInterval(() => {
      loadDashboardData("silent");
      if (selectedFixtureId) {
        loadMatchDetail(selectedFixtureId, true);
      }
    }, 60000);
    return () => clearInterval(timer);
  }, [loadDashboardData, loadMatchDetail, selectedFixtureId]);
  useEffect(() => {
    loadDashboardData("filter");
  }, [selectedDate, selectedMarket, phaseFilter, searchFilter, loadDashboardData]);
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
      dayData,
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
    },
    dataQuality: {
      report: qualityReport,
      isLoading: qualityLoading,
      error: qualityError,
      onLoadReport: loadDataQuality,
    },
    betslip: {
      targetDate: betslipDate,
      onChangeTargetDate: setBetslipDate,
      report: betslipReport,
      isLoading: betslipLoading,
      error: betslipError,
      onLoadReport: loadBetslip,
      dayData,
    },
    monitoring: {
      market: monitoringMarket,
      onChangeMarket: setMonitoringMarket,
      markets,
      report: monitoringReport,
      alerts: monitoringAlerts,
      isLoading: monitoringLoading,
      error: monitoringError,
      onLoadReport: loadMonitoring,
    },
    settings: {
      jobs: jobSettingsRows,
      isLoading: jobSettingsLoading,
      error: jobSettingsError,
      savingJobId: jobSettingsSavingId,
      onToggleJob: toggleJobSetting,
      onRefreshJobs: loadJobSettings,
      quota: apiQuota,
      quotaLoading: apiQuotaLoading,
      quotaError: apiQuotaError,
      onRefreshQuota: refreshApiQuotaLive,
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
          />
        )}
      </main>
    </div>
  );
}