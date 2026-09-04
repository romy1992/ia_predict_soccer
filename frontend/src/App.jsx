import { useCallback, useEffect, useMemo, useState } from "react";
import {
  API_BASE_URL,
  getDataQuality,
  getDashboardDay,
  getDashboardLive,
  getDashboardMatchDetail,
  getDashboardOverview,
  getBetslipGenerate,
  getHealth,
  getJobs,
  getMarkets,
  getPredictions,
  predict,
  triggerFutureSync,
  triggerImport,
  triggerSettlement,
  triggerTodayUpdate,
  triggerRetrain,
} from "./api";
import AppRouter from "./features/layout/AppRouter";
import Sidebar from "./features/layout/Sidebar";
import TopFilters from "./features/layout/TopFilters";
import MatchDetailPanel from "./features/matches/components/MatchDetailPanel";
import { todayIso } from "./features/shared/formatters";

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [selectedDate, setSelectedDate] = useState(todayIso());
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

  const [predictOutput, setPredictOutput] = useState("");
  const [opsMessage, setOpsMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
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
  const showMatchFilters = ["dashboard", "live", "today", "predictions"].includes(activePage);

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
    const [healthData, marketsData, jobsData, predData] = await Promise.all([
      getHealth(),
      getMarkets(),
      getJobs(80),
      getPredictions(80),
    ]);

    setHealth({ ...healthData, apiBaseUrl: API_BASE_URL });
    const apiMarkets = marketsData?.markets || [];
    const mergedMarkets = ["all", ...apiMarkets];
    setMarkets(mergedMarkets);

    setSelectedMarket((prev) => (mergedMarkets.includes(prev) ? prev : "all"));
    setManualMarket((prev) => (apiMarkets.includes(prev) ? prev : apiMarkets[0] || "under_over_2_5"));

    setJobsRows(jobsData?.rows || []);
    setPredictionRows(predData?.rows || []);
  }, []);

  const loadDashboardData = useCallback(
    async (silent = false) => {
      if (!silent) {
        setIsLoading(true);
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
        if (!silent) {
          setIsLoading(false);
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

  const loadEverything = useCallback(async () => {
    setIsLoading(true);
    setError("");
    try {
      await loadMetaData();
      await loadDashboardData(true);
      setLastRefresh(new Date().toLocaleString("it-IT"));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [loadMetaData, loadDashboardData]);

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
    setActivePage(previousPage || "today");
  }, [previousPage]);

  useEffect(() => {
    loadEverything();
  }, [loadEverything]);

  useEffect(() => {
    const timer = setInterval(() => {
      loadDashboardData(true);
      if (selectedFixtureId) {
        loadMatchDetail(selectedFixtureId, true);
      }
    }, 60000);
    return () => clearInterval(timer);
  }, [loadDashboardData, loadMatchDetail, selectedFixtureId]);

  useEffect(() => {
    loadDashboardData();
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
      await loadDashboardData(true);
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
    },
    live: {
      liveData,
      selectedFixtureId,
      onOpenMatch: openMatchDetail,
      onOpenOracleDetail: openOracleDetail,
    },
    today: {
      dayData,
      safeRows,
      phaseFilter,
      onChangePhaseFilter: setPhaseFilter,
      selectedFixtureId,
      onOpenMatch: openMatchDetail,
      onOpenOracleDetail: openOracleDetail,
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
  };

  return (
    <div className="layout">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        onRefreshAll={loadEverything}
        lastRefresh={lastRefresh}
        healthStatus={health.status}
      />

      <main className="content">
        {showMatchFilters && (
          <TopFilters
            selectedDate={selectedDate}
            onChangeDate={setSelectedDate}
            markets={markets}
            selectedMarket={selectedMarket}
            onChangeSelectedMarket={setSelectedMarket}
            searchInput={searchInput}
            onChangeSearchInput={setSearchInput}
            onApplySearch={() => setSearchFilter(searchInput.trim())}
          />
        )}

        {error && <div className="error-box">Errore: {error}</div>}
        {isLoading && <div className="info-box">Caricamento dashboard...</div>}

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


















