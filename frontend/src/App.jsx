import { useCallback, useEffect, useMemo, useState } from "react";
import {
  API_BASE_URL,
  getDashboardDay,
  getDashboardLive,
  getDashboardMatchDetail,
  getDashboardOverview,
  getHealth,
  getJobs,
  getMarkets,
  getPredictions,
  predict,
  triggerImport,
  triggerRetrain,
} from "./api";

const MENU_ITEMS = [
  { id: "dashboard", label: "Dashboard live" },
  { id: "live", label: "Partite in diretta" },
  { id: "today", label: "Partite del giorno" },
  { id: "predictions", label: "Storico previsioni" },
  { id: "ops", label: "Operazioni ML" },
];

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function formatPercent(value) {
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "--";
  }
  return `${(num * 100).toFixed(1)}%`;
}

function marketLabel(market) {
  const map = {
    h2h: "Vincitore partita",
    goal_no_goal: "Goal / No Goal",
    dc: "Doppia chance",
    corners: "Corners",
    cards: "Cards",
    under_over_1_5: "Over/Under 1.5",
    under_over_2_5: "Over/Under 2.5",
    under_over_3_5: "Over/Under 3.5",
    under_over_4_5: "Over/Under 4.5",
  };
  return map[market] || market;
}

function predictionLabel(market, prediction, row) {
  if (market === "goal_no_goal") {
    return prediction === 1 ? "Goal" : "No Goal";
  }
  if (market === "dc") {
    return prediction === 1 ? "1X" : "X2";
  }
  if (market.startsWith("under_over_")) {
    const threshold = market.replace("under_over_", "").replace("_", ".");
    return prediction === 1 ? `Over ${threshold}` : `Under ${threshold}`;
  }
  if (market === "h2h") {
    return prediction === 1 ? row.home : "Non casa";
  }
  if (market === "corners") {
    return prediction === 1 ? "Over corners" : "Under corners";
  }
  if (market === "cards") {
    return prediction === 1 ? "Over cards" : "Under cards";
  }
  return String(prediction);
}

function phaseLabel(phase) {
  if (phase === "live") {
    return "In diretta";
  }
  if (phase === "finished") {
    return "Finita";
  }
  return "Da giocare";
}

function phaseClass(phase) {
  if (phase === "live") {
    return "badge-live";
  }
  if (phase === "finished") {
    return "badge-finished";
  }
  return "badge-upcoming";
}

function confidenceClass(probability) {
  if (probability >= 0.8) {
    return "prediction-strong";
  }
  if (probability >= 0.65) {
    return "prediction-medium";
  }
  return "prediction-low";
}

function formatOdd(value) {
  const num = Number(value);
  if (Number.isNaN(num) || num <= 0) {
    return "-";
  }
  return num.toFixed(2);
}

function formatEdge(value) {
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  const pct = num * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

function valueClass(valueLabel) {
  if (valueLabel === "PLAY") {
    return "value-play";
  }
  if (valueLabel === "BORDERLINE") {
    return "value-borderline";
  }
  return "value-no-bet";
}

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

  const [predictOutput, setPredictOutput] = useState("");
  const [opsMessage, setOpsMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState("");
  const [selectedFixtureId, setSelectedFixtureId] = useState(null);
  const [matchDetail, setMatchDetail] = useState(null);
  const [matchDetailLoading, setMatchDetailLoading] = useState(false);
  const [matchDetailError, setMatchDetailError] = useState("");

  const marketsQuery = useMemo(() => {
    if (selectedMarket === "all") {
      return undefined;
    }
    return [selectedMarket];
  }, [selectedMarket]);

  const safeRows = dayData?.rows || [];

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

  async function handleImport() {
    try {
      const data = await triggerImport(asyncRun);
      setOpsMessage(JSON.stringify(data, null, 2));
      const jobsData = await getJobs(80);
      setJobsRows(jobsData?.rows || []);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setOpsMessage(message);
    }
  }

  async function handleRetrain() {
    try {
      const data = await triggerRetrain(asyncRun);
      setOpsMessage(JSON.stringify(data, null, 2));
      const jobsData = await getJobs(80);
      setJobsRows(jobsData?.rows || []);
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
      const latestLog = await getPredictions(80);
      setPredictionRows(latestLog?.rows || []);
      await loadDashboardData(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setPredictOutput(message);
    }
  }

  function renderPredictionBadges(row) {
    const predictions = row?.predictions || {};
    const entries = Object.entries(predictions);
    if (entries.length === 0) {
      return <span className="empty-state">Nessuna previsione</span>;
    }

    return (
      <div className="prediction-badges">
        {entries.map(([marketKey, payload]) => (
          <span className={`prediction-chip ${confidenceClass(payload.probability)}`} key={`${row.fixture_id}-${marketKey}`}>
            <strong>{marketLabel(marketKey)}</strong>
            <em>{predictionLabel(marketKey, payload.prediction, row)}</em>
            <small>{formatPercent(payload.probability)}</small>
          </span>
        ))}
      </div>
    );
  }

  function renderMatchRows(rows) {
    if (!rows || rows.length === 0) {
      return <div className="empty-panel">Nessuna partita trovata per i filtri correnti.</div>;
    }

    return (
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ora</th>
              <th>Torneo</th>
              <th>Match</th>
              <th>Score</th>
              <th>Stato</th>
              <th>Previsioni</th>
              <th>Dettaglio</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`match-${row.fixture_id}`}
                className={selectedFixtureId === row.fixture_id ? "row-selected" : ""}
              >
                <td>{row.time}</td>
                <td>{row.league || "-"}</td>
                <td>
                  <div className="match-title">{row.home} vs {row.away}</div>
                  <div className="match-sub">Fixture {row.fixture_id}</div>
                </td>
                <td>
                  {row.score?.home ?? "-"} - {row.score?.away ?? "-"}
                </td>
                <td>
                  <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
                </td>
                <td>{renderPredictionBadges(row)}</td>
                <td>
                  <button className="btn-secondary" onClick={() => openMatchDetail(row.fixture_id)}>
                    Apri
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  function renderMatchDetailPanel() {
    if (!selectedFixtureId && !matchDetail) {
      return null;
    }

    if (matchDetailLoading) {
      return <section className="panel detail-panel"><div className="empty-panel">Caricamento dettaglio partita...</div></section>;
    }

    if (matchDetailError) {
      return <section className="panel detail-panel"><div className="error-box">{matchDetailError}</div></section>;
    }

    const fixture = matchDetail?.fixture;
    if (!fixture) {
      return <section className="panel detail-panel"><div className="empty-panel">Dettaglio non disponibile per il fixture selezionato.</div></section>;
    }

    const decisionCards = matchDetail?.decision_cards || [];
    const timeline = matchDetail?.timeline || [];
    const oddsSummary = matchDetail?.odds_summary || {};
    const oddsMarkets = Object.keys(oddsSummary);

    return (
      <section className="panel detail-panel">
        <div className="panel-header">
          <h3>Dettaglio match: {fixture.home} vs {fixture.away}</h3>
          <button
            className="btn-secondary"
            onClick={() => {
              setSelectedFixtureId(null);
              setMatchDetail(null);
            }}
          >
            Chiudi dettaglio
          </button>
        </div>

        <div className="detail-head-meta">
          <span className={`phase-badge ${phaseClass(fixture.phase)}`}>{phaseLabel(fixture.phase)}</span>
          <span>{fixture.date} {fixture.time}</span>
          <span>{fixture.league || "-"}</span>
          <span>Score: {fixture.score?.home ?? "-"} - {fixture.score?.away ?? "-"}</span>
          <span>Fonte: {fixture.source || "-"}</span>
        </div>

        <div className="detail-grid">
          <article className="detail-block">
            <h4>Consiglio valore (PLAY / BORDERLINE / NO BET)</h4>
            {decisionCards.length === 0 ? (
              <div className="empty-panel">Nessun modello disponibile o feature non ancora presenti per questo fixture.</div>
            ) : (
              <div className="decision-grid">
                {decisionCards.map((card) => (
                  <div className="decision-card" key={`${card.market}-${card.run_id || card.pick}`}>
                    <div className="decision-head">
                      <strong>{marketLabel(card.market)}</strong>
                      <span className={`value-badge ${valueClass(card.value_label)}`}>{card.value_label}</span>
                    </div>
                    <p className="decision-pick">{card.pick}</p>
                    <div className="decision-metrics">
                      <span>Conf.: {formatPercent(card.predicted_probability)}</span>
                      <span>Quota media: {formatOdd(card.odd)}</span>
                      <span>Edge: {formatEdge(card.edge)}</span>
                    </div>
                    <small>{card.value_reason}</small>
                  </div>
                ))}
              </div>
            )}
          </article>

          <article className="detail-block">
            <h4>Timeline eventi</h4>
            {timeline.length === 0 ? (
              <div className="empty-panel">Nessun evento disponibile per questo match.</div>
            ) : (
              <div className="timeline-list">
                {timeline.map((event, idx) => (
                  <div className="timeline-item" key={`${event.minute}-${idx}`}>
                    <span className="timeline-minute">{event.minute}</span>
                    <div>
                      <strong>{event.team || "-"}</strong>
                      <p>{event.type || "Evento"}{event.detail ? ` - ${event.detail}` : ""}</p>
                      {(event.player || event.assist || event.comments) && (
                        <small>
                          {[event.player, event.assist ? `assist ${event.assist}` : null, event.comments]
                            .filter(Boolean)
                            .join(" | ")}
                        </small>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </article>
        </div>

        <article className="detail-block">
          <h4>Quote medie bookmaker</h4>
          {oddsMarkets.length === 0 ? (
            <div className="empty-panel">Nessuna quota disponibile al momento.</div>
          ) : (
            <div className="odds-market-grid">
              {oddsMarkets.map((marketKey) => (
                <div className="odds-market-card" key={`odds-${marketKey}`}>
                  <h5>{marketLabel(marketKey)}</h5>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Outcome</th>
                          <th>Avg</th>
                          <th>Min</th>
                          <th>Max</th>
                          <th>Book</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(oddsSummary[marketKey] || []).map((odd, idx) => (
                          <tr key={`${marketKey}-${idx}`}>
                            <td>{odd.outcome}</td>
                            <td>{formatOdd(odd.avg_odd)}</td>
                            <td>{formatOdd(odd.min_odd)}</td>
                            <td>{formatOdd(odd.max_odd)}</td>
                            <td>{odd.bookmakers ?? "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          )}

          {matchDetail?.odds_updated_at && (
            <small className="muted">Aggiornamento quote API: {matchDetail.odds_updated_at}</small>
          )}
        </article>
      </section>
    );
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h2>soccer_oracle</h2>
          <p>Live center + predizioni</p>
        </div>

        <button className="btn-primary full" onClick={loadEverything}>Aggiorna tutto</button>

        <div className="sidebar-meta">
          <small>Ultimo update: {lastRefresh || "-"}</small>
          <small>API: {health.status || "offline"}</small>
        </div>

        <nav className="menu">
          {MENU_ITEMS.map((item) => (
            <button
              key={item.id}
              className={activePage === item.id ? "menu-item active" : "menu-item"}
              onClick={() => setActivePage(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <h1>Dashboard partite e previsioni</h1>
            <p>
              Vista giornaliera stile tennis_oracle con menu, live match e previsioni multi-mercato.
            </p>
          </div>

          <div className="filters">
            <label>
              Data
              <input type="date" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} />
            </label>

            <label>
              Mercato
              <select value={selectedMarket} onChange={(e) => setSelectedMarket(e.target.value)}>
                {markets.map((item) => (
                  <option key={item} value={item}>
                    {item === "all" ? "Tutti i mercati" : marketLabel(item)}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Cerca match
              <input
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Es: Inter, Premier, Real"
              />
            </label>

            <button className="btn-secondary" onClick={() => setSearchFilter(searchInput.trim())}>Cerca</button>
          </div>
        </header>

        {error && <div className="error-box">Errore: {error}</div>}
        {isLoading && <div className="info-box">Caricamento dashboard...</div>}

        {activePage === "dashboard" && (
          <section className="stack">
            <div className="stats-grid">
              <article className="stat-card">
                <span>Partite del giorno</span>
                <strong>{overview?.counts?.total ?? 0}</strong>
              </article>
              <article className="stat-card">
                <span>In diretta</span>
                <strong>{overview?.counts?.live ?? 0}</strong>
              </article>
              <article className="stat-card">
                <span>Da giocare</span>
                <strong>{overview?.counts?.to_play ?? 0}</strong>
              </article>
              <article className="stat-card">
                <span>Mercati con modello</span>
                <strong>{overview?.model_markets?.length ?? 0}</strong>
              </article>
            </div>

            <section className="panel">
              <div className="panel-header">
                <h3>Partite in diretta</h3>
                <span className="pill">{liveData.returned}/{liveData.total}</span>
              </div>
              <div className="live-grid">
                {(liveData.rows || []).slice(0, 8).map((row) => (
                  <article className="live-card live-clickable" key={`live-${row.fixture_id}`} onClick={() => openMatchDetail(row.fixture_id)}>
                    <div className="live-head">
                      <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
                      <small>{row.time}</small>
                    </div>
                    <h4>{row.home} vs {row.away}</h4>
                    <p className="score-big">{row.score?.home ?? "-"} - {row.score?.away ?? "-"}</p>
                    <small>{row.league || "-"}</small>
                  </article>
                ))}
                {(liveData.rows || []).length === 0 && <div className="empty-panel">Nessuna partita live al momento.</div>}
              </div>
            </section>

            <section className="panel">
              <div className="panel-header">
                <h3>Partite del giorno con previsioni</h3>
                <span className="pill">{dayData.returned}/{dayData.total}</span>
              </div>
              {renderMatchRows(safeRows.slice(0, 12))}
            </section>
          </section>
        )}

        {activePage === "live" && (
          <section className="panel">
            <div className="panel-header">
              <h3>Diretta completa</h3>
              <span className="pill">{liveData.returned}/{liveData.total}</span>
            </div>
            {renderMatchRows(liveData.rows || [])}
          </section>
        )}

        {activePage === "today" && (
          <section className="panel">
            <div className="panel-header">
              <h3>Calendario del giorno</h3>
              <span className="pill">{dayData.returned}/{dayData.total}</span>
            </div>

            <div className="tabs">
              {["all", "to_play", "live", "finished"].map((item) => (
                <button
                  key={item}
                  className={phaseFilter === item ? "tab active" : "tab"}
                  onClick={() => setPhaseFilter(item)}
                >
                  {item === "all" ? "Tutte" : phaseLabel(item)}
                </button>
              ))}
            </div>

            {renderMatchRows(safeRows)}
          </section>
        )}

        {activePage === "predictions" && (
          <section className="stack">
            <section className="panel">
              <div className="panel-header">
                <h3>Crea previsione manuale</h3>
              </div>
              <div className="inline-form">
                <label>
                  Fixture id
                  <input value={manualFixtureId} onChange={(e) => setManualFixtureId(e.target.value)} />
                </label>
                <label>
                  Mercato
                  <select value={manualMarket} onChange={(e) => setManualMarket(e.target.value)}>
                    {markets.filter((item) => item !== "all").map((item) => (
                      <option key={item} value={item}>{marketLabel(item)}</option>
                    ))}
                  </select>
                </label>
                <button className="btn-primary" onClick={handleManualPredict}>Calcola previsione</button>
              </div>
              {predictOutput && <pre className="code-block">{predictOutput}</pre>}
            </section>

            <section className="panel">
              <div className="panel-header">
                <h3>Storico previsioni</h3>
                <button className="btn-secondary" onClick={async () => setPredictionRows((await getPredictions(80)).rows || [])}>
                  Aggiorna storico
                </button>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Fixture</th>
                      <th>Mercato</th>
                      <th>Prediction</th>
                      <th>Probabilita</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionRows.map((row, idx) => (
                      <tr key={`prediction-${idx}`}>
                        <td>{row.timestamp || "-"}</td>
                        <td>{row.fixture_id}</td>
                        <td>{marketLabel(row.market)}</td>
                        <td>{String(row.prediction)}</td>
                        <td>{formatPercent(row.probability)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </section>
        )}

        {activePage === "ops" && (
          <section className="stack">
            <section className="panel">
              <div className="panel-header">
                <h3>Controlli operativi</h3>
              </div>
              <div className="inline-form">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={asyncRun}
                    onChange={(e) => setAsyncRun(e.target.checked)}
                  />
                  <span>Esegui async</span>
                </label>
                <button className="btn-primary" onClick={handleImport}>Import giornaliero</button>
                <button className="btn-primary" onClick={handleRetrain}>Retrain mercati</button>
              </div>
              {opsMessage && <pre className="code-block">{opsMessage}</pre>}
            </section>

            <section className="panel">
              <div className="panel-header">
                <h3>Stato servizio</h3>
              </div>
              <pre className="code-block">{JSON.stringify(health, null, 2)}</pre>
            </section>

            <section className="panel">
              <div className="panel-header">
                <h3>Storico jobs</h3>
                <button className="btn-secondary" onClick={async () => setJobsRows((await getJobs(80)).rows || [])}>
                  Aggiorna jobs
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Tipo job</th>
                      <th>Stato</th>
                      <th>Durata (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobsRows.map((row, idx) => (
                      <tr key={`job-${idx}`}>
                        <td>{row.timestamp || "-"}</td>
                        <td>{row.job_type || "-"}</td>
                        <td>{row.status || "-"}</td>
                        <td>{Number(row.duration_seconds || 0).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </section>
        )}

        {renderMatchDetailPanel()}
      </main>
    </div>
  );
}










