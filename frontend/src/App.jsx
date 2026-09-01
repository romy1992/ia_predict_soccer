import { useEffect, useMemo, useState } from "react";
import {
  API_BASE_URL,
  getHealth,
  getJobs,
  getMarkets,
  getMetrics,
  getPredictions,
  predict,
  triggerImport,
  triggerRetrain,
} from "./api";

function pretty(value) {
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

export default function App() {
  const [markets, setMarkets] = useState([]);
  const [market, setMarket] = useState("under_over_2_5");
  const [fixtureId, setFixtureId] = useState("1326590");
  const [asyncRun, setAsyncRun] = useState(true);

  const [health, setHealth] = useState("Caricamento...");
  const [jobsOutput, setJobsOutput] = useState("In attesa");
  const [predictOutput, setPredictOutput] = useState("In attesa");
  const [metricsOutput, setMetricsOutput] = useState("In attesa");
  const [jobsHistoryOutput, setJobsHistoryOutput] = useState("In attesa");
  const [predHistoryOutput, setPredHistoryOutput] = useState("In attesa");

  const currentMarket = useMemo(() => market || markets[0] || "under_over_2_5", [market, markets]);

  async function refreshBaseData() {
    try {
      const [healthData, marketsData, jobsData, predData] = await Promise.all([
        getHealth(),
        getMarkets(),
        getJobs(50),
        getPredictions(50),
      ]);

      setHealth(pretty({ apiBaseUrl: API_BASE_URL, ...healthData }));

      const marketList = marketsData?.markets || [];
      setMarkets(marketList);
      if (!market && marketList.length > 0) {
        setMarket(marketList[0]);
      }

      setJobsHistoryOutput(pretty(jobsData));
      setPredHistoryOutput(pretty(predData));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setHealth(`Errore: ${message}`);
    }
  }

  async function handleImport() {
    try {
      const data = await triggerImport(asyncRun);
      setJobsOutput(pretty(data));
      const jobsData = await getJobs(50);
      setJobsHistoryOutput(pretty(jobsData));
    } catch (error) {
      setJobsOutput(pretty(error instanceof Error ? error.message : error));
    }
  }

  async function handleRetrain() {
    try {
      const data = await triggerRetrain(asyncRun);
      setJobsOutput(pretty(data));
      const jobsData = await getJobs(50);
      setJobsHistoryOutput(pretty(jobsData));
    } catch (error) {
      setJobsOutput(pretty(error instanceof Error ? error.message : error));
    }
  }

  async function handlePredict() {
    try {
      const data = await predict(currentMarket, fixtureId);
      setPredictOutput(pretty(data));
      const predData = await getPredictions(50);
      setPredHistoryOutput(pretty(predData));
    } catch (error) {
      setPredictOutput(pretty(error instanceof Error ? error.message : error));
    }
  }

  async function handleMetrics() {
    try {
      const data = await getMetrics(currentMarket, 20);
      setMetricsOutput(pretty(data));
    } catch (error) {
      setMetricsOutput(pretty(error instanceof Error ? error.message : error));
    }
  }

  useEffect(() => {
    refreshBaseData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page">
      <header>
        <h1>Soccer ML Dashboard</h1>
        <p>Frontend React per monitoraggio training, predizioni e job.</p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>API Health</h2>
          <pre>{health}</pre>
          <button onClick={refreshBaseData}>Aggiorna Stato</button>
        </section>

        <section className="card">
          <h2>Job Manuali</h2>
          <label className="row">
            <input
              type="checkbox"
              checked={asyncRun}
              onChange={(e) => setAsyncRun(e.target.checked)}
            />
            <span>Async run</span>
          </label>
          <div className="row">
            <button onClick={handleImport}>Run Import</button>
            <button onClick={handleRetrain}>Run Retrain</button>
          </div>
          <pre>{jobsOutput}</pre>
        </section>

        <section className="card">
          <h2>Predizione</h2>
          <div className="row">
            <label>Mercato</label>
            <select value={currentMarket} onChange={(e) => setMarket(e.target.value)}>
              {markets.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <div className="row">
            <label>Fixture ID</label>
            <input value={fixtureId} onChange={(e) => setFixtureId(e.target.value)} />
            <button onClick={handlePredict}>Predict</button>
          </div>
          <pre>{predictOutput}</pre>
        </section>

        <section className="card">
          <h2>Metriche Mercato</h2>
          <button onClick={handleMetrics}>Load Metrics</button>
          <pre>{metricsOutput}</pre>
        </section>

        <section className="card">
          <h2>Storico Job</h2>
          <button onClick={async () => setJobsHistoryOutput(pretty(await getJobs(50)))}>
            Aggiorna Storico
          </button>
          <pre>{jobsHistoryOutput}</pre>
        </section>

        <section className="card">
          <h2>Storico Predizioni</h2>
          <button onClick={async () => setPredHistoryOutput(pretty(await getPredictions(50)))}>
            Aggiorna Storico
          </button>
          <pre>{predHistoryOutput}</pre>
        </section>
      </main>
    </div>
  );
}

