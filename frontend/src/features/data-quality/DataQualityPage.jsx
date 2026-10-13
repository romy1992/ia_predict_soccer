import { useMemo, useState } from "react";

function parseCsvInts(value) {
  if (!value || !value.trim()) {
    return undefined;
  }
  const parsed = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item))
    .filter((item) => Number.isInteger(item));
  return parsed.length > 0 ? parsed : undefined;
}

function ratioPercent(value) {
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  return `${(num * 100).toFixed(2)}%`;
}

function coverageClass(value) {
  const num = Number(value);
  if (num >= 0.8) return "ok";
  if (num >= 0.4) return "warning";
  return "danger";
}

export default function DataQualityPage({ report, isLoading, error, onLoadReport }) {
  const [topN, setTopN] = useState("20");
  const [seasonsInput, setSeasonsInput] = useState("");
  const [leaguesInput, setLeaguesInput] = useState("");
  const [localError, setLocalError] = useState("");

  const seasons = useMemo(() => parseCsvInts(seasonsInput), [seasonsInput]);
  const leagues = useMemo(() => parseCsvInts(leaguesInput), [leaguesInput]);

  async function handleLoad() {
    const parsedTopN = Number(topN);
    if (!Number.isInteger(parsedTopN) || parsedTopN < 1) {
      setLocalError("top_n deve essere un intero >= 1");
      return;
    }
    setLocalError("");
    try {
      await onLoadReport({ topN: parsedTopN, seasons, leagues });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLocalError(message);
    }
  }

  const marketRows = Object.entries(report?.coverage?.markets || {}).map(([market, payload]) => ({
    market,
    fixtures: payload?.fixtures ?? 0,
    coverageRatio: payload?.coverage_ratio ?? 0,
  }));

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Data Quality Dashboard</h3>
          <button className="btn-primary" onClick={handleLoad} disabled={isLoading}>Aggiorna report</button>
        </div>
        <p className="muted">
          Ricalcola il report tramite il job <strong>"Aggiorna report Data Quality"</strong> (schedulabile/
          disattivabile in <strong>Impostazioni</strong>, di default ogni 60 minuti) - nessuna chiamata API-Sports,
          lavora solo sui dati gia' a DB.
          {report?.job_id && (
            <>
              {" "}Ultima esecuzione: job <code>{report.job_id}</code>
              {typeof report.duration_seconds === "number" && ` in ${report.duration_seconds.toFixed(2)}s`}.
            </>
          )}
        </p>

        <div className="inline-form">
          <label>
            top_n
            <input value={topN} onChange={(e) => setTopN(e.target.value)} />
          </label>
          <label>
            Seasons (csv)
            <input value={seasonsInput} onChange={(e) => setSeasonsInput(e.target.value)} placeholder="es. 2025,2026" />
          </label>
          <label>
            Leagues (csv)
            <input value={leaguesInput} onChange={(e) => setLeaguesInput(e.target.value)} placeholder="es. 135,39,140" />
          </label>
        </div>

        {localError && <div className="error-box">Errore: {localError}</div>}
        {error && <div className="error-box">Errore API: {error}</div>}
        {isLoading && <div className="info-box">Caricamento report qualità...</div>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Source</h3>
        </div>
        <div className="stats-grid">
          <article className="stat-card">
            <span>Fixtures</span>
            <strong>{report?.source?.fixtures_total ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Statistics rows</span>
            <strong>{report?.source?.statistics_rows_total ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Odds rows</span>
            <strong>{report?.source?.odds_rows_total ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Odds snapshots</span>
            <strong>{report?.source?.odds_snapshot_rows_total ?? 0}</strong>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Coverage per mercato</h3>
        </div>
        <div className="table-wrap">
          <table className="table-cards">
            <thead>
              <tr>
                <th>Mercato</th>
                <th>Fixtures</th>
                <th>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {marketRows.map((row) => (
                <tr key={`coverage-${row.market}`}>
                  <td data-label="Mercato">{row.market}</td>
                  <td data-label="Fixtures">{row.fixtures}</td>
                  <td data-label="Coverage">
                    {ratioPercent(row.coverageRatio)}
                    <div className="coverage-bar" aria-hidden="true">
                      <i
                        className={coverageClass(row.coverageRatio)}
                        style={{ width: `${Math.min(100, Math.max(0, Number(row.coverageRatio) * 100 || 0))}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Anomalie</h3>
        </div>
        <div className="stats-grid">
          <article className="stat-card">
            <span>Fixture incomplete</span>
            <strong>{report?.anomalies?.fixtures_incomplete_count ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Fixture senza statistics</span>
            <strong>{report?.anomalies?.fixtures_missing_statistics ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Fixture senza odds</span>
            <strong>{report?.anomalies?.fixtures_missing_odds ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Duplicati fixture</span>
            <strong>{report?.anomalies?.duplicate_fixture_count ?? 0}</strong>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Distribuzione (top)</h3>
        </div>
        <div className="detail-grid">
          <article className="detail-block">
            <h4>Per stagione</h4>
            <div className="table-wrap">
              <table className="table-cards">
                <thead>
                  <tr>
                    <th>Season</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(report?.distribution?.by_season || []).map((row) => (
                    <tr key={`season-${row.season}`}>
                      <td data-label="Season">{row.season}</td>
                      <td data-label="Count">{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="detail-block">
            <h4>Per lega</h4>
            <div className="table-wrap">
              <table className="table-cards">
                <thead>
                  <tr>
                    <th>Lega</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(report?.distribution?.by_league || []).map((row) => (
                    <tr key={`league-${row.league}`}>
                      <td data-label="Lega">{row.league}</td>
                      <td data-label="Count">{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Temporal & Settlement</h3>
        </div>
        <div className="stats-grid">
          <article className="stat-card">
            <span>Match datetime invalidi</span>
            <strong>{report?.temporal_checks?.invalid_match_datetime_count ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Snapshot dopo kickoff</span>
            <strong>{report?.temporal_checks?.snapshot_after_kickoff_count ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Settlement complete</span>
            <strong>{report?.settlement?.complete ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Settlement pending</span>
            <strong>{report?.settlement?.pending ?? 0}</strong>
          </article>
        </div>
      </section>
    </section>
  );
}

