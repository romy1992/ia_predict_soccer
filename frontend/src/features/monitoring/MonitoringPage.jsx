import { formatEdge, formatNumber, formatOdd, formatPercent, marketLabel, severityClass } from "../shared/formatters";

function ratioPercent(value) {
  if (value === null || value === undefined) {
    return "n/d";
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "n/d";
  }
  return `${(num * 100).toFixed(1)}%`;
}

function SeverityBadge({ severity }) {
  return <span className={`value-badge ${severityClass(severity)}`}>{severity}</span>;
}

/**
 * OPS-03: Monitoring performance modello e data drift. Mostra SOLO dati
 * gia' calcolati dal backend (`MonitoringService`) - nessun calcolo di
 * drift/ROI/coverage qui, solo formattazione (stesso principio "FE non
 * ricalcola logica scientifica" gia' applicato al resto della dashboard).
 */
export default function MonitoringPage({
  report,
  alerts,
  isLoading,
  error,
  market,
  onChangeMarket,
  markets = ["all"],
  onLoadReport,
}) {
  const roiWindows = report?.roi_rolling?.windows || {};
  const volumeSeries = (report?.prediction_volume?.series || []).slice(-14);
  const featureRows = report?.feature_coverage?.per_feature || [];
  const calibration = report?.calibration_drift;

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Monitoring</h3>
          <button className="btn-primary" onClick={onLoadReport} disabled={isLoading}>
            Aggiorna
          </button>
        </div>

        <div className="inline-form">
          <label>
            Mercato
            <select value={market} onChange={(e) => onChangeMarket(e.target.value)}>
              {markets.map((item) => (
                <option key={item} value={item}>
                  {item === "all" ? "Tutti i mercati" : marketLabel(item)}
                </option>
              ))}
            </select>
          </label>
        </div>

        {error && <div className="error-box">Errore API: {error}</div>}
        {isLoading && <div className="info-box">Caricamento monitoring...</div>}
        {report?.generated_at && (
          <p className="muted">
            Generato il {new Date(report.generated_at).toLocaleString("it-IT")} · policy {report.thresholds_version}
          </p>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Alert ({(alerts || []).length})</h3>
        </div>
        {(alerts || []).length === 0 && <div className="empty-state">Nessun alert attivo.</div>}
        {(alerts || []).length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Codice</th>
                  <th>Messaggio</th>
                  <th>Osservato</th>
                  <th>Soglia</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((alert, index) => (
                  <tr key={`${alert.code}-${index}`}>
                    <td>
                      <SeverityBadge severity={alert.severity} />
                    </td>
                    <td>{alert.code}</td>
                    <td>{alert.message}</td>
                    <td>{alert.observed ?? "n/d"}</td>
                    <td>{alert.threshold ?? "n/d"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Prediction volume</h3>
        </div>
        <div className="stats-grid">
          <article className="stat-card">
            <span>Totale ({report?.prediction_volume?.days ?? 0}gg)</span>
            <strong>{report?.prediction_volume?.total ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Media/giorno</span>
            <strong>{formatNumber(report?.prediction_volume?.average_per_day, 1)}</strong>
          </article>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Data</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {volumeSeries.map((row) => (
                <tr key={row.date}>
                  <td>{row.date}</td>
                  <td>{row.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>ROI rolling</h3>
        </div>
        <p className="muted">Solo diagnostico: non influenza la Decision Policy ne' il gate di promozione.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Finestra</th>
                <th>Bet piazzate</th>
                <th>Hit rate</th>
                <th>ROI</th>
                <th>Profit</th>
                <th>Avg odd</th>
                <th>Max drawdown</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(roiWindows).map(([days, payload]) => {
                const overall = payload?.overall || {};
                return (
                  <tr key={days}>
                    <td>{days}gg</td>
                    <td>{overall.bets ?? 0}</td>
                    <td>{formatPercent(overall.hit_rate)}</td>
                    <td>{formatEdge(overall.roi)}</td>
                    <td>{formatNumber(overall.profit)}</td>
                    <td>{formatOdd(overall.avg_odds)}</td>
                    <td>{formatNumber(overall.max_drawdown)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Calibration drift</h3>
        </div>
        {!market || market === "all" ? (
          <div className="empty-state">Seleziona un mercato per calcolare il drift di calibrazione.</div>
        ) : calibration === null || calibration === undefined ? (
          <div className="empty-state">Nessuna prediction settled disponibile per questo mercato.</div>
        ) : calibration.available === false ? (
          <div className="info-box">{calibration.reason}</div>
        ) : (
          <div className="stats-grid">
            <article className="stat-card">
              <span>ECE recente</span>
              <strong>{formatNumber(calibration.recent_metrics?.ece, 4)}</strong>
            </article>
            <article className="stat-card">
              <span>ECE baseline</span>
              <strong>{formatNumber(calibration.baseline_metrics?.ece, 4)}</strong>
            </article>
            <article className="stat-card">
              <span>Drift ECE</span>
              <strong>{formatNumber(calibration.drift?.ece_drift, 4)}</strong>
            </article>
            <article className="stat-card">
              <span>Campioni (recent/baseline)</span>
              <strong>
                {calibration.drift?.recent_sample_size ?? 0} / {calibration.drift?.baseline_sample_size ?? 0}
              </strong>
            </article>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Feature coverage</h3>
        </div>
        {!market || market === "all" ? (
          <div className="empty-state">Seleziona un mercato per la coverage delle feature.</div>
        ) : !report?.feature_coverage ? (
          <div className="empty-state">Nessun modello registrato per questo mercato.</div>
        ) : (
          <>
            <div className="stats-grid">
              <article className="stat-card">
                <span>Fixture coverage</span>
                <strong>{ratioPercent(report.feature_coverage.fixture_coverage_ratio)}</strong>
              </article>
              <article className="stat-card">
                <span>Presenza feature (media)</span>
                <strong>{ratioPercent(report.feature_coverage.overall_column_presence_ratio)}</strong>
              </article>
              <article className="stat-card">
                <span>Feature sempre assenti</span>
                <strong>{(report.feature_coverage.fully_missing_features || []).length}</strong>
              </article>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Presenza</th>
                  </tr>
                </thead>
                <tbody>
                  {featureRows.map((row) => (
                    <tr key={row.feature}>
                      <td>{row.feature}</td>
                      <td>{ratioPercent(row.column_presence_ratio)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </section>
  );
}

