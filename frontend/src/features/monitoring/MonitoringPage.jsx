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
  officialPerformance,
  officialDays,
  onChangeOfficialDays,
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
          <label>
            Finestra ufficiale
            <select value={officialDays} onChange={(e) => onChangeOfficialDays(Number(e.target.value))}>
              <option value={7}>7 giorni</option>
              <option value={30}>30 giorni</option>
              <option value={90}>90 giorni</option>
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
          <h3>Performance ufficiale / Paper</h3>
          <span className="value-badge value-play">UFFICIALE/PAPER</span>
        </div>
        {!officialPerformance || officialPerformance.sample_size === 0 ? (
          <div className="empty-state">
            Nessuna PLAY ufficiale ancora registrata · <strong>DATI INSUFFICIENTI</strong>
          </div>
        ) : (
          <>
            <p className="muted">
              Solo PLAY catturate automaticamente prima del kickoff. Edge ed EV sono ex-ante; ROI e profitto
              sono osservati; CLV confronta la quota presa con la chiusura.
            </p>
            <div className="stats-grid">
              {[
                ["PLAY", officialPerformance.overall?.plays],
                ["Vinte", officialPerformance.overall?.wins],
                ["Perse", officialPerformance.overall?.losses],
                ["VOID", officialPerformance.overall?.void],
                ["Pending", officialPerformance.overall?.pending],
                ["Profitto", formatNumber(officialPerformance.overall?.total_profit, 2)],
                [
                  "ROI",
                  officialPerformance.overall?.roi == null
                    ? "Dati insufficienti"
                    : formatPercent(officialPerformance.overall.roi),
                ],
                [
                  "Hit rate",
                  officialPerformance.overall?.hit_rate == null
                    ? "Dati insufficienti"
                    : formatPercent(officialPerformance.overall.hit_rate),
                ],
                ["Quota media", formatOdd(officialPerformance.overall?.avg_odd)],
                ["Edge probabilistico (p.p.)", formatPercent(officialPerformance.overall?.avg_prob_edge)],
                ["EV teorico", formatPercent(officialPerformance.overall?.avg_ev)],
                ["Max drawdown", formatNumber(officialPerformance.overall?.max_drawdown, 2)],
                [
                  "CLV medio",
                  officialPerformance.overall?.avg_clv_odd_pct == null
                    ? "Dati insufficienti"
                    : formatPercent(officialPerformance.overall.avg_clv_odd_pct),
                ],
                [
                  "Copertura CLV",
                  officialPerformance.overall?.clv_coverage == null
                    ? "Dati insufficienti"
                    : formatPercent(officialPerformance.overall.clv_coverage),
                ],
              ].map(([label, value]) => (
                <article className="stat-card" key={label}>
                  <span>{label}</span>
                  <strong>{value ?? 0}</strong>
                </article>
              ))}
            </div>
            <p className="muted">
              Gross stake {formatNumber(officialPerformance.overall?.gross_stake, 2)} · Stake VOID restituito{" "}
              {formatNumber(officialPerformance.overall?.void_stake, 2)} · Active stake{" "}
              {formatNumber(officialPerformance.overall?.active_stake, 2)} · Campione{" "}
              {officialPerformance.sample_size}
            </p>
          </>
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
          <h3>Previsioni correnti</h3>
          <span className="value-badge">CORRENTI</span>
        </div>
        <p className="muted">
          I suggerimenti correnti restano nella Dashboard e non vengono sommati alla performance
          UFFICIALE/PAPER finché il job server-side non registra una PLAY pre-kickoff.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Performance non ufficiale / legacy</h3>
        </div>
        <p className="muted">
          Separata dalla coorte UFFICIALE/PAPER e dal BACKTEST OOS. Solo diagnostico: non influenza la Decision
          Policy né il gate di promozione.
        </p>
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
          <h3>Backtest OOS</h3>
          <span className="value-badge value-borderline">BACKTEST OOS</span>
        </div>
        <div className="empty-state">
          Report storico separato dalla performance ufficiale; nessun risultato OOS caricato in questa vista.
        </div>
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

