export default function OpsPage({
  asyncRun,
  onChangeAsyncRun,
  onImport,
  onRetrain,
  opsMessage,
  health,
  jobsRows,
  onRefreshJobs,
}) {
  return (
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
              onChange={(e) => onChangeAsyncRun(e.target.checked)}
            />
            <span>Esegui async</span>
          </label>
          <button className="btn-primary" onClick={onImport}>Import giornaliero</button>
          <button className="btn-primary" onClick={onRetrain}>Retrain mercati</button>
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
          <button className="btn-secondary" onClick={onRefreshJobs}>Aggiorna jobs</button>
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
  );
}

