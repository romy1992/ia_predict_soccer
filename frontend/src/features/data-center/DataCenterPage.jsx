import { useMemo, useState } from "react";

function parseCsvInts(value) {
  if (!value || !value.trim()) {
    return undefined;
  }
  const values = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item))
    .filter((item) => Number.isInteger(item));
  return values.length > 0 ? values : undefined;
}

export default function DataCenterPage({
  asyncRun,
  onChangeAsyncRun,
  onRunHistoricalImport,
  onRunTodayUpdate,
  onRunFutureSync,
  onRunSettlement,
  jobsRows,
  onRefreshJobs,
  opsMessage,
  health,
}) {
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [targetDate, setTargetDate] = useState("");
  const [daysAhead, setDaysAhead] = useState("7");
  const [statuses, setStatuses] = useState("FT-AET-PEN-ABD");
  const [seasonsInput, setSeasonsInput] = useState("");
  const [leaguesInput, setLeaguesInput] = useState("");
  const [localError, setLocalError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const parsedSeasons = useMemo(() => parseCsvInts(seasonsInput), [seasonsInput]);
  const parsedLeagues = useMemo(() => parseCsvInts(leaguesInput), [leaguesInput]);

  async function runAction(action) {
    setLocalError("");
    setIsSubmitting(true);
    try {
      await action();
      await onRefreshJobs();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLocalError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleHistoricalImport() {
    if (!fromDate || !toDate) {
      setLocalError("Inserisci from_date e to_date per import storico.");
      return;
    }
    if (fromDate > toDate) {
      setLocalError("from_date deve essere minore o uguale a to_date.");
      return;
    }

    await runAction(() =>
      onRunHistoricalImport({
        async_run: asyncRun,
        from_date: fromDate,
        to_date: toDate,
        statuses,
        seasons: parsedSeasons,
        leagues: parsedLeagues,
      })
    );
  }

  async function handleTodayUpdate() {
    await runAction(() =>
      onRunTodayUpdate({
        async_run: asyncRun,
        target_date: targetDate || undefined,
        seasons: parsedSeasons,
        leagues: parsedLeagues,
      })
    );
  }

  async function handleFutureSync() {
    const parsedDays = Number(daysAhead);
    if (!Number.isInteger(parsedDays) || parsedDays < 1) {
      setLocalError("days_ahead deve essere un intero >= 1.");
      return;
    }

    await runAction(() =>
      onRunFutureSync({
        async_run: asyncRun,
        days_ahead: parsedDays,
        seasons: parsedSeasons,
        leagues: parsedLeagues,
      })
    );
  }

  async function handleSettlement() {
    await runAction(() =>
      onRunSettlement({
        async_run: asyncRun,
        from_date: fromDate || undefined,
        to_date: toDate || undefined,
        seasons: parsedSeasons,
        leagues: parsedLeagues,
      })
    );
  }

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Data Center - Import manuale</h3>
        </div>

        <div className="inline-form">
          <label>
            Seasons (csv)
            <input
              value={seasonsInput}
              onChange={(e) => setSeasonsInput(e.target.value)}
              placeholder="es. 2025,2026"
            />
          </label>

          <label>
            Leagues (csv)
            <input
              value={leaguesInput}
              onChange={(e) => setLeaguesInput(e.target.value)}
              placeholder="es. 135,39,140"
            />
          </label>

          <label>
            Statuses
            <input
              value={statuses}
              onChange={(e) => setStatuses(e.target.value)}
              placeholder="FT-AET-PEN-ABD"
            />
          </label>

          <label className="check">
            <input
              type="checkbox"
              checked={asyncRun}
              onChange={(e) => onChangeAsyncRun(e.target.checked)}
            />
            <span>Esegui async</span>
          </label>
        </div>

        <div className="inline-form" style={{ marginTop: 10 }}>
          <label>
            From date
            <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} />
          </label>
          <label>
            To date
            <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} />
          </label>
          <button className="btn-primary" disabled={isSubmitting} onClick={handleHistoricalImport}>
            Import storico
          </button>
          <button className="btn-secondary" disabled={isSubmitting} onClick={handleSettlement}>
            Settlement finali
          </button>
        </div>

        <div className="inline-form" style={{ marginTop: 10 }}>
          <label>
            Target date
            <input type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
          </label>
          <button className="btn-primary" disabled={isSubmitting} onClick={handleTodayUpdate}>
            Aggiorna oggi
          </button>

          <label>
            Days ahead
            <input value={daysAhead} onChange={(e) => setDaysAhead(e.target.value)} />
          </label>
          <button className="btn-primary" disabled={isSubmitting} onClick={handleFutureSync}>
            Sync future
          </button>
        </div>

        {localError && <div className="error-box">Errore: {localError}</div>}
        {opsMessage && <pre className="code-block">{opsMessage}</pre>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Stato servizio</h3>
          <button className="btn-secondary" onClick={onRefreshJobs}>Aggiorna jobs</button>
        </div>
        <pre className="code-block">{JSON.stringify(health, null, 2)}</pre>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Storico jobs</h3>
          <span className="pill">{jobsRows.length}</span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Job id</th>
                <th>Tipo</th>
                <th>Stato</th>
                <th>Durata (s)</th>
              </tr>
            </thead>
            <tbody>
              {jobsRows.map((row, idx) => (
                <tr key={`${row.job_id || "job"}-${idx}`}>
                  <td>{row.timestamp || "-"}</td>
                  <td>{row.job_id || "-"}</td>
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
