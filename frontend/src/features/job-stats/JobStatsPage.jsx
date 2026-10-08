import { useState } from "react";

const JOB_TYPE_OPTIONS = [
  { value: "", label: "Tutti i tipi" },
  { value: "import", label: "import" },
  { value: "daily_refresh", label: "daily_refresh" },
  { value: "today_update", label: "today_update" },
  { value: "future_sync", label: "future_sync" },
  { value: "settlement", label: "settlement" },
];

export default function JobStatsPage({ report, isLoading, error, onLoadReport }) {
  const [days, setDays] = useState("30");
  const [jobType, setJobType] = useState("");
  const [localError, setLocalError] = useState("");

  async function handleLoad() {
    const parsedDays = Number(days);
    if (!Number.isInteger(parsedDays) || parsedDays < 1) {
      setLocalError("days deve essere un intero >= 1");
      return;
    }
    setLocalError("");
    try {
      await onLoadReport({ days: parsedDays, jobType: jobType || undefined });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setLocalError(message);
    }
  }

  const dayRows = report?.days || [];

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Storico Import</h3>
          <button className="btn-primary" onClick={handleLoad} disabled={isLoading}>Aggiorna</button>
        </div>
        <p className="muted">
          Andamento giorno per giorno dei job di import (fixture inserite/aggiornate/saltate/fallite) - a
          differenza di Data Quality (stato attuale cumulativo del DB), qui si vede l'esecuzione dei singoli
          job nel tempo.
        </p>

        <div className="inline-form">
          <label>
            Giorni
            <select value={days} onChange={(e) => setDays(e.target.value)}>
              <option value="7">7</option>
              <option value="30">30</option>
              <option value="90">90</option>
            </select>
          </label>
          <label>
            Tipo job
            <select value={jobType} onChange={(e) => setJobType(e.target.value)}>
              {JOB_TYPE_OPTIONS.map((option) => (
                <option key={option.value || "all"} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
        </div>

        {localError && <div className="error-box">Errore: {localError}</div>}
        {error && <div className="error-box">Errore API: {error}</div>}
        {isLoading && <div className="info-box">Caricamento storico...</div>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Per giorno</h3>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Data</th>
                <th>Esecuzioni</th>
                <th>Saltate (lock)</th>
                <th>Inserite</th>
                <th>Aggiornate</th>
                <th>Scartate</th>
                <th>Fallite</th>
              </tr>
            </thead>
            <tbody>
              {dayRows.map((row) => (
                <tr key={`job-day-${row.date}`}>
                  <td>{row.date}</td>
                  <td>{row.runs}</td>
                  <td>{row.locked_skips}</td>
                  <td>{row.inserted}</td>
                  <td>{row.updated}</td>
                  <td>{row.skipped}</td>
                  <td>{row.failed}</td>
                </tr>
              ))}
              {dayRows.length === 0 && (
                <tr>
                  <td colSpan={7} className="muted">Nessun job nella finestra selezionata.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
