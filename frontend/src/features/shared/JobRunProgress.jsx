/**
 * Barra di avanzamento per un job lanciato manualmente e pollato via
 * `GET /jobs/{id}` (stesso meccanismo di "Esegui ora" in Impostazioni,
 * 2026-09-21 - riusato qui COSI' COM'E', nessuna logica di polling
 * duplicata: il chiamante aggiorna `row` con lo stesso formato di
 * `jobRunRows` in App.jsx).
 *
 * `row.summary.percent` esiste solo per i job che pubblicano avanzamento
 * incrementale (es. "Ricalcola previsioni del giorno"): quando assente,
 * mostra uno spinner indeterminato invece di un numero inventato.
 */
export default function JobRunProgress({ row }) {
  if (!row || !["queued", "running"].includes(row.status)) {
    return null;
  }
  const summary = row.summary || {};
  const percent = typeof summary.percent === "number" ? summary.percent : null;
  const dettaglio =
    percent !== null && summary.fixtures_total
      ? `${summary.fixtures_done ?? 0}/${summary.fixtures_total} (${Math.round(percent)}%)`
      : row.status === "queued"
        ? "In coda..."
        : "In esecuzione...";

  return (
    <div className="settings-job-progress">
      {percent !== null ? (
        <div
          className="progress-bar determinate"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(percent)}
          aria-label="Avanzamento job"
        >
          <div className="progress-bar-value" style={{ width: `${Math.max(4, percent)}%` }} />
        </div>
      ) : (
        <div className="progress-bar" role="progressbar" aria-label="Job in esecuzione">
          <div className="progress-bar-fill" />
        </div>
      )}
      <small className="muted">{dettaglio}</small>
    </div>
  );
}
