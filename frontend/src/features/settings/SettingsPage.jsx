function quotaBarClass(percentage) {
  if (percentage === null || percentage === undefined) {
    return "";
  }
  if (percentage >= 90) {
    return "danger";
  }
  if (percentage >= 70) {
    return "warning";
  }
  return "ok";
}

/**
 * Pagina Impostazioni: attivazione/disattivazione dei job schedulati in
 * background (nessun restart necessario, effetto dal prossimo giro dello
 * scheduler - vedi `src/jobs/job_settings.py`) + barra di avanzamento della
 * quota giornaliera del provider esterno API-Sports.
 *
 * Il caricamento automatico (mount + polling) legge solo la cache locale
 * (`GET /settings/quota`, nessuna chiamata reale consumata) e puo' quindi
 * mostrare una STIMA disallineata dalla realta'. Il bottone "Aggiorna"
 * invece interroga DAVVERO API-Sports (`POST /settings/quota/refresh`,
 * endpoint ufficiale `/status`) e mostra il numero autoritativo, identico
 * a quello della dashboard account api-sports.io (bug diagnosticato
 * 2026-09-05: la vecchia versione poteva mostrare 0% con la quota reale
 * gia' al 100%, perche' non faceva mai un vero controllo).
 */
export default function SettingsPage({
  jobs = [],
  isLoading,
  error,
  savingJobId,
  onToggleJob,
  onRefreshJobs,
  quota,
  quotaLoading,
  quotaError,
  onRefreshQuota,
  quotaPaused,
  quotaPausedSince,
}) {
  const percentage = quota?.daily_used_percentage ?? null;
  const barWidth = Math.min(Math.max(percentage ?? 0, 0), 100);
  const barClass = quotaBarClass(percentage);
  const isLive = quota?.source === "live";

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Quota giornaliera API-Sports</h3>
          <button className="btn-secondary" onClick={onRefreshQuota} disabled={quotaLoading}>
            {quotaLoading ? "Verifica in corso..." : "Aggiorna"}
          </button>
        </div>
        <p className="muted">
          {isLive ? (
            <>
              Valore <strong>reale</strong>, letto direttamente da API-Sports (endpoint <code>/status</code>) al
              momento dell'ultimo click su "Aggiorna" - lo stesso numero della dashboard del tuo account.
            </>
          ) : (
            <>
              Valore <strong>stimato</strong> dall'ultima chiamata dati osservata (nessun controllo diretto ancora
              eseguito in questo deploy): puo' non riflettere consumi fatti fuori da questa app. Clicca "Aggiorna" per
              un controllo reale.
            </>
          )}
        </p>

        {quotaError && <div className="error-box">Errore: {quotaError}</div>}

        {quota && quota.available ? (
          <>
            <div className="quota-bar">
              <div className={`quota-bar-fill ${barClass}`} style={{ width: `${barWidth}%` }} />
            </div>
            <div className="quota-meta">
              <strong>{percentage !== null ? `${percentage}%` : "n/d"}</strong> consumata &middot;{" "}
              {quota.daily_used ?? "?"} / {quota.daily_limit ?? "?"} chiamate giornaliere ({quota.daily_remaining ?? "?"}{" "}
              rimanenti){!isLive && " (stima)"}
            </div>
            <small className="muted">
              Minuto corrente: {quota.minute_remaining ?? "?"}/{quota.minute_limit ?? "?"} rimanenti &middot; ultimo
              aggiornamento{" "}
              {quota.updated_at ? new Date(quota.updated_at).toLocaleString("it-IT") : "n/d"}
              {quota.message ? ` — ${quota.message}` : ""}
            </small>
          </>
        ) : (
          <div className="info-box">
            {quota?.message || "Nessuna chiamata API-Sports ancora registrata: clicca \"Aggiorna\" per un controllo reale."}
          </div>
        )}
      </section>


      <section className="panel">
        <div className="panel-header">
          <h3>Job automatici (scheduler)</h3>
          <button className="btn-secondary" onClick={onRefreshJobs} disabled={isLoading}>
            {isLoading ? "Aggiornamento..." : "Aggiorna"}
          </button>
        </div>
        <p className="muted">
          Attiva/disattiva i job che girano in background sul server. Un job disattivato NON viene eseguito finche' non
          lo riattivi: nessun restart di nessun servizio necessario, il cambiamento ha effetto dal prossimo giro dello
          scheduler.
        </p>

        {quotaPaused && (
          <div className="error-box">
            ⏸️ I job che chiamano API-Sports (Aggiorna tutto/Sync oggi/Sync futuro/Sync live) sono stati messi
            automaticamente in pausa il {quotaPausedSince || "oggi"}: la quota giornaliera risultava esaurita al 100%.
            Riprenderanno da soli non appena la quota si resetta (mezzanotte UTC) - un toggle manuale su questi job
            viene corretto di nuovo finche' la pausa e' attiva. Settlement e Retrain modelli non lavorano con
            API-Sports: restano sempre disponibili.
          </div>
        )}

        {error && <div className="error-box">Errore: {error}</div>}

        <div className="settings-jobs-list">
          {jobs.map((job) => (
            <div className="settings-job-row" key={job.job_id}>
              <div className="settings-job-info">
                <strong>
                  {job.label}
                  {job.calls_api_sports && (
                    <span className="pill quota-pill" title="Chiama il provider esterno API-Sports">
                      API-Sports
                    </span>
                  )}
                </strong>
                <small className="muted">{job.description}</small>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={Boolean(job.enabled)}
                  disabled={savingJobId === job.job_id}
                  onChange={(e) => onToggleJob(job.job_id, e.target.checked)}
                />
                <span className="switch-slider" />
              </label>
            </div>
          ))}
          {jobs.length === 0 && !isLoading && <div className="empty-panel">Nessun job trovato.</div>}
        </div>
      </section>
    </section>
  );
}

