import { useEffect, useState } from "react";

function pad2(value) {
  return String(value).padStart(2, "0");
}

/**
 * Editor inline dell'orario/intervallo di un job (2026-09-09): la forma dei
 * campi dipende da `schedule_kind` (vedi `src/jobs/job_settings.py`) -
 * "daily" mostra un time picker HH:MM, "interval_minutes"/"interval_seconds"
 * un semplice input numerico. Il salvataggio ha effetto immediato lato
 * server (nessun restart), quindi qui basta chiamare `onSave`/`onReset` e
 * lasciare che la riga del job si aggiorni dalla risposta.
 */
function JobScheduleEditor({ job, saving, onSave, onReset }) {
  const schedule = job.schedule || {};
  const [timeValue, setTimeValue] = useState(
    schedule.hour !== undefined ? `${pad2(schedule.hour)}:${pad2(schedule.minute)}` : "00:00"
  );
  const [intervalValue, setIntervalValue] = useState(
    schedule.interval_minutes ?? schedule.interval_seconds ?? 0
  );
  const [localError, setLocalError] = useState("");

  useEffect(() => {
    if (job.schedule_kind === "daily") {
      setTimeValue(`${pad2(job.schedule?.hour ?? 0)}:${pad2(job.schedule?.minute ?? 0)}`);
    } else if (job.schedule_kind === "interval_minutes") {
      setIntervalValue(job.schedule?.interval_minutes ?? 0);
    } else if (job.schedule_kind === "interval_seconds") {
      setIntervalValue(job.schedule?.interval_seconds ?? 0);
    }
  }, [job.schedule_kind, job.schedule?.hour, job.schedule?.minute, job.schedule?.interval_minutes, job.schedule?.interval_seconds]);

  async function handleSaveDaily() {
    setLocalError("");
    const [hourStr, minuteStr] = timeValue.split(":");
    const hour = Number(hourStr);
    const minute = Number(minuteStr);
    if (Number.isNaN(hour) || Number.isNaN(minute)) {
      setLocalError("Orario non valido.");
      return;
    }
    try {
      await onSave(job.job_id, { hour, minute });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSaveInterval() {
    setLocalError("");
    const value = Number(intervalValue);
    if (Number.isNaN(value)) {
      setLocalError("Valore non valido.");
      return;
    }
    const key = job.schedule_kind === "interval_seconds" ? "interval_seconds" : "interval_minutes";
    try {
      await onSave(job.job_id, { [key]: value });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleReset() {
    setLocalError("");
    try {
      await onReset(job.job_id);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    }
  }

  const unitLabel = job.schedule_kind === "interval_seconds" ? "secondi" : "minuti";

  return (
    <div className="settings-job-schedule">
      {job.schedule_kind === "daily" ? (
        <input
          type="time"
          value={timeValue}
          disabled={saving}
          onChange={(e) => setTimeValue(e.target.value)}
          onBlur={handleSaveDaily}
        />
      ) : (
        <div className="settings-job-schedule-interval">
          <input
            type="number"
            min="1"
            value={intervalValue}
            disabled={saving}
            onChange={(e) => setIntervalValue(e.target.value)}
            onBlur={handleSaveInterval}
          />
          <span className="muted">{unitLabel}</span>
        </div>
      )}
      {!job.schedule_is_default && (
        <button className="btn-link" disabled={saving} onClick={handleReset} title="Torna al valore di default">
          ripristina default
        </button>
      )}
      {saving && <small className="muted">Salvataggio...</small>}
      {localError && <small className="error-text">{localError}</small>}
    </div>
  );
}

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
  scheduleSavingJobId,
  onSaveSchedule,
  onResetSchedule,
  runningJobId,
  runFeedback = {},
  onRunJob,
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
                {runFeedback[job.job_id] && (
                  <small className={runFeedback[job.job_id].status === "error" ? "error-text" : "muted"}>
                    {runFeedback[job.job_id].message}
                  </small>
                )}
              </div>
              {job.schedule_kind && onSaveSchedule && onResetSchedule && (
                <JobScheduleEditor
                  job={job}
                  saving={scheduleSavingJobId === job.job_id}
                  onSave={onSaveSchedule}
                  onReset={onResetSchedule}
                />
              )}
              {onRunJob && (
                <button
                  className="btn-secondary"
                  disabled={runningJobId === job.job_id}
                  title="Esegue il job subito, senza aspettare il prossimo giro schedulato (indipendente dal toggle enabled/disabled)"
                  onClick={() => onRunJob(job.job_id)}
                >
                  {runningJobId === job.job_id ? "Avvio..." : "Esegui ora"}
                </button>
              )}
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

