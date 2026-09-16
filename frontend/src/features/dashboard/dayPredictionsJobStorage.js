export const DAY_PREDICTIONS_JOB_STORAGE_KEY = "dashboard.dayPredictionsJob";

export function readDayPredictionsJob() {
  try {
    const raw = window.localStorage.getItem(DAY_PREDICTIONS_JOB_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || !parsed.jobId) {
      return null;
    }
    return parsed;
  } catch (_) {
    return null;
  }
}

export function writeDayPredictionsJob(payload) {
  if (!payload || !payload.jobId) {
    window.localStorage.removeItem(DAY_PREDICTIONS_JOB_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(DAY_PREDICTIONS_JOB_STORAGE_KEY, JSON.stringify(payload));
}

/** Ricostruisce lo stato React dal localStorage, inclusa l'ultima
 *  percentuale nota: al refresh la barra non deve tornare a 0 in attesa
 *  del primo poll. */
export function jobSnapshotFromStorage(stored) {
  if (!stored || !stored.jobId) {
    return null;
  }
  const targetDate = stored.targetDate || stored.summary?.target_date || "";
  return {
    job_id: stored.jobId,
    status: stored.status || "running",
    params: { target_date: targetDate },
    summary: stored.summary || {
      target_date: targetDate,
      fixtures_total: 0,
      fixtures_done: 0,
      percent: 0,
    },
  };
}

export function persistDayPredictionsJob(job) {
  if (!job || !job.job_id) {
    writeDayPredictionsJob(null);
    return;
  }
  writeDayPredictionsJob({
    jobId: job.job_id,
    targetDate: job.params?.target_date || job.summary?.target_date || "",
    status: job.status,
    summary: job.summary || {},
  });
}

export function dayPredictionsProgressFromJob(job) {
  const summary = job?.summary || {};
  const total = Number(summary.fixtures_total);
  const done = Number(summary.fixtures_done);
  const percentRaw = Number(summary.percent);
  const percent = Number.isFinite(percentRaw)
    ? Math.max(0, Math.min(100, percentRaw))
    : Number.isFinite(total) && total > 0 && Number.isFinite(done)
      ? Math.max(0, Math.min(100, (100 * done) / total))
      : 0;
  return {
    percent,
    done: Number.isFinite(done) ? done : 0,
    total: Number.isFinite(total) ? total : 0,
    status: job?.status || "",
    targetDate: summary.target_date || job?.params?.target_date || "",
  };
}
