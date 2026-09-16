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
