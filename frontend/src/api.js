const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const text = await response.text();

  let payload;
  try {
    payload = JSON.parse(text);
  } catch (_) {
    payload = text;
  }

  if (!response.ok) {
    const detail = typeof payload === "string" ? payload : JSON.stringify(payload);
    throw new Error(detail);
  }

  return payload;
}

export function getHealth() {
  return request("/health");
}

export function getMarkets() {
  return request("/markets");
}

export function getDashboardOverview(targetDate) {
  const params = new URLSearchParams();
  if (targetDate) {
    params.set("target_date", targetDate);
  }
  const query = params.toString();
  return request(`/dashboard/overview${query ? `?${query}` : ""}`);
}

export function getDashboardBundle({
  targetDate,
  limit = 400,
  search,
  forceRefresh = false,
} = {}) {
  const params = new URLSearchParams();
  if (targetDate) params.set("target_date", targetDate);
  params.set("limit", String(limit));
  if (search) params.set("search", search);
  if (forceRefresh) params.set("force_refresh", "true");
  return request(`/dashboard/bundle?${params.toString()}`);
}

export function getDashboardAvailableDates() {
  return request("/dashboard/available-dates");
}

export function getDashboardLive({ targetDate, limit = 20, withPredictions = true, markets } = {}) {
  const params = new URLSearchParams();
  if (targetDate) {
    params.set("target_date", targetDate);
  }
  if (limit) {
    params.set("limit", String(limit));
  }
  params.set("with_predictions", String(withPredictions));
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  return request(`/dashboard/live?${params.toString()}`);
}

export function getDashboardDay({
  targetDate,
  limit = 300,
  withPredictions = true,
  markets,
  phase,
  search,
  forceRefresh = false,
} = {}) {
  const params = new URLSearchParams();
  if (targetDate) {
    params.set("target_date", targetDate);
  }
  if (limit) {
    params.set("limit", String(limit));
  }
  params.set("with_predictions", String(withPredictions));
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  if (phase) {
    params.set("phase", phase);
  }
  if (search) {
    params.set("search", search);
  }
  if (forceRefresh) {
    params.set("force_refresh", "true");
  }
  return request(`/dashboard/day?${params.toString()}`);
}

export function getDashboardMatchDetail(fixtureId, { withPredictions = true, markets } = {}) {
  const params = new URLSearchParams();
  params.set("with_predictions", String(withPredictions));
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  return request(`/dashboard/match/${fixtureId}?${params.toString()}`);
}

export function recomputeMatchPredictions(fixtureId, { markets } = {}) {
  return request(`/dashboard/matches/${fixtureId}/recompute-predictions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ markets: markets && markets.length > 0 ? markets : null }),
  });
}

/**
 * Bottone "Ricalcola previsioni del giorno": accoda il job e torna subito
 * `job_id`. Il frontend polla `/jobs/{id}` (barra + resume dopo refresh).
 * Non chiama il provider esterno: nessuna quota API consumata.
 */
export function refreshDayPredictions(targetDate) {
  return request("/jobs/prediction-snapshot-refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_date: targetDate, async_run: true }),
  });
}

export function getJob(jobId) {
  return request(`/jobs/${encodeURIComponent(jobId)}`);
}

export function getOracleMatchDetail(fixtureId, { markets } = {}) {
  const params = new URLSearchParams();
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  const query = params.toString();
  return request(`/dashboard/match/${fixtureId}/oracle-detail${query ? `?${query}` : ""}`);
}

export function getBetslipGenerate({
  targetDate,
  includeBorderline = false,
  minOdd,
  maxOdd,
  minEv,
  markets,
} = {}) {
  const params = new URLSearchParams();
  if (targetDate) {
    params.set("target_date", targetDate);
  }
  params.set("include_borderline", String(includeBorderline));
  if (minOdd !== undefined && minOdd !== null && minOdd !== "") {
    params.set("min_odd", String(minOdd));
  }
  if (maxOdd !== undefined && maxOdd !== null && maxOdd !== "") {
    params.set("max_odd", String(maxOdd));
  }
  if (minEv !== undefined && minEv !== null && minEv !== "") {
    params.set("min_ev", String(minEv));
  }
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  return request(`/betslip/generate?${params.toString()}`);
}

export function saveBetslipGeneration({ targetDate } = {}) {
  const params = new URLSearchParams();
  if (targetDate) params.set("target_date", targetDate);
  return request(`/betslip/generate/snapshot?${params.toString()}`, {
    method: "POST",
  });
}

export function getSavedBetslipProposals({ targetDate, latestOnly = true, limit = 200 } = {}) {
  const params = new URLSearchParams();
  params.set("reference_date", targetDate);
  params.set("latest_only", String(latestOnly));
  params.set("limit", String(limit));
  return request(`/betslip/proposals?${params.toString()}`);
}

export function getOfficialBetslips({ targetDate, status, limit = 200 } = {}) {
  const params = new URLSearchParams();
  if (targetDate) params.set("reference_date", targetDate);
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  return request(`/betslip/official?${params.toString()}`);
}

export function getOfficialBetslipStatistics() {
  return request("/betslip/official/statistics");
}

export function getBettingStatistics({ days = 30 } = {}) {
  return request(`/betting/statistics?days=${encodeURIComponent(days)}`);
}

function normalizeJobBody(asyncRunOrPayload, fallbackPayload = {}) {
  if (typeof asyncRunOrPayload === "object" && asyncRunOrPayload !== null) {
    return asyncRunOrPayload;
  }
  return { ...fallbackPayload, async_run: Boolean(asyncRunOrPayload) };
}

export function triggerImport(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerTodayUpdate(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/today-update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerFutureSync(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/future-sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerDailyRefresh(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/daily-refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerSettlement(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/settlement", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerRetrain(asyncRunOrPayload = true, payload = {}) {
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/retrain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function triggerDataQualityReport(asyncRunOrPayload = true, payload = {}) {
  // Bottone "Aggiorna report" della pagina Data Quality: a differenza di
  // getDataQuality() (semplice GET), questa passa dal job "data_quality_report"
  // (loggato in storico job, stessa funzione del job schedulato omonimo -
  // vedi POST /jobs/data-quality-report in src/api/main.py). In modalita'
  // sincrona (async_run: false, uso di default lato App.jsx) la risposta
  // contiene gia' il report calcolato in `details`.
  const body = normalizeJobBody(asyncRunOrPayload, payload);
  return request("/jobs/data-quality-report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function predict(market, fixtureId) {
  return request(`/predict/${market}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fixture_id: Number(fixtureId) }),
  });
}

export function getMetrics(market, limit = 20) {
  return request(`/metrics/${market}?limit=${limit}`);
}

export function getJobs(limit = 50, { jobType, status } = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (jobType) {
    params.set("job_type", jobType);
  }
  if (status) {
    params.set("status", status);
  }
  return request(`/jobs/history?${params.toString()}`);
}

export function getDataQuality({ topN = 20, seasons, leagues } = {}) {
  const params = new URLSearchParams();
  params.set("top_n", String(topN));
  if (Array.isArray(seasons) && seasons.length > 0) {
    params.set("seasons", seasons.join(","));
  }
  if (Array.isArray(leagues) && leagues.length > 0) {
    params.set("leagues", leagues.join(","));
  }
  return request(`/data/quality?${params.toString()}`);
}

export function getPredictions(limit = 50) {
  return request(`/predictions/log?limit=${limit}`);
}

export function getMonitoringOverview({ market } = {}) {
  const params = new URLSearchParams();
  if (market && market !== "all") {
    params.set("market", market);
  }
  const query = params.toString();
  return request(`/monitoring/overview${query ? `?${query}` : ""}`);
}

export function getMonitoringAlerts({ market } = {}) {
  const params = new URLSearchParams();
  if (market && market !== "all") {
    params.set("market", market);
  }
  const query = params.toString();
  return request(`/monitoring/alerts${query ? `?${query}` : ""}`);
}

export function getOfficialPerformance({ market, days = 30 } = {}) {
  const params = new URLSearchParams();
  params.set("days", String(days));
  if (market && market !== "all") {
    params.set("market", market);
  }
  return request(`/predictions/official/performance?${params.toString()}`);
}

export function getJobSettings() {
  return request("/settings/jobs");
}

export function updateJobSettings(updates) {
  return request("/settings/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });
}

export function updateJobSchedule(jobId, schedule) {
  // Effetto immediato senza restart: lo scheduler rilegge lo schedule
  // effettivo ad ogni tick dell'heartbeat (vedi src/jobs/scheduler.py).
  return request(`/settings/jobs/${encodeURIComponent(jobId)}/schedule`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(schedule),
  });
}

export function resetJobSchedule(jobId) {
  return request(`/settings/jobs/${encodeURIComponent(jobId)}/schedule`, {
    method: "DELETE",
  });
}

export function getApiQuota() {
  return request("/settings/quota");
}

export function refreshApiQuota() {
  // A differenza di getApiQuota() (rilegge solo la cache locale), questa
  // interroga DAVVERO API-Sports (endpoint /status) - vedi
  // `POST /settings/quota/refresh` in src/api/main.py. Usata SOLO dal
  // click esplicito sul bottone "Aggiorna" di Impostazioni.
  return request("/settings/quota/refresh", { method: "POST" });
}

export function getModelDiagnostics({ markets, forceRefresh = false } = {}) {
  const params = new URLSearchParams();
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  if (forceRefresh) {
    params.set("force_refresh", "true");
  }
  const query = params.toString();
  return request(`/models/diagnostics${query ? `?${query}` : ""}`);
}

export { API_BASE_URL };




