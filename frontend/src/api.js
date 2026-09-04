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

export function getDashboardDay({ targetDate, limit = 300, withPredictions = true, markets, phase, search } = {}) {
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

export function getOracleMatchDetail(fixtureId, { markets } = {}) {
  const params = new URLSearchParams();
  if (markets && markets.length > 0) {
    params.set("markets", markets.join(","));
  }
  const query = params.toString();
  return request(`/dashboard/match/${fixtureId}/oracle-detail${query ? `?${query}` : ""}`);
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

export { API_BASE_URL };




