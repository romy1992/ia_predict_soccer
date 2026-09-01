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

export function triggerImport(asyncRun = true) {
  return request("/jobs/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ async_run: asyncRun }),
  });
}

export function triggerRetrain(asyncRun = true) {
  return request("/jobs/retrain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ async_run: asyncRun }),
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

export function getJobs(limit = 50) {
  return request(`/jobs/history?limit=${limit}`);
}

export function getPredictions(limit = 50) {
  return request(`/predictions/log?limit=${limit}`);
}

export { API_BASE_URL };


