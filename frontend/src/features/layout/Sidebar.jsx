import { MENU_ITEMS } from "../shared/menu";

// Riepilogo esito job "Aggiorna tutto" (run_daily_refresh: ieri + prossimi
// giorni) - visibile da QUALUNQUE pagina dato che la Sidebar resta sempre
// montata (a differenza del vecchio banner, confinato al solo Data Center).
function RefreshBanner({ refreshJobRow, isRefreshing }) {
  if (isRefreshing) {
    return (
      <div className="job-status-banner">
        <span className="spinner spinner-dark" />
        Job in corso: puoi continuare a navigare, si aggiorna da solo.
      </div>
    );
  }
  if (!refreshJobRow) {
    return null;
  }
  if (refreshJobRow.status === "failed") {
    return (
      <div className="job-status-banner failed">
        ❌ Aggiornamento fallito: {refreshJobRow.error?.message || "errore sconosciuto"}
      </div>
    );
  }
  if (refreshJobRow.status === "success") {
    const summary = refreshJobRow.summary || {};
    const quotaExceeded = summary.played_matches?.quota_exceeded || summary.upcoming_matches?.quota_exceeded;
    if (quotaExceeded) {
      return (
        <div className="job-status-banner failed">
          ⚠️ Quota API-Sports esaurita: alcuni campionati non aggiornati. Riprova piu' tardi/domani.
        </div>
      );
    }
    const played = summary.played_matches || {};
    const upcoming = summary.upcoming_matches || {};
    return (
      <div className="job-status-banner success">
        ✅ Ieri: {played.inserted ?? 0} nuove/{played.updated ?? 0} agg. · Prossimi: {upcoming.inserted ?? 0} nuove/
        {upcoming.updated ?? 0} agg.
      </div>
    );
  }
  return null;
}

export default function Sidebar({
  activePage,
  onNavigate,
  onRefreshAll,
  lastRefresh,
  healthStatus,
  isRefreshing,
  refreshJobRow,
  quotaExhausted,
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <h2>soccer_oracle</h2>
        <p>Live center + predizioni</p>
      </div>

      <button className="btn-primary full" onClick={onRefreshAll} disabled={isRefreshing || quotaExhausted}>
        {isRefreshing ? (
          <>
            <span className="spinner" />
            Aggiornamento in corso...
          </>
        ) : (
          "Aggiorna tutto"
        )}
      </button>
      {quotaExhausted && !isRefreshing ? (
        <small className="sidebar-refresh-hint quota-exhausted-hint">
          ⚠️ Quota API-Sports al 100%: riprova dopo il reset di mezzanotte UTC.
        </small>
      ) : (
        <small className="sidebar-refresh-hint">
          Ieri (risultati+quote) + prossimi 7 giorni, tutti i campionati censiti.
        </small>
      )}

      {isRefreshing && (
        <div className="progress-bar">
          <div className="progress-bar-fill" />
        </div>
      )}
      <RefreshBanner refreshJobRow={refreshJobRow} isRefreshing={isRefreshing} />

      <div className="sidebar-meta">
        <small>Ultimo update: {lastRefresh || "-"}</small>
        <small>API: {healthStatus || "offline"}</small>
      </div>

      <nav className="menu">
        {MENU_ITEMS.map((item) => (
          <button
            key={item.id}
            className={activePage === item.id ? "menu-item active" : "menu-item"}
            onClick={() => onNavigate(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
}
