import { MENU_ITEMS } from "../shared/menu";

export default function Sidebar({ activePage, onNavigate, onRefreshAll, lastRefresh, healthStatus }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <h2>soccer_oracle</h2>
        <p>Live center + predizioni</p>
      </div>

      <button className="btn-primary full" onClick={onRefreshAll}>Aggiorna tutto</button>

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

