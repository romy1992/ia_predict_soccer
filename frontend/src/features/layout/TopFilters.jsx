import { marketLabel } from "../shared/formatters";

export default function TopFilters({
  selectedDate,
  onChangeDate,
  markets,
  selectedMarket,
  onChangeSelectedMarket,
  searchInput,
  onChangeSearchInput,
  onApplySearch,
}) {
  return (
    <header className="topbar">
      <div>
        <h1>Dashboard partite e previsioni</h1>
        <p>
          Vista giornaliera stile tennis_oracle con menu, live match e previsioni multi-mercato.
        </p>
      </div>

      <div className="filters">
        <label>
          Data
          <input type="date" value={selectedDate} onChange={(e) => onChangeDate(e.target.value)} />
        </label>

        <label>
          Mercato
          <select value={selectedMarket} onChange={(e) => onChangeSelectedMarket(e.target.value)}>
            {markets.map((item) => (
              <option key={item} value={item}>
                {item === "all" ? "Tutti i mercati" : marketLabel(item)}
              </option>
            ))}
          </select>
        </label>

        <label>
          Cerca match
          <input
            value={searchInput}
            onChange={(e) => onChangeSearchInput(e.target.value)}
            placeholder="Es: Inter, Premier, Real"
          />
        </label>

        <button className="btn-secondary" onClick={onApplySearch}>Cerca</button>
      </div>
    </header>
  );
}

