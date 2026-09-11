import { formatDateIt, marketLabel } from "../shared/formatters";

export default function TopFilters({
  selectedDate,
  onChangeDate,
  availableDates,
  markets,
  selectedMarket,
  onChangeSelectedMarket,
  searchInput,
  onChangeSearchInput,
  onApplySearch,
  onForceRefresh,
  forceRefreshDisabled,
}) {
  const dateOptions = availableDates && availableDates.length > 0 ? availableDates : [selectedDate];

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
          <select value={selectedDate} onChange={(e) => onChangeDate(e.target.value)}>
            {dateOptions.map((iso) => (
              <option key={iso} value={iso}>
                {formatDateIt(iso)}
              </option>
            ))}
          </select>
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

        <button
          className="btn-secondary"
          onClick={onApplySearch}
          title="Applica soltanto il testo inserito nel campo Cerca match. Data e mercato si aggiornano automaticamente."
        >
          Cerca match
        </button>
        <button
          className="btn-secondary"
          onClick={onForceRefresh}
          disabled={forceRefreshDisabled}
          title="Casi eccezionali: ri-forza la sincronizzazione con il provider esterno anche per date gia' presenti a DB (es. correzione tardiva di quote/risultato)"
        >
           Forza aggiornamento
        </button>
      </div>
    </header>
  );
}
