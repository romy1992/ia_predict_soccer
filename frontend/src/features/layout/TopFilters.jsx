import { formatDateIt, marketLabel, todayIso } from "../shared/formatters";

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
  onRefreshDayPredictions,
  refreshingDayPredictions,
  dayPredictionsProgress,
  refreshDayPredictionsError,
}) {
  const dateOptions = availableDates && availableDates.length > 0 ? availableDates : [selectedDate];
  const today = todayIso();

  return (
    <header className="topbar">
      <div>
        <h1>Dashboard partite e previsioni</h1>
        <p>
          Vista giornaliera stile tennis_oracle con menu, live match e previsioni multi-mercato.
        </p>
      </div>

      <div className="filters">
        <label className="filter-card">
          Data
          <select value={selectedDate} onChange={(e) => onChangeDate(e.target.value)}>
            {dateOptions.map((iso) => (
              <option key={iso} value={iso} style={iso === today ? { fontWeight: "bold" } : undefined}>
                {formatDateIt(iso)}
                {iso === today ? " (Oggi)" : ""}
              </option>
            ))}
          </select>
        </label>

        <label className="filter-card">
          Mercato
          <select value={selectedMarket} onChange={(e) => onChangeSelectedMarket(e.target.value)}>
            {markets.map((item) => (
              <option key={item} value={item}>
                {item === "all" ? "Tutti i mercati" : marketLabel(item)}
              </option>
            ))}
          </select>
        </label>

        <form
          className="filter-card filter-search"
          onSubmit={(event) => {
            event.preventDefault();
            onApplySearch();
          }}
        >
          <label>
            Cerca match
            <span className="filter-search-controls">
              <input
                value={searchInput}
                onChange={(e) => onChangeSearchInput(e.target.value)}
                placeholder="Es: Inter, Premier, Real"
              />
              <button
                type="submit"
                className="btn-secondary"
                title="Applica soltanto il testo inserito nel campo Cerca match. Data e mercato si aggiornano automaticamente."
              >
                Cerca match
              </button>
            </span>
          </label>
        </form>

        <div className="filter-card filter-refresh">
          <span className="filter-label">Aggiornamento dati</span>
          <button
            className="btn-secondary"
            onClick={onForceRefresh}
            disabled={forceRefreshDisabled}
            title="Casi eccezionali: ri-forza la sincronizzazione con il provider esterno anche per date gia' presenti a DB (es. correzione tardiva di quote/risultato)"
          >
            Forza aggiornamento
          </button>
          <small>Usalo solo per richiedere nuovi dati al provider.</small>
        </div>

        {/* Distinto da "Forza aggiornamento" qui sopra: quello richiede dati
            NUOVI al provider esterno (consuma quota API), questo ricalcola
            solo le previsioni ML sui dati gia' a DB per la data selezionata
            (nessuna quota). Serve quando un mercato appena promosso a
            production lascia le partite gia' salvate senza riga per quel
            mercato, in attesa del giro schedulato. */}
        {onRefreshDayPredictions && (
          <div className="filter-card filter-refresh">
            <span className="filter-label">Previsioni del giorno</span>
            <button
              className="btn-secondary"
              onClick={onRefreshDayPredictions}
              disabled={refreshingDayPredictions}
              title="Ricalcola e salva subito le previsioni ML per tutte le partite della data selezionata, senza aspettare il giro automatico. Non chiama il provider esterno: nessuna quota API consumata."
            >
              {refreshingDayPredictions ? "Ricalcolo in corso..." : "Ricalcola previsioni"}
            </button>
            {refreshingDayPredictions && (
              <div
                className="progress-bar determinate"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(dayPredictionsProgress?.percent || 0)}
                aria-label="Avanzamento ricalcolo previsioni"
              >
                <div
                  className="progress-bar-value"
                  style={{ width: `${Math.max(4, dayPredictionsProgress?.percent || 0)}%` }}
                />
              </div>
            )}
            <small>
              {refreshDayPredictionsError
                ? `Errore: ${refreshDayPredictionsError}`
                : refreshingDayPredictions
                  ? dayPredictionsProgress?.total
                    ? `Ricalcolo ${dayPredictionsProgress.targetDate || ""}: ${dayPredictionsProgress.done}/${dayPredictionsProgress.total} partite (${Math.round(dayPredictionsProgress.percent)}%)`
                    : "Avvio ricalcolo..."
                  : "Popola i mercati ancora “In coda” per questo giorno."}
            </small>
          </div>
        )}
      </div>
    </header>
  );
}
