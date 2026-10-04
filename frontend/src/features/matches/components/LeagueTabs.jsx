/**
 * Tab di filtro per campionato: un tab per ogni campionato presente tra le
 * partite della giornata caricata, piu' "Tutti i campionati". Stesso
 * pattern di `MarketTabs` (stato controllato dal genitore), ma la lista
 * viene dai dati del giorno (`row.league`) invece che da un catalogo
 * statico di mercati.
 */
export default function LeagueTabs({ leagues, value, onChange }) {
  if (!leagues || leagues.length === 0) {
    return null;
  }
  return (
    <div className="tabs tabs-league">
      <button
        type="button"
        className={value === "all" ? "tab active" : "tab"}
        onClick={() => onChange("all")}
      >
        Tutti i campionati
      </button>
      {leagues.map((league) => (
        <button
          key={league}
          type="button"
          className={value === league ? "tab active" : "tab"}
          onClick={() => onChange(league)}
        >
          {league}
        </button>
      ))}
    </div>
  );
}
