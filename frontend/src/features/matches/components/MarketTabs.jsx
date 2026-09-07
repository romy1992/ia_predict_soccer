import { marketLabel } from "../../shared/formatters";

/**
 * Tab di filtro per mercato di previsione: un tab per ogni mercato con
 * modello registrato, piu' "Tutti i mercati". Cambiare tab aggiorna lo
 * stesso stato globale `selectedMarket` gia' usato dal dropdown "Mercato"
 * in TopFilters, cosi' restano sempre sincronizzati.
 */
export default function MarketTabs({ markets, value, onChange }) {
  const items = markets && markets.length > 0 ? markets : ["all"];
  return (
    <div className="tabs tabs-market">
      {items.map((item) => (
        <button
          key={item}
          type="button"
          className={value === item ? "tab active" : "tab"}
          onClick={() => onChange(item)}
        >
          {item === "all" ? "Tutti i mercati" : marketLabel(item)}
        </button>
      ))}
    </div>
  );
}
