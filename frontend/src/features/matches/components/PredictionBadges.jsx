import { confidenceClass, formatPercent, marketLabel, predictionLabel } from "../../shared/formatters";

export default function PredictionBadges({ row, modelMarkets }) {
  const predictions = row?.predictions || {};
  const entries = Object.entries(predictions);

  // Mercati attesi (registrati) ma assenti dal payload: con la vista
  // storica ora "sempre veloce" (2026-09-10, `allow_compute=False`), un
  // mercato senza riga gia' salvata NON innesca piu' un calcolo al volo -
  // semplicemente non compare tra `predictions`. Senza un placeholder
  // esplicito questo risultava ambiguo (badge mancante == mercato non
  // supportato? errore? in coda?) - qui rendiamo visibile la differenza
  // tra "nessun mercato disponibile per questa fixture" e "mercato X non
  // ancora coperto dalla banca dati".
  const pendingMarkets = (modelMarkets || []).filter((marketKey) => !(marketKey in predictions));

  if (entries.length === 0 && pendingMarkets.length === 0) {
    return <span className="empty-state">Nessuna previsione</span>;
  }

  return (
    <div className="prediction-badges">
      {entries.map(([marketKey, payload]) => (
        <span className={`prediction-chip ${confidenceClass(payload.probability)}`} key={`${row.fixture_id}-${marketKey}`}>
          <strong>{marketLabel(marketKey)}</strong>
          <em>{predictionLabel(marketKey, payload.prediction, row)}</em>
          <small>{formatPercent(payload.probability)}</small>
        </span>
      ))}
      {pendingMarkets.map((marketKey) => (
        <span className="prediction-chip prediction-pending" key={`${row.fixture_id}-${marketKey}-pending`}>
          <strong>{marketLabel(marketKey)}</strong>
          <em>In coda</em>
        </span>
      ))}
    </div>
  );
}
