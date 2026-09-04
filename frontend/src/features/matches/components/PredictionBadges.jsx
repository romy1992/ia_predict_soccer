import { confidenceClass, formatPercent, marketLabel, predictionLabel } from "../../shared/formatters";

export default function PredictionBadges({ row }) {
  const predictions = row?.predictions || {};
  const entries = Object.entries(predictions);
  if (entries.length === 0) {
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
    </div>
  );
}
