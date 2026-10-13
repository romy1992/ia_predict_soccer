import { formatPercent, marketLabel } from "../shared/formatters";

export default function PredictionsPage({
  markets,
  manualFixtureId,
  onChangeFixtureId,
  manualMarket,
  onChangeManualMarket,
  onManualPredict,
  predictOutput,
  predictionRows,
  onRefreshPredictions,
}) {
  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Crea previsione manuale</h3>
        </div>
        <div className="inline-form">
          <label>
            Fixture id
            <input value={manualFixtureId} onChange={(e) => onChangeFixtureId(e.target.value)} />
          </label>
          <label>
            Mercato
            <select value={manualMarket} onChange={(e) => onChangeManualMarket(e.target.value)}>
              {markets.filter((item) => item !== "all").map((item) => (
                <option key={item} value={item}>{marketLabel(item)}</option>
              ))}
            </select>
          </label>
          <button className="btn-primary" onClick={onManualPredict}>Calcola previsione</button>
        </div>
        {predictOutput && <pre className="code-block">{predictOutput}</pre>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Storico previsioni</h3>
          <button className="btn-secondary" onClick={onRefreshPredictions}>Aggiorna storico</button>
        </div>

        <div className="table-wrap">
          <table className="table-cards">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Fixture</th>
                <th>Mercato</th>
                <th>Prediction</th>
                <th>Probabilita</th>
              </tr>
            </thead>
            <tbody>
              {predictionRows.map((row, idx) => (
                <tr key={`prediction-${idx}`}>
                  <td data-label="Timestamp">{row.timestamp || "-"}</td>
                  <td data-label="Fixture">{row.fixture_id}</td>
                  <td data-label="Mercato">{marketLabel(row.market)}</td>
                  <td data-label="Prediction">{String(row.prediction)}</td>
                  <td data-label="Probabilita">{formatPercent(row.probability)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
