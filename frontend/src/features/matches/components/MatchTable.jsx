import {
  formatOdd,
  formatPercent,
  formatPercentagePoints,
  formatSignedNumber,
  phaseClass,
  phaseLabel,
  valueClass,
} from "../../shared/formatters";
import PredictionBadges from "./PredictionBadges";

export default function MatchTable({
  rows,
  selectedFixtureId,
  onOpenMatch,
  onOpenOracleDetail,
  modelMarkets,
  selectedMarket,
}) {
  if (!rows || rows.length === 0) {
    return <div className="empty-panel">Nessuna partita trovata per i filtri correnti.</div>;
  }

  const showAllMarkets = !selectedMarket || selectedMarket === "all";

  return (
    <>
    <div className="table-wrap match-center-desktop">
      <table className="match-center-table">
        <thead>
          <tr>
            <th>Ora</th>
            <th>Torneo</th>
            <th>Partita</th>
            <th>Score/Stato</th>
            {showAllMarkets && <th>Tutti i mercati</th>}
            <th>{showAllMarkets ? "Pronostico vincitore" : "Pronostico"}</th>
            <th>Probabilità IA</th>
            <th>Quota mercato</th>
            <th title="Quota di pareggio economico del modello: 1 / probabilità IA. Non è lo stato VOID di una giocata.">Quota void IA</th>
            <th title="Differenza assoluta tra quota mercato e quota void IA. Nel dettaglio è disponibile anche l'edge percentuale.">Edge</th>
            <th title="Rendimento teorico della singola selezione; non è il ROI realmente ottenuto.">ROI atteso</th>
            <th>Situazione</th>
            <th title="Stato della PLAY congelata dal job prima del calcio d'inizio: PENDING, WON, LOST o VOID. Se non è stata registrata, non contribuisce al ROI ufficiale.">Giocata ufficiale</th>
            <th>Dettaglio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const decision = row.best_decision;
            return (
            <tr
              key={`match-${row.fixture_id}`}
              className={selectedFixtureId === row.fixture_id ? "row-selected" : ""}
            >
              <td>{row.time}</td>
              <td>{row.league || "-"}</td>
              <td>
                <div className="match-title">{row.home} vs {row.away}</div>
                <div className="match-sub">Fixture {row.fixture_id}</div>
              </td>
              <td>
                <div>{row.score?.home ?? "-"} - {row.score?.away ?? "-"}</div>
                <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
              </td>
              {showAllMarkets && <td className="all-markets-cell"><PredictionBadges row={row} modelMarkets={modelMarkets} /></td>}
              <td>{decision?.pick || "N/D"}</td>
              <td>{decision?.predicted_probability == null ? "N/D" : formatPercent(decision.predicted_probability)}</td>
              <td>{decision?.market_odd == null ? "N/D" : formatOdd(decision.market_odd)}</td>
              <td>{decision?.model_void_odd == null ? "N/D" : formatOdd(decision.model_void_odd)}</td>
              <td title={decision?.odds_edge_percent == null ? "" : `Edge percentuale: ${formatPercentagePoints(decision.odds_edge_percent)}`}>
                {decision?.odds_edge_absolute == null ? "N/D" : formatSignedNumber(decision.odds_edge_absolute)}
              </td>
              <td>{decision?.expected_roi_percent == null ? "N/D" : formatPercentagePoints(decision.expected_roi_percent)}</td>
              <td>
                <span className={`value-badge ${valueClass(decision?.value_label || "SENZA QUOTA")}`} title={decision?.value_reason || "Quota mercato non disponibile"}>
                  {decision?.value_label || "SENZA QUOTA"}
                </span>
              </td>
              <td>
                <strong>{decision?.is_official ? decision.official_outcome : "Non registrata"}</strong>
                {decision?.is_official && decision?.pnl != null && <small className="official-pnl">PnL {formatSignedNumber(decision.pnl)}</small>}
              </td>
              <td>
                <div className="cell-actions">
                  <button className="btn-secondary" onClick={() => onOpenMatch(row.fixture_id)}>
                    Apri
                  </button>
                  {onOpenOracleDetail && (
                    <button className="btn-secondary" onClick={() => onOpenOracleDetail(row.fixture_id)}>
                      Oracle
                    </button>
                  )}
                </div>
              </td>
            </tr>
          )})}
        </tbody>
      </table>
    </div>
    <div className="match-center-mobile">
      {rows.map((row) => {
        const decision = row.best_decision;
        return (
          <article className="match-mobile-card" key={`mobile-${row.fixture_id}`}>
            <div className="match-mobile-head">
              <strong>{row.home} vs {row.away}</strong>
              <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
            </div>
            {showAllMarkets && (
              <div className="match-mobile-all-markets">
                <span>Tutti i mercati</span>
                <PredictionBadges row={row} modelMarkets={modelMarkets} />
              </div>
            )}
            <dl>
              <div><dt>{showAllMarkets ? "Pronostico vincitore" : "Pronostico"}</dt><dd>{decision?.pick || "N/D"}</dd></div>
              <div><dt>Probabilità IA</dt><dd>{decision?.predicted_probability == null ? "N/D" : formatPercent(decision.predicted_probability)}</dd></div>
              <div><dt>Quota mercato</dt><dd>{decision?.market_odd == null ? "N/D" : formatOdd(decision.market_odd)}</dd></div>
              <div><dt>Quota void IA</dt><dd>{decision?.model_void_odd == null ? "N/D" : formatOdd(decision.model_void_odd)}</dd></div>
              <div><dt>Edge</dt><dd>{decision?.odds_edge_absolute == null ? "N/D" : formatSignedNumber(decision.odds_edge_absolute)}</dd></div>
              <div><dt>ROI atteso</dt><dd>{decision?.expected_roi_percent == null ? "N/D" : formatPercentagePoints(decision.expected_roi_percent)}</dd></div>
              <div><dt>Situazione</dt><dd><span className={`value-badge ${valueClass(decision?.value_label || "SENZA QUOTA")}`}>{decision?.value_label || "SENZA QUOTA"}</span></dd></div>
              <div><dt>Giocata ufficiale</dt><dd>{decision?.is_official ? decision.official_outcome : "Non registrata"}</dd></div>
            </dl>
            <button className="btn-secondary" onClick={() => onOpenMatch(row.fixture_id)}>Apri dettaglio</button>
          </article>
        );
      })}
    </div>
    </>
  );
}
