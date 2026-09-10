import PredictionBadges from "./PredictionBadges";
import ValueBadge from "./ValueBadge";
import { phaseClass, phaseLabel } from "../../shared/formatters";

export default function MatchTable({ rows, selectedFixtureId, onOpenMatch, onOpenOracleDetail, modelMarkets }) {
  if (!rows || rows.length === 0) {
    return <div className="empty-panel">Nessuna partita trovata per i filtri correnti.</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ora</th>
            <th>Torneo</th>
            <th>Match</th>
            <th>Score</th>
            <th>Stato</th>
            <th>Previsioni</th>
            <th>Value</th>
            <th>Dettaglio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
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
                {row.score?.home ?? "-"} - {row.score?.away ?? "-"}
              </td>
              <td>
                <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
              </td>
              <td><PredictionBadges row={row} modelMarkets={modelMarkets} /></td>
              <td><ValueBadge decision={row.best_decision} /></td>
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
          ))}
        </tbody>
      </table>
    </div>
  );
}
