import PredictionBadges from "./PredictionBadges";
import { phaseClass, phaseLabel } from "../../shared/formatters";

export default function MatchTable({ rows, selectedFixtureId, onOpenMatch }) {
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
              <td><PredictionBadges row={row} /></td>
              <td>
                <button className="btn-secondary" onClick={() => onOpenMatch(row.fixture_id)}>
                  Apri
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

