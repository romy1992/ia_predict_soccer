import MatchTable from "../matches/components/MatchTable";
import { phaseLabel } from "../shared/formatters";

export default function TodayPage({ dayData, safeRows, phaseFilter, onChangePhaseFilter, selectedFixtureId, onOpenMatch, onOpenOracleDetail }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Match Center</h3>
        <span className="pill">{dayData.returned}/{dayData.total}</span>
      </div>

      <div className="tabs">
        {["all", "to_play", "live", "finished"].map((item) => (
          <button
            key={item}
            className={phaseFilter === item ? "tab active" : "tab"}
            onClick={() => onChangePhaseFilter(item)}
          >
            {item === "all" ? "Tutte" : phaseLabel(item)}
          </button>
        ))}
      </div>

      <MatchTable rows={safeRows} selectedFixtureId={selectedFixtureId} onOpenMatch={onOpenMatch} onOpenOracleDetail={onOpenOracleDetail} />
    </section>
  );
}
