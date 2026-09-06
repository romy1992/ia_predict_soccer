import MatchTable from "../matches/components/MatchTable";
import PhaseTabs from "../matches/components/PhaseTabs";
import MarketTabs from "../matches/components/MarketTabs";
import { phaseClass, phaseLabel } from "../shared/formatters";

export default function DashboardPage({
  overview,
  liveData,
  dayData,
  onOpenMatch,
  onOpenOracleDetail,
  selectedFixtureId,
  phaseFilter,
  onChangePhaseFilter,
  markets,
  selectedMarket,
  onChangeSelectedMarket,
  isFilterLoading,
}) {
  const safeRows = dayData?.rows || [];

  return (
    <section className="stack">
      <div className="stats-grid">
        <article className="stat-card">
          <span>Partite del giorno</span>
          <strong>{overview?.counts?.total ?? 0}</strong>
        </article>
        <article className="stat-card">
          <span>In diretta</span>
          <strong>{overview?.counts?.live ?? 0}</strong>
        </article>
        <article className="stat-card">
          <span>Da giocare</span>
          <strong>{overview?.counts?.to_play ?? 0}</strong>
        </article>
        <article className="stat-card">
          <span>Mercati con modello</span>
          <strong>{overview?.model_markets?.length ?? 0}</strong>
        </article>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h3>Partite in diretta</h3>
          <span className="pill">{liveData.returned}/{liveData.total}</span>
        </div>
        <div className="live-grid">
          {(liveData.rows || []).slice(0, 8).map((row) => (
            <article className="live-card live-clickable" key={`live-${row.fixture_id}`} onClick={() => onOpenMatch(row.fixture_id)}>
              <div className="live-head">
                <span className={`phase-badge ${phaseClass(row.phase)}`}>{phaseLabel(row.phase)}</span>
                <small>{row.time}</small>
              </div>
              <h4>{row.home} vs {row.away}</h4>
              <p className="score-big">{row.score?.home ?? "-"} - {row.score?.away ?? "-"}</p>
              <small>{row.league || "-"}</small>
            </article>
          ))}
          {(liveData.rows || []).length === 0 && <div className="empty-panel">Nessuna partita live al momento.</div>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3>Partite del giorno</h3>
          <span className="pill">{dayData.returned}/{dayData.total}</span>
          {isFilterLoading && <span className="pill pill-loading">Aggiornamento...</span>}
        </div>

        <PhaseTabs phases={["all", "to_play", "live", "finished"]} value={phaseFilter} onChange={onChangePhaseFilter} />
        <MarketTabs markets={markets} value={selectedMarket} onChange={onChangeSelectedMarket} />

        <MatchTable rows={safeRows} selectedFixtureId={selectedFixtureId} onOpenMatch={onOpenMatch} onOpenOracleDetail={onOpenOracleDetail} />
      </section>
    </section>
  );
}
