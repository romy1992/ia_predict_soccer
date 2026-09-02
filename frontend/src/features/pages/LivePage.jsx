import MatchTable from "../matches/components/MatchTable";

export default function LivePage({ liveData, selectedFixtureId, onOpenMatch }) {
  return (
    <section className="panel stack">
      <div className="panel-header">
        <h3>Diretta completa</h3>
        <span className="pill">{liveData.returned}/{liveData.total}</span>
      </div>
      <MatchTable rows={liveData.rows || []} selectedFixtureId={selectedFixtureId} onOpenMatch={onOpenMatch} />
    </section>
  );
}

