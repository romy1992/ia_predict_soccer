import { useMemo, useState } from "react";
import MatchTable from "../matches/components/MatchTable";
import PhaseTabs from "../matches/components/PhaseTabs";
import MarketTabs from "../matches/components/MarketTabs";
import { marketLabel, phaseClass, phaseLabel } from "../shared/formatters";

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
  const [situationFilter, setSituationFilter] = useState("all");
  const counters = useMemo(() => {
    const initial = { PLAY: 0, BORDERLINE: 0, "NO BET": 0, "SENZA QUOTA": 0, official: 0, WON: 0, LOST: 0, VOID: 0 };
    return safeRows.reduce((acc, row) => {
      const cards = selectedMarket === "all"
        ? (row.decision_cards || []).filter((card) => card.is_market_best !== false)
        : [row.best_decision].filter(Boolean);
      if (cards.length === 0) acc["SENZA QUOTA"] += 1;
      cards.forEach((card) => {
        const label = card?.value_label || "SENZA QUOTA";
        if (Object.prototype.hasOwnProperty.call(acc, label)) acc[label] += 1;
        if (card?.is_official) acc.official += 1;
        if (["WON", "LOST", "VOID"].includes(card?.official_outcome)) acc[card.official_outcome] += 1;
      });
      return acc;
    }, initial);
  }, [safeRows, selectedMarket]);
  const visibleRows = useMemo(
    () => situationFilter === "all"
      ? safeRows
      : safeRows.filter((row) => {
        const cards = selectedMarket === "all"
          ? (row.decision_cards || []).filter((card) => card.is_market_best !== false)
          : [row.best_decision].filter(Boolean);
        return cards.length === 0
          ? situationFilter === "SENZA QUOTA"
          : cards.some((card) => (card.value_label || "SENZA QUOTA") === situationFilter);
      }),
    [safeRows, selectedMarket, situationFilter],
  );

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
        <div className="match-center-controls">
          <label>
            Mercato
            <select value={selectedMarket} onChange={(event) => onChangeSelectedMarket(event.target.value)}>
              {(markets || ["all"]).map((market) => (
                <option key={market} value={market}>{market === "all" ? "Tutti i mercati" : marketLabel(market)}</option>
              ))}
            </select>
          </label>
          <label>
            Situazione
            <select value={situationFilter} onChange={(event) => setSituationFilter(event.target.value)}>
              <option value="all">Tutte</option>
              <option value="PLAY">PLAY</option>
              <option value="BORDERLINE">BORDERLINE</option>
              <option value="NO BET">NO BET</option>
              <option value="SENZA QUOTA">Senza quota</option>
            </select>
          </label>
        </div>
        <div className="decision-counters" aria-label="Contatori decisioni del mercato selezionato">
          {[
            ["PLAY", counters.PLAY],
            ["BORDERLINE", counters.BORDERLINE],
            ["NO BET", counters["NO BET"]],
            ["SENZA QUOTA", counters["SENZA QUOTA"]],
            ["UFFICIALI", counters.official],
            ["WON", counters.WON],
            ["LOST", counters.LOST],
            ["VOID", counters.VOID],
          ].map(([label, count]) => <span className="pill" key={label}>{label}: {count}</span>)}
        </div>

        <MatchTable
          rows={visibleRows}
          selectedFixtureId={selectedFixtureId}
          onOpenMatch={onOpenMatch}
          onOpenOracleDetail={onOpenOracleDetail}
          modelMarkets={dayData?.model_markets}
          selectedMarket={selectedMarket}
        />
      </section>
    </section>
  );
}
