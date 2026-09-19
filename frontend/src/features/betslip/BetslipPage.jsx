import { useMemo, useState } from "react";
import ModelLegend from "../shared/ModelLegend";
import { formatNumber, formatOdd, formatPercent, marketLabel, todayIso } from "../shared/formatters";

/**
 * SLIP-03: Schedina Oracle — schedine 2/3/4 eventi generate dal backend
 * (`GET /betslip/generate`) per i tre profili di rischio (Safe/Balanced/
 * Aggressive). Nessuna logica scientifica qui: quota combinata,
 * probabilita' (naive/corretta per la correlazione), EV e risk score sono
 * gia' calcolati da `betslip_builder.py` - il frontend si limita a
 * formattare/visualizzare i valori ricevuti, con un unico arricchimento
 * "cosmetico" (nessun calcolo): il nome delle squadre per fixture_id,
 * gia' disponibile in `dayData` (stesso giorno, nessuna nuova fetch).
 */

const PROFILE_ORDER = ["SAFE", "BALANCED", "AGGRESSIVE"];
const PROFILE_LABELS = {
  SAFE: "Prudente",
  BALANCED: "Bilanciata",
  AGGRESSIVE: "Spinta",
};
const PROFILE_HELP = {
  SAFE: "2 eventi, probabilità più alte e soglie più restrittive.",
  BALANCED: "2 o 3 eventi, equilibrio tra probabilità e quota.",
  AGGRESSIVE: "3 o 4 eventi e rischio maggiore; ogni selezione resta comunque PLAY.",
};

function riskClass(riskLabel) {
  if (riskLabel === "LOW") {
    return "slip-risk-low";
  }
  if (riskLabel === "HIGH") {
    return "slip-risk-high";
  }
  return "slip-risk-medium";
}

function legMatchLabel(leg, fixtureIndex) {
  const info = fixtureIndex[leg.fixture_id];
  if (leg.home_team || leg.away_team) {
    return `${leg.home_team || "Casa"} vs ${leg.away_team || "Trasferta"}`;
  }
  if (!info) {
    return `Fixture #${leg.fixture_id}`;
  }
  return `${info.home} vs ${info.away}`;
}

function statusClass(status) {
  if (status === "PLAY" || status === "WON") return "value-play";
  if (status === "BORDERLINE") return "value-borderline";
  if (status === "NO BET" || status === "LOST") return "value-no-bet";
  return "value-unavailable";
}

function legStatusLabel(status, isOfficial, isShadow = false) {
  if (!isOfficial && !isShadow) return "Proposta";
  const labels = { PENDING: "In corso", WON: "Vinta", LOST: "Persa", VOID: "Rimborsata" };
  const label = labels[status] || status || "In corso";
  return !isOfficial && isShadow ? `Sim. ${label}` : label;
}

export default function BetslipPage({
  targetDate,
  onChangeTargetDate,
  report,
  isLoading,
  error,
  onLoadReport,
  dayData,
  bettingStatistics,
  bettingStatsDays,
  onLoadStatistics,
}) {
  const [activeView, setActiveView] = useState("proposals");
  const [activeDecision, setActiveDecision] = useState("PLAY");
  const [activeFilter, setActiveFilter] = useState("all");
  const [stake, setStake] = useState(10);
  const [copiedSlipId, setCopiedSlipId] = useState(null);
  const isPastDate = targetDate < todayIso();

  const fixtureIndex = useMemo(() => {
    const map = {};
    (dayData?.rows || []).forEach((row) => {
      map[row.fixture_id] = row;
    });
    return map;
  }, [dayData]);

  const profiles = report?.profiles || {};
  const proposalGroups = useMemo(() => {
    const groups = report?.decision_groups || { PLAY: profiles };
    return Object.fromEntries(
      ["PLAY", "BORDERLINE", "NO BET"].map((decision) => [
        decision,
        PROFILE_ORDER.flatMap((profile) =>
          (groups?.[decision]?.[profile] || []).map((slip) => ({
            ...slip,
            profile_name: slip.profile_name || profile,
            is_official: false,
          }))
        ),
      ])
    );
  }, [profiles, report?.decision_groups]);
  const proposedSlips = proposalGroups[activeDecision] || [];
  const totalProposed = Object.values(proposalGroups).reduce(
    (total, slips) => total + slips.length,
    0
  );
  const officialSlips = useMemo(
    () =>
      (report?.official_slips || []).map((slip) => ({
        ...slip,
        profile_name: slip.profile_name || slip.profile,
        is_official: true,
        legs: slip.picks || slip.legs || [],
      })),
    [report?.official_slips]
  );
  const sourceSlips = activeView === "official" ? officialSlips : proposedSlips;
  const visibleSlips = useMemo(() => {
    if (activeFilter === "play") {
      return sourceSlips.filter((slip) => (slip.situation || slip.initial_situation) === "PLAY");
    }
    if (activeFilter === "settled") {
      return sourceSlips.filter((slip) => ["WON", "LOST", "VOID"].includes(slip.status));
    }
    if (PROFILE_ORDER.includes(activeFilter)) {
      return sourceSlips.filter((slip) => (slip.profile_name || slip.profile) === activeFilter);
    }
    return sourceSlips;
  }, [activeFilter, sourceSlips]);

  async function copySlip(slip) {
    const text = [
      `Schedina ${PROFILE_LABELS[slip.profile_name || slip.profile] || slip.profile || ""}`,
      ...(slip.legs || []).map(
        (leg) =>
          `${legMatchLabel(leg, fixtureIndex)} | ${marketLabel(leg.market)} | ${leg.outcome} @ ${formatOdd(leg.odd ?? leg.market_odd)}`
      ),
      `Quota combinata: ${formatOdd(slip.combined_odd)}`,
    ].join("\n");
    await navigator.clipboard.writeText(text);
    const id = slip.slip_id || slip.id;
    setCopiedSlipId(id);
    window.setTimeout(() => setCopiedSlipId(null), 1800);
  }

  return (
    <section className="stack betslip-workspace">
      <ModelLegend />

      <section className="panel betslip-toolbar">
        <div className="betslip-toolbar-main">
          <div>
            <h3>Schedina Oracle</h3>
            <p className="muted">Combinazioni calcistiche validate dal motore centrale, senza mercati incompatibili.</p>
          </div>
          <div className="panel-header-actions">
            <label className="betslip-date">
              Data
              <input type="date" value={targetDate} onChange={(e) => onChangeTargetDate(e.target.value)} />
            </label>
            <button
              className="btn-primary"
              onClick={() => onLoadReport()}
              disabled={isLoading || isPastDate}
              title={isPastDate ? "Le date passate mostrano soltanto schedine salvate prima degli eventi." : ""}
            >
              {isPastDate ? "Solo consultazione" : "Genera e salva"}
            </button>
          </div>
        </div>
        <div className="stake-simulator">
          <label>
            Simula puntata
            <input
              type="number"
              min="1"
              step="1"
              value={stake}
              onChange={(event) => setStake(Math.max(1, Number(event.target.value) || 1))}
            />
          </label>
          <div className="stake-presets">
            {[1, 5, 10, 25, 50].map((value) => (
              <button
                key={value}
                className={stake === value ? "stake-chip active" : "stake-chip"}
                onClick={() => setStake(value)}
              >
                {value} €
              </button>
            ))}
          </div>
        </div>
        {error && <div className="error-box">Errore: {error}</div>}
        {isLoading && <div className="info-box">Generazione schedine in corso...</div>}
        {isPastDate && !isLoading && (
          <div className="info-box">
            Storico in sola lettura: vengono mostrate esclusivamente le schedine già salvate, senza rigenerazioni retroattive.
          </div>
        )}
      </section>

      <section className="panel betslip-browser">
        <div className="betslip-view-tabs" role="tablist" aria-label="Vista schedine">
          <button className={activeView === "proposals" ? "tab active" : "tab"} onClick={() => setActiveView("proposals")}>
            Schedine <span>{totalProposed}</span>
          </button>
          <button className={activeView === "official" ? "tab active" : "tab"} onClick={() => setActiveView("official")}>
            Ufficiali <span>{officialSlips.length}</span>
          </button>
        </div>

        {activeView === "proposals" && (
          <>
            <div className="betslip-decision-tabs" role="tablist" aria-label="Qualità schedine">
              {[
                ["PLAY", "Consigliate"],
                ["BORDERLINE", "Sperimentali"],
                ["NO BET", "Non consigliate"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  className={`decision-tab ${statusClass(value)} ${activeDecision === value ? "active" : ""}`}
                  onClick={() => setActiveDecision(value)}
                >
                  {label} <span>{proposalGroups[value]?.length || 0}</span>
                </button>
              ))}
            </div>
            <p className="muted">
              {activeDecision === "PLAY"
                ? "Solo selezioni PLAY: sono le uniche candidabili alla cattura ufficiale."
                : activeDecision === "BORDERLINE"
                  ? "Simulazioni con almeno una selezione BORDERLINE: salvate e monitorate, ma non ufficiali."
                  : "Simulazioni contenenti NO BET: monitoraggio sperimentale, mai raccomandazioni ufficiali."}
            </p>
          </>
        )}

        <div className="betslip-filter-tabs">
          {[
            ["all", "Tutte"],
            ["SAFE", "Prudenti"],
            ["BALANCED", "Bilanciate"],
            ["AGGRESSIVE", "Spinte"],
            ["settled", "Concluse"],
          ].map(([value, label]) => (
            <button
              key={value}
              className={activeFilter === value ? "tab active" : "tab"}
              onClick={() => setActiveFilter(value)}
            >
              {label}
            </button>
          ))}
        </div>

        <p className="muted betslip-filter-help">
          {activeFilter === "all"
            ? "Tutte le schedine disponibili per la data selezionata."
            : PROFILE_HELP[activeFilter] || "Mostra soltanto le schedine già concluse."}
        </p>

        <div className="betslip-legend">
          <span><i className="status-dot status-won" />Vinta</span>
          <span><i className="status-dot status-lost" />Persa</span>
          <span><i className="status-dot status-pending" />In corso</span>
          <span><i className="status-dot status-void" />Rimborsata</span>
          <span><i className="status-dot status-proposal" />Proposta</span>
        </div>

        {!isLoading && !error && !report && (
          <div className="empty-state">Nessuna schedina generata: premi “Genera schedine”.</div>
        )}
        {report && visibleSlips.length === 0 && (
          <div className="empty-state">Nessuna schedina disponibile per questo filtro.</div>
        )}

        <div className="betslip-list">
          {visibleSlips.map((slip) => (
            <SlipCard
              key={slip.slip_id || slip.id}
              slip={slip}
              fixtureIndex={fixtureIndex}
              stake={stake}
              copied={copiedSlipId === (slip.slip_id || slip.id)}
              onCopy={() => copySlip(slip)}
            />
          ))}
        </div>

        {report?.warnings?.length > 0 && (
          <details className="betslip-warnings">
            <summary>Avvisi di generazione ({report.warnings.length})</summary>
            {report.warnings.map((warning) => <div key={warning}>{warning}</div>)}
          </details>
        )}
      </section>

      <BettingStatistics
        report={bettingStatistics}
        fallback={report?.official_statistics}
        days={bettingStatsDays}
        onChangeDays={onLoadStatistics}
      />
    </section>
  );
}

function BettingStatistics({ report, fallback, days, onChangeDays }) {
  const [activeTab, setActiveTab] = useState("overview");
  const overview = report?.overview || {};
  const predictions = overview.official_predictions || {};
  const proposals = overview.proposals || {};
  const simulated = overview.simulated_portfolios || {};
  const official = overview.official_slips || fallback || {};
  return (
    <section className="panel betslip-statistics">
      <div className="panel-header">
        <div>
          <h3>Statistiche Betting</h3>
          <p className="muted">Proposte salvate e performance ufficiale restano sempre distinte.</p>
        </div>
        <label>
          Periodo
          <select value={days || 30} onChange={(event) => onChangeDays?.(Number(event.target.value))}>
            {[7, 30, 90, 365].map((value) => <option key={value} value={value}>{value} giorni</option>)}
          </select>
        </label>
      </div>
      <div className="betslip-view-tabs" role="tablist" aria-label="Statistiche betting">
        {[["overview", "Panoramica"], ["markets", "Mercati"], ["slips", "Schedine"]].map(([value, label]) => (
          <button key={value} className={activeTab === value ? "tab active" : "tab"} onClick={() => setActiveTab(value)}>
            {label}
          </button>
        ))}
      </div>
      {!report && !fallback ? (
        <div className="empty-state">Statistiche non ancora disponibili.</div>
      ) : activeTab === "overview" ? (
        <>
          <div className="stats-grid betting-overview-grid">
            {[
              ["Pronostici ufficiali", predictions.plays ?? 0],
              ["Pronostici vinti", predictions.wins ?? 0],
              ["Proposte salvate", proposals.generated ?? 0],
              ["Revisioni", proposals.revisions ?? 0],
              ["Schedine ufficiali", official.total ?? 0],
              ["Schedine vinte", official.won ?? 0],
              ["Profitto ufficiale", formatNumber(official.net_profit, 2)],
              ["ROI ufficiale", formatPercent(official.realized_roi)],
            ].map(([label, value]) => <article className="stat-card" key={label}><span>{label}</span><strong>{value}</strong></article>)}
          </div>
          <ShadowPortfolioTable portfolios={simulated} />
        </>
      ) : activeTab === "markets" ? (
        <DailyMarketTable rows={report?.markets?.daily || []} />
      ) : (
        <SlipStatistics report={report?.slips} proposals={proposals} official={official} />
      )}
      <p className="muted statistics-boundary">
        ROI, profitto e bankroll comprendono esclusivamente giocate e schedine ufficiali congelate prima del kickoff.
      </p>
    </section>
  );
}

function ShadowPortfolioTable({ portfolios }) {
  const rows = [
    ["ALL", "Complessivo simulato"],
    ["PLAY", "PLAY simulato"],
    ["BORDERLINE", "BORDERLINE simulato"],
    ["NO BET", "NO BET simulato"],
  ];
  return (
    <section className="shadow-portfolios">
      <h4>Portafogli simulati</h4>
      <p className="muted">Una unità per ogni ultima revisione pre-kickoff; risultati separati dal capitale ufficiale.</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Portafoglio</th><th>Schedine</th><th>Vinte</th><th>Perse</th><th>Pending</th><th>VOID</th><th>Capitale</th><th>Stake</th><th>Ritorno</th><th>Profitto</th><th>ROI simulato</th></tr></thead>
          <tbody>{rows.map(([key, label]) => {
            const row = portfolios?.[key] || {};
            return <tr key={key}><td>{label}</td><td>{row.total ?? 0}</td><td>{row.won ?? 0}</td><td>{row.lost ?? 0}</td><td>{row.pending ?? 0}</td><td>{row.void ?? 0}</td><td>{formatNumber(row.current_bankroll, 2)}</td><td>{formatNumber(row.stake, 2)}</td><td>{formatNumber(row.return, 2)}</td><td>{formatNumber(row.profit, 2)}</td><td>{formatPercent(row.roi)}</td></tr>;
          })}</tbody>
        </table>
      </div>
    </section>
  );
}

function DailyMarketTable({ rows }) {
  if (rows.length === 0) return <div className="empty-state">Nessuna statistica ufficiale per il periodo.</div>;
  return (
    <div className="table-wrap">
      <table className="betting-statistics-table">
        <thead><tr><th>Data</th><th>Mercato</th><th>Giocate</th><th>Vinte</th><th>Perse</th><th>Pending</th><th>VOID</th><th>Hit rate</th><th>Quota media</th><th>Profitto</th><th>ROI</th></tr></thead>
        <tbody>{rows.map((row) => (
          <tr key={`${row.date}-${row.market}`}>
            <td>{row.date}</td><td>{marketLabel(row.market)}</td><td>{row.plays}</td><td>{row.wins}</td><td>{row.losses}</td>
            <td>{row.pending}</td><td>{row.void}</td><td>{formatPercent(row.hit_rate)}</td><td>{formatOdd(row.avg_odd)}</td>
            <td>{formatNumber(row.total_profit, 2)}</td><td>{formatPercent(row.roi)}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function SlipStatistics({ report, proposals, official }) {
  const proposalRows = report?.proposals_daily || [];
  const officialRows = Object.entries(report?.official_daily || {});
  return (
    <div className="statistics-split">
      <section>
        <h4>Attività generata</h4>
        <p className="muted">Ogni revisione diversa viene conservata; non rappresenta una puntata piazzata.</p>
        {proposalRows.length === 0 ? <div className="empty-state">Nessuna proposta salvata.</div> : (
          <div className="table-wrap"><table><thead><tr><th>Data evento</th><th>Generate</th><th>PLAY</th><th>BORDERLINE</th><th>NO BET</th><th>Quota media</th><th>Expected ROI medio</th></tr></thead>
            <tbody>{proposalRows.map((row) => <tr key={row.date}><td>{row.date}</td><td>{row.generated}</td><td>{row.play}</td><td>{row.borderline}</td><td>{row.no_bet}</td><td>{formatOdd(row.average_combined_odd)}</td><td>{formatPercent(row.average_expected_roi)}</td></tr>)}</tbody>
          </table></div>
        )}
        <small>Totale revisioni salvate: {proposals.generated ?? 0}</small>
      </section>
      <section>
        <h4>Performance schedine ufficiali</h4>
        <p className="muted">Solo snapshot ufficiali pre-partita, mai proposte rigenerate a posteriori.</p>
        {officialRows.length === 0 ? <div className="empty-state">Nessuna schedina ufficiale nel periodo.</div> : (
          <div className="table-wrap"><table><thead><tr><th>Data evento</th><th>Totali</th><th>Vinte</th><th>Perse</th><th>Pending</th><th>VOID</th><th>Profitto</th><th>ROI</th></tr></thead>
            <tbody>{officialRows.map(([date, row]) => <tr key={date}><td>{date}</td><td>{row.total}</td><td>{row.won}</td><td>{row.lost}</td><td>{row.pending}</td><td>{row.void}</td><td>{formatNumber(row.profit, 2)}</td><td>{formatPercent(row.roi)}</td></tr>)}</tbody>
          </table></div>
        )}
        <small>Schedine ufficiali nel periodo: {official.total ?? 0}</small>
      </section>
    </div>
  );
}

function SlipCard({ slip, fixtureIndex, stake, copied, onCopy }) {
  const shadowLegs = slip.shadow_settlement?.legs;
  const legs = !slip.is_official && shadowLegs?.length ? shadowLegs : slip.legs || [];
  const situation = slip.situation || slip.initial_situation || "N/D";
  const hasShadowStatus = !slip.is_official && Boolean(slip.shadow_status);
  const status = slip.is_official ? slip.status || "PENDING" : slip.shadow_status || "PROPOSTA";
  const profile = slip.profile_name || slip.profile;
  const simulatedReturn = Number(stake) * Number(slip.combined_odd || 0);
  const simulatedProfit = simulatedReturn - Number(stake);
  return (
    <article className={`slip-card ${riskClass(slip.risk_label)}`}>
      <div className="slip-card-header">
        <div>
          <strong>{situation === "PLAY" ? "Play" : "Valutazione"} · {PROFILE_LABELS[profile] || profile || "Generica"}</strong>
          <small>{PROFILE_HELP[profile] || slip.situation_reason || slip.initial_reason}</small>
        </div>
        <div className="slip-card-status">
          <span className={`value-badge ${statusClass(situation)}`}>{situation}</span>
          <span className={`settlement-label ${statusClass(status)}`}>
            {slip.is_official ? status : hasShadowStatus ? `SIM · ${status}` : "PROPOSTA"}
          </span>
          <small>{legs.length} eventi</small>
        </div>
      </div>

      <div className="table-wrap">
        <table className="slip-picks-table">
          <thead>
            <tr>
              <th>Esito</th><th>Ora</th><th>Torneo</th><th>Partita</th><th>Mercato</th><th>Pick</th><th>Probabilità</th>
              <th>Quota</th><th title="Quota teorica di pareggio: 1 / probabilità modello.">Quota void modello</th>
              <th title="Differenza tra quota bookmaker e quota void modello.">Edge</th>
              <th title="Rendimento teorico della selezione, non quello realizzato.">Expected ROI</th><th>Situazione</th>
            </tr>
          </thead>
          <tbody>
            {legs.map((leg, index) => (
              <tr key={`${slip.slip_id || slip.id}-${index}`}>
                <td title={leg.status === "VOID" ? `Esito rimborsato: ${leg.void_reason || "evento void"}` : ""}>
                  <span className="pick-status">
                    <i className={`status-dot status-${(slip.is_official || hasShadowStatus ? leg.status || "pending" : "proposal").toLowerCase()}`} />
                    {legStatusLabel(leg.status, slip.is_official, hasShadowStatus)}
                  </span>
                </td>
                <td>{leg.kickoff_at ? new Date(leg.kickoff_at).toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" }) : "—"}</td>
                <td>{leg.competition || "—"}</td>
                <td>
                  <strong>{legMatchLabel(leg, fixtureIndex)}</strong>
                  {leg.final_score && <small>Finale: {leg.final_score}</small>}
                </td>
                <td>{marketLabel(leg.market)}{leg.line ? ` · ${leg.line}` : ""}</td>
                <td>{leg.outcome}</td>
                <td>{formatPercent(leg.p_model)}</td>
                <td>{formatOdd(leg.odd ?? leg.market_odd)}</td>
                <td>{formatOdd(leg.model_void_odd)}</td>
                <td title={`Edge percentuale: ${formatPercent(leg.odds_edge_percent == null ? null : leg.odds_edge_percent / 100)}`}>
                  {formatNumber(leg.odds_edge_absolute, 3)}
                </td>
                <td>{formatPercent(leg.ev ?? leg.expected_roi)}</td>
                <td><span className={`value-badge ${statusClass(leg.decision || leg.situation)}`}>{leg.decision || leg.situation}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="slip-footer">
        <div className="slip-footer-metrics">
          <span><small>Quota combinata</small><strong>{formatOdd(slip.combined_odd)}</strong></span>
          <span title="Quota teorica di pareggio ricavata dalla probabilità combinata corretta."><small>Quota void combinata</small><strong>{formatOdd(slip.combined_model_void_odd)}</strong></span>
          <span><small>Edge combinato</small><strong>{formatNumber(slip.combined_edge_absolute, 3)}</strong></span>
          <span title="Rendimento teorico pre-partita; non è il ROI realizzato."><small>Expected ROI</small><strong>{formatPercent(slip.combined_expected_roi)}</strong></span>
          <span title="Probabilità aggregata restituita dal motore centrale delle correlazioni."><small>Probabilità corretta</small><strong>{formatPercent(slip.adjusted_probability)}</strong></span>
          <span><small>Puntata simulata</small><strong>{formatNumber(stake, 2)} €</strong></span>
          <span><small>Vincita potenziale</small><strong>{formatNumber(simulatedReturn, 2)} €</strong></span>
          <span><small>Profitto potenziale</small><strong>{formatNumber(simulatedProfit, 2)} €</strong></span>
        </div>
        <div className="slip-actions">
          <button className="btn-secondary" onClick={onCopy}>{copied ? "Copiata!" : "Copia schedina"}</button>
          <button className="btn-secondary" onClick={() => window.print()}>Stampa / PDF</button>
        </div>
      </div>

      <details className="slip-technical">
        <summary>Dettagli tecnici e correlazioni</summary>
        <p>{slip.explanation || slip.initial_reason || "Nessuna nota aggiuntiva."}</p>
        <span>Policy: {slip.decision_policy_version || slip.policy_version || "N/D"}</span>
        <span> · Correlazioni: {slip.correlation_ruleset_version || slip.correlation_version || "N/D"}</span>
      </details>
    </article>
  );
}

