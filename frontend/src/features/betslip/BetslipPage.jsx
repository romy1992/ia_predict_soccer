import { useMemo, useState } from "react";
import { formatNumber, formatOdd, formatPercent, marketLabel } from "../shared/formatters";

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
    return "risk-low";
  }
  if (riskLabel === "HIGH") {
    return "risk-high";
  }
  return "risk-medium";
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

export default function BetslipPage({
  targetDate,
  onChangeTargetDate,
  report,
  isLoading,
  error,
  onLoadReport,
  dayData,
}) {
  const [activeProfile, setActiveProfile] = useState("SAFE");

  const fixtureIndex = useMemo(() => {
    const map = {};
    (dayData?.rows || []).forEach((row) => {
      map[row.fixture_id] = row;
    });
    return map;
  }, [dayData]);

  const profiles = report?.profiles || {};
  const activeSlips = profiles[activeProfile] || [];

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <h3>Schedina Oracle</h3>
          <div className="panel-header-actions">
            <input
              type="date"
              value={targetDate}
              onChange={(e) => onChangeTargetDate(e.target.value)}
            />
            <button className="btn-primary" onClick={() => onLoadReport()} disabled={isLoading}>
              Genera schedine
            </button>
          </div>
        </div>

        <p className="muted">
          Combinazioni di 2/3/4 eventi dal Pick Pool (solo PLAY), validate dal Correlation Engine:
          nessuna schedina con esiti logicamente incompatibili nella stessa partita. Quota combinata e
          probabilita' sono sempre dichiarate con il metodo di calcolo esplicito (vedi spiegazione su ogni schedina).
        </p>

        {error && <div className="error-box">Errore: {error}</div>}
        {isLoading && <div className="info-box">Generazione schedine in corso...</div>}
        {!isLoading && !error && !report && (
          <div className="empty-state">Nessuna schedina generata ancora: premi "Genera schedine".</div>
        )}

        {report && (
          <div className="detail-head-meta">
            <span>Generato: {new Date(report.generated_at).toLocaleString("it-IT")}</span>
            <span>Pick nel pool: {report.pool_considered}</span>
            <span>Ruleset correlazione: {report.correlation_ruleset_version}</span>
            {report.pool_policy_version && <span>Policy pool: {report.pool_policy_version}</span>}
          </div>
        )}

        {report?.warnings?.length > 0 && (
          <div className="info-box">
            {report.warnings.map((warning) => (
              <div key={warning}>{warning}</div>
            ))}
          </div>
        )}
      </section>

      {report && (
        <section className="panel">
          <div className="tabs">
            {PROFILE_ORDER.map((name) => (
              <button
                key={name}
                className={activeProfile === name ? "tab active" : "tab"}
                onClick={() => setActiveProfile(name)}
              >
                {PROFILE_LABELS[name]} ({(profiles[name] || []).length})
              </button>
            ))}
          </div>
          <p className="muted">{PROFILE_HELP[activeProfile]}</p>

          {activeSlips.length === 0 && (
            <div className="empty-state">
              Nessuna schedina generata per il profilo {PROFILE_LABELS[activeProfile]} con le pick disponibili oggi.
            </div>
          )}

          <div className="decision-grid">
            {activeSlips.map((slip) => (
              <SlipCard key={slip.slip_id} slip={slip} fixtureIndex={fixtureIndex} />
            ))}
          </div>
        </section>
      )}

      {report?.official_statistics && (
        <section className="panel">
          <h3>Rendimento schedine ufficiali</h3>
          <div className="decision-counters">
            <span>Totali: {report.official_statistics.total}</span>
            <span>Pending: {report.official_statistics.pending}</span>
            <span>Vinte: {report.official_statistics.won}</span>
            <span>Perse: {report.official_statistics.lost}</span>
            <span>Rimborsate: {report.official_statistics.void}</span>
            <span>Profitto: {formatNumber(report.official_statistics.net_profit, 2)}</span>
            <span>ROI realizzato: {formatPercent(report.official_statistics.realized_roi)}</span>
          </div>
          <p className="muted">Calcolato esclusivamente sulle schedine congelate dal job server-side.</p>
        </section>
      )}

      {report?.official_slips?.length > 0 && (
        <section className="panel">
          <h3>Schedine ufficiali del giorno</h3>
          <div className="decision-grid">
            {report.official_slips.map((slip) => (
              <SlipCard key={slip.id} slip={{ ...slip, is_official: true, legs: slip.picks }} fixtureIndex={fixtureIndex} />
            ))}
          </div>
        </section>
      )}
    </section>
  );
}

function SlipCard({ slip, fixtureIndex }) {
  const legs = slip.legs || [];
  const situation = slip.situation || slip.initial_situation || "N/D";
  const status = slip.status || "PROPOSTA";
  return (
    <article className={`decision-card slip-card ${riskClass(slip.risk_label)}`}>
      <div className="decision-head">
        <span className={`value-badge ${statusClass(situation)}`}>
          {situation}
        </span>
        <strong>{slip.is_official ? "UFFICIALE" : "PROPOSTA"} · {status}</strong>
      </div>

      <div className="decision-metrics slip-summary">
        <span>Profilo: {PROFILE_LABELS[slip.profile_name || slip.profile] || slip.profile}</span>
        <span>Eventi: {slip.n_legs || slip.event_count}</span>
        <span>Quota combinata: {formatOdd(slip.combined_odd)}</span>
        <span title="Quota teorica di pareggio ricavata dalla probabilità combinata corretta.">
          Quota void combinata: {formatOdd(slip.combined_model_void_odd)}
        </span>
        <span>Edge combinato: {formatNumber(slip.combined_edge_absolute, 3)} ({formatPercent(slip.combined_edge_percent == null ? null : slip.combined_edge_percent / 100)})</span>
        <span title="Rendimento teorico pre-partita; non è il ROI realizzato.">
          Expected ROI: {formatPercent(slip.combined_expected_roi)}
        </span>
        <span title="Probabilità aggregata restituita dal motore centrale delle correlazioni.">
          Probabilità corretta: {formatPercent(slip.adjusted_probability)}
        </span>
        <span>Rischio: {formatPercent(slip.risk_score)}</span>
        <span>Policy: {slip.decision_policy_version || slip.policy_version}</span>
        <span>Correlazioni: {slip.correlation_ruleset_version || slip.correlation_version}</span>
      </div>

      <div className="table-wrap">
        <table className="slip-picks-table">
          <thead>
            <tr>
              <th>Partita</th><th>Mercato</th><th>Selezione</th><th>Probabilità</th>
              <th>Quota</th><th title="Quota teorica di pareggio: 1 / probabilità modello.">Quota void modello</th>
              <th title="Differenza tra quota bookmaker e quota void modello.">Edge</th>
              <th title="Rendimento teorico della selezione, non quello realizzato.">Expected ROI</th>
              <th>Situazione</th><th>Esito</th>
            </tr>
          </thead>
          <tbody>
            {legs.map((leg, index) => (
              <tr key={`${slip.slip_id || slip.id}-${index}`}>
                <td>
                  <strong>{legMatchLabel(leg, fixtureIndex)}</strong>
                  <small>{leg.competition ? `${leg.competition} · ` : ""}{leg.kickoff_at ? new Date(leg.kickoff_at).toLocaleString("it-IT") : ""}</small>
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
                <td title={leg.status === "VOID" ? `Esito rimborsato: ${leg.void_reason || "evento void"}` : ""}>
                  <span className={`value-badge ${statusClass(leg.status || "PENDING")}`}>{leg.status || "PENDING"}</span>
                  {leg.final_score && <small>{leg.final_score}</small>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="muted slip-explanation">{slip.explanation || slip.initial_reason}</p>
    </article>
  );
}

