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
  SAFE: "Safe",
  BALANCED: "Balanced",
  AGGRESSIVE: "Aggressive",
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
  if (!info) {
    return `Fixture #${leg.fixture_id}`;
  }
  return `${info.home} vs ${info.away}`;
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
    </section>
  );
}

function SlipCard({ slip, fixtureIndex }) {
  return (
    <article className={`decision-card slip-card ${riskClass(slip.risk_label)}`}>
      <div className="decision-head">
        <span className={`value-badge ${riskClass(slip.risk_label)}`}>
          {slip.n_legs} eventi · rischio {slip.risk_label}
        </span>
        <strong>quota {formatOdd(slip.combined_odd)}</strong>
      </div>

      <ul className="slip-legs">
        {slip.legs.map((leg, index) => (
          <li key={`${slip.slip_id}-${index}`}>
            <div className="match-title">{legMatchLabel(leg, fixtureIndex)}</div>
            <div className="match-sub">
              {marketLabel(leg.market)}: <strong>{leg.outcome}</strong> · quota {formatOdd(leg.odd)} · p.
              modello {formatPercent(leg.p_model)}
            </div>
          </li>
        ))}
      </ul>

      <div className="decision-metrics">
        <span>Probabilita' ingenua (prodotto semplice): {formatPercent(slip.naive_probability)}</span>
        <span>Probabilita' corretta per correlazione: {formatPercent(slip.adjusted_probability)}</span>
        <span>EV combinato: {formatNumber(slip.combined_ev, 3)}</span>
        <span>Coppie penalizzate per correlazione: {slip.penalty_pairs}</span>
      </div>

      <p className="muted slip-explanation">{slip.explanation}</p>
    </article>
  );
}

