import { useEffect, useState } from "react";
import { getOracleMatchDetail } from "../../api";
import {
  formatEdge,
  formatNumber,
  formatOdd,
  formatPercent,
  marketLabel,
  phaseClass,
  phaseLabel,
  valueClass,
} from "../shared/formatters";

const SCORE_MATRIX_DISPLAY_MAX_GOALS = 5;

function TeamStrengthCard({ label, rating }) {
  if (!rating) {
    return (
      <article className="detail-block">
        <h5>{label}</h5>
        <div className="empty-panel">Non disponibile</div>
      </article>
    );
  }

  return (
    <article className="detail-block team-strength-card">
      <h5>{label}</h5>
      <div className="metric-row"><span>Attacco</span><strong>{formatNumber(rating.team_attack_rating)}</strong></div>
      <div className="metric-row"><span>Difesa</span><strong>{formatNumber(rating.team_defense_rating)}</strong></div>
      <div className="metric-row"><span>Fattore campo</span><strong>{formatNumber(rating.team_home_advantage)}</strong></div>
      <div className="metric-row"><span>Forma (ultime 5)</span><strong>{formatPercent(rating.team_rolling_form)}</strong></div>
      <div className="metric-row"><span>Diff. gol recente</span><strong>{formatNumber(rating.team_rolling_goal_diff)}</strong></div>
      <div className="metric-row"><span>Partite analizzate</span><strong>{rating.team_matches_played ?? "-"}</strong></div>
    </article>
  );
}

function ScoreMatrixTable({ scoreMatrix }) {
  if (!scoreMatrix) {
    return <div className="empty-panel">Score matrix non disponibile.</div>;
  }

  const rowKeys = Object.keys(scoreMatrix)
    .map(Number)
    .filter((n) => n <= SCORE_MATRIX_DISPLAY_MAX_GOALS)
    .sort((a, b) => a - b);
  if (rowKeys.length === 0) {
    return <div className="empty-panel">Score matrix non disponibile.</div>;
  }
  const colKeys = Object.keys(scoreMatrix[String(rowKeys[0])] || {})
    .map(Number)
    .filter((n) => n <= SCORE_MATRIX_DISPLAY_MAX_GOALS)
    .sort((a, b) => a - b);

  let bestCell = { home: 0, away: 0, prob: -1 };
  rowKeys.forEach((home) => {
    colKeys.forEach((away) => {
      const prob = Number(scoreMatrix[String(home)]?.[String(away)] || 0);
      if (prob > bestCell.prob) {
        bestCell = { home, away, prob };
      }
    });
  });

  return (
    <div className="table-wrap">
      <table className="score-matrix-table">
        <thead>
          <tr>
            <th>Home \ Away</th>
            {colKeys.map((away) => (
              <th key={`col-${away}`}>{away}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowKeys.map((home) => (
            <tr key={`row-${home}`}>
              <th>{home}</th>
              {colKeys.map((away) => {
                const prob = Number(scoreMatrix[String(home)]?.[String(away)] || 0);
                const isBest = home === bestCell.home && away === bestCell.away;
                return (
                  <td key={`cell-${home}-${away}`} className={isBest ? "score-matrix-best" : ""}>
                    {formatPercent(prob)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OddsMovementTable({ rows }) {
  if (!rows || rows.length === 0) {
    return <div className="empty-panel">Nessuno storico quote (odds snapshot) disponibile per questa fixture.</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Bookmaker</th>
            <th>Mercato</th>
            <th>Outcome</th>
            <th>Apertura</th>
            <th>Ultima</th>
            <th>Chiusura</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={`movement-${idx}`}>
              <td>{row.bookmaker}</td>
              <td>{marketLabel(row.market)}{row.line ? ` (${row.line})` : ""}</td>
              <td>{row.outcome}</td>
              <td>{formatOdd(row.opening?.odd)}</td>
              <td>{formatOdd(row.latest?.odd)}</td>
              <td>{row.closing ? formatOdd(row.closing.odd) : "N/D"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelConsensusSection({ consensusByMarket }) {
  const markets = Object.keys(consensusByMarket || {});
  if (markets.length === 0) {
    return <div className="empty-panel">Model consensus non disponibile (nessun esperto/modello utilizzabile per questa fixture).</div>;
  }

  return (
    <div className="decision-grid">
      {markets.map((market) => {
        const report = consensusByMarket[market];
        const oracleFinal = report.oracle_final;
        return (
          <div className="decision-card" key={`consensus-${market}`}>
            <div className="decision-head">
              <strong>{marketLabel(market)}</strong>
              {report.consensus?.agreement_level && (
                <span className={`value-badge ${valueClass(report.consensus.agreement_level === "high" ? "PLAY" : report.consensus.agreement_level === "medium" ? "BORDERLINE" : "NO BET")}`}>
                  accordo {report.consensus.agreement_level}
                </span>
              )}
            </div>
            {oracleFinal ? (
              <p className="decision-pick">
                Oracle: {formatPercent(oracleFinal.probability)} <small>({oracleFinal.source})</small>
              </p>
            ) : (
              <p className="decision-pick">Oracle finale non disponibile</p>
            )}
            <div className="decision-metrics">
              {(report.experts || []).map((expert) => (
                <span key={`${market}-${expert.expert_name}`}>
                  {expert.expert_name}: {expert.comparable ? formatPercent(expert.probability) : "n/d"}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function OracleMatchDetailPage({ fixtureId, onBack }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!fixtureId) {
      setDetail(null);
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    setError("");

    getOracleMatchDetail(fixtureId)
      .then((payload) => {
        if (!cancelled) {
          setDetail(payload);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [fixtureId]);

  if (!fixtureId) {
    return <div className="empty-panel">Seleziona una partita dal Match Center per aprire l'Oracle Match Detail.</div>;
  }

  const fixture = detail?.overview?.fixture;

  return (
    <section className="stack">
      <div className="panel-header">
        <h3>Oracle Match Detail</h3>
        <button className="btn-secondary" onClick={onBack}>&larr; Torna al Match Center</button>
      </div>

      {loading && <div className="info-box">Caricamento dettaglio Oracle...</div>}
      {error && <div className="error-box">Errore: {error}</div>}

      {!loading && !error && detail && (
        <>
          {/* 1. Overview */}
          <section className="panel">
            {fixture ? (
              <>
                <div className="detail-head-meta">
                  <span className={`phase-badge ${phaseClass(fixture.phase)}`}>{phaseLabel(fixture.phase)}</span>
                  <span>{fixture.date} {fixture.time}</span>
                  <span>{fixture.league || "-"}</span>
                  <span>Score: {fixture.score?.home ?? "-"} - {fixture.score?.away ?? "-"}</span>
                </div>
                <h2>{fixture.home} vs {fixture.away}</h2>
              </>
            ) : (
              <div className="empty-panel">Fixture non trovata (ne' dal DB locale ne' dal feed live/API).</div>
            )}

            {(detail.overview?.timeline || []).length > 0 && (
              <div className="timeline-list">
                {detail.overview.timeline.map((event, idx) => (
                  <div className="timeline-item" key={`ev-${idx}`}>
                    <span className="timeline-minute">{event.minute}</span>
                    <div>
                      <strong>{event.team || "-"}</strong>
                      <p>{event.type || "Evento"}{event.detail ? ` - ${event.detail}` : ""}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* 2/3. Probabilities + Value Bets */}
          <section className="panel">
            <div className="panel-header">
              <h4>Value Bets</h4>
              <span className="pill">{detail.value_bets?.length || 0} su {detail.probabilities?.length || 0} mercati</span>
            </div>
            {(detail.probabilities || []).length === 0 ? (
              <div className="empty-panel">Nessuna probabilita' calcolata (nessun modello disponibile per questa fixture).</div>
            ) : (
              <div className="decision-grid">
                {detail.probabilities.map((card) => (
                  <div className="decision-card" key={`prob-${card.market}`}>
                    <div className="decision-head">
                      <strong>{marketLabel(card.market)}</strong>
                      <span className={`value-badge ${valueClass(card.value_label)}`}>{card.value_label}</span>
                    </div>
                    <p className="decision-pick">{card.pick}</p>
                    <div className="decision-metrics">
                      <span>Conf.: {formatPercent(card.predicted_probability)}</span>
                      <span>Quota: {formatOdd(card.odd)}</span>
                      <span>Fair: {formatOdd(card.fair_odd)}</span>
                      <span>Edge: {formatEdge(card.edge)}</span>
                      <span>EV: {formatEdge(card.ev)}</span>
                    </div>
                    <small>{card.value_reason}</small>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* 4. Team Strength */}
          <section className="panel">
            <div className="panel-header"><h4>Team Strength</h4></div>
            {detail.team_strength ? (
              <div className="detail-grid">
                <TeamStrengthCard label={fixture?.home || "Home"} rating={detail.team_strength.home} />
                <TeamStrengthCard label={fixture?.away || "Away"} rating={detail.team_strength.away} />
              </div>
            ) : (
              <div className="empty-panel">Rating non disponibile (storico insufficiente per una o entrambe le squadre).</div>
            )}
          </section>

          {/* 5/6. Expected Goals + Score Matrix */}
          <section className="panel">
            <div className="panel-header"><h4>Expected Goals &amp; Score Matrix</h4></div>
            {detail.expected_goals ? (
              <>
                <div className="detail-head-meta">
                  <span>xG Home: {formatNumber(detail.expected_goals.home_lambda)}</span>
                  <span>xG Away: {formatNumber(detail.expected_goals.away_lambda)}</span>
                  <span>xG Totali: {formatNumber(detail.expected_goals.total_lambda)}</span>
                </div>
                <div className="decision-metrics">
                  {Object.entries(detail.expected_goals.over_under || {}).map(([key, value]) => (
                    <span key={key}>{key.replace("over_", "Over ").replace("_", ".")}: {formatPercent(value)}</span>
                  ))}
                </div>
                <ScoreMatrixTable scoreMatrix={detail.score_matrix} />
              </>
            ) : (
              <div className="empty-panel">Expected goals non disponibili (dipende dal Team Strength).</div>
            )}
          </section>

          {/* 7. Odds Movement */}
          <section className="panel">
            <div className="panel-header"><h4>Odds Movement</h4></div>
            <OddsMovementTable rows={detail.odds_movement} />
          </section>

          {/* 8. Model Consensus */}
          <section className="panel">
            <div className="panel-header"><h4>Model Consensus</h4></div>
            <ModelConsensusSection consensusByMarket={detail.model_consensus} />
          </section>

          {(detail.warnings || []).length > 0 && (
            <div className="info-box">
              Sezioni non disponibili: {detail.warnings.join(", ")}
            </div>
          )}
        </>
      )}
    </section>
  );
}

