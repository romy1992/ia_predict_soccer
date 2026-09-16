import {
  formatOdd,
  formatPercent,
  formatPercentagePoints,
  formatSignedNumber,
  marketLabel,
  phaseClass,
  phaseLabel,
  pickProbability,
  predictionLabel,
  valueClass,
} from "../../shared/formatters";

export default function MatchDetailPanel({
  selectedFixtureId,
  matchDetail,
  matchDetailLoading,
  matchDetailError,
  onOpenOracleDetail,
  onClose,
  onRecomputePredictions,
  recomputingPredictions,
  recomputePredictionsError,
}) {
  if (!selectedFixtureId && !matchDetail) {
    return null;
  }

  if (matchDetailLoading) {
    return <section className="panel detail-panel"><div className="empty-panel">Caricamento dettaglio partita...</div></section>;
  }

  if (matchDetailError) {
    return <section className="panel detail-panel"><div className="error-box">{matchDetailError}</div></section>;
  }

  const fixture = matchDetail?.fixture;
  if (!fixture) {
    return <section className="panel detail-panel"><div className="empty-panel">Dettaglio non disponibile per il fixture selezionato.</div></section>;
  }

  const decisionCards = matchDetail?.decision_cards || [];
  const timeline = matchDetail?.timeline || [];
  const oddsSummary = matchDetail?.odds_summary || {};
  const oddsMarkets = Object.keys(oddsSummary);
  // Corners/Cards a linea configurabile (MARKET-05/06, 2026-09-13): servite
  // come predizione ML grezza (pick/probabilita'/segnale a soglia ottima),
  // NON ancora come decision card completa - il motore fair-odds/EV lavora
  // per outcome di mercato "intero", non per linea configurabile dentro lo
  // stesso mercato quote (vedi IMPLEMENTATION_LOG.md). Lette da `predictions`
  // (non da `decisionCards`, dove queste linee non compaiono mai).
  const linePredictions = Object.entries(matchDetail?.predictions || {}).filter(
    ([marketKey]) => marketKey.startsWith("corners_line_") || marketKey.startsWith("cards_line_")
  );

  return (
    <section className="panel detail-panel">
      <div className="panel-header">
        <h3>Dettaglio match: {fixture.home} vs {fixture.away}</h3>
        <div className="panel-header-actions">
          {onOpenOracleDetail && (
            <button className="btn-primary" onClick={() => onOpenOracleDetail(selectedFixtureId)}>
              Oracle Match Detail &rarr;
            </button>
          )}
          {onRecomputePredictions && (
            <button
              className="btn-secondary"
              onClick={() => onRecomputePredictions(selectedFixtureId)}
              disabled={recomputingPredictions}
              title="Ricalcola e salva una nuova previsione per questa partita, anche se una e' gia' salvata."
            >
              {recomputingPredictions ? "Ricalcolo in corso..." : "Ricalcola previsione"}
            </button>
          )}
          <button className="btn-secondary" onClick={onClose}>Chiudi dettaglio</button>
        </div>
      </div>

      {recomputePredictionsError && <div className="error-box">Errore ricalcolo: {recomputePredictionsError}</div>}

      <div className="detail-head-meta">
        <span className={`phase-badge ${phaseClass(fixture.phase)}`}>{phaseLabel(fixture.phase)}</span>
        <span>{fixture.date} {fixture.time}</span>
        <span>{fixture.league || "-"}</span>
        <span>Score: {fixture.score?.home ?? "-"} - {fixture.score?.away ?? "-"}</span>
        <span>Fonte: {fixture.source || "-"}</span>
      </div>

      <div className="detail-grid">
        <article className="detail-block">
          <h4>Consiglio valore (PLAY / BORDERLINE / NO BET)</h4>
          {decisionCards.length === 0 ? (
            <div className="empty-panel">Nessun modello disponibile o feature non ancora presenti per questo fixture.</div>
          ) : (
            <div className="decision-grid">
              {decisionCards.map((card) => (
                <div className="decision-card" key={`${card.market}-${card.run_id || card.pick}`}>
                  <div className="decision-head">
                    <strong>{marketLabel(card.market)}</strong>
                    <span className={`value-badge ${valueClass(card.value_label)}`}>{card.value_label}</span>
                  </div>
                  <p className="decision-pick">{card.pick}</p>
                  {card.bet_over_signal?.signal && (
                    <span
                      className="bet-over-badge"
                      title={`Soglia orientata a precisione: P(Over) >= ${formatPercent(card.bet_over_signal.threshold)} (precisione attesa ${formatPercent(card.bet_over_signal.expected_precision)}, recall atteso ${formatPercent(card.bet_over_signal.expected_recall)}). Segnale indipendente dal pick sopra.`}
                    >
                      BET OVER
                    </span>
                  )}
                  <div className="decision-metrics">
                    <span>Conf.: {formatPercent(card.predicted_probability)}</span>
                    <span>Quota mercato: {formatOdd(card.market_odd)}</span>
                    <span title="Quota di pareggio economico ricavata dalla probabilità IA.">Quota void IA: {formatOdd(card.model_void_odd)}</span>
                    <span>Quota fair mercato: {formatOdd(card.market_fair_odd)}</span>
                    <span title={`Edge percentuale: ${formatPercentagePoints(card.odds_edge_percent)}`}>Edge quota: {formatSignedNumber(card.odds_edge_absolute)}</span>
                    <span>Edge probabilistico: {formatPercent(card.prob_edge)}</span>
                    <span title="Rendimento teorico della singola selezione; non è il ROI realmente ottenuto.">ROI atteso: {formatPercentagePoints(card.expected_roi_percent)}</span>
                    <span>Soglia PLAY: {formatOdd(card.play_threshold_odd)}</span>
                    <span>Bookmaker: {card.bookmakers_count ?? "-"}</span>
                    {card.is_official && <span title="Stato di settlement, distinto dalla quota void IA.">Esito: {card.official_outcome}</span>}
                    {card.is_official && card.pnl != null && <span>PnL: {formatSignedNumber(card.pnl)}</span>}
                  </div>
                  <small>{card.value_reason} · Policy {card.policy_version || "N/D"}</small>
                </div>
              ))}
            </div>
          )}
        </article>

        <article className="detail-block">
          <h4>Timeline eventi</h4>
          {timeline.length === 0 ? (
            <div className="empty-panel">Nessun evento disponibile per questo match.</div>
          ) : (
            <div className="timeline-list">
              {timeline.map((event, idx) => (
                <div className="timeline-item" key={`${event.minute}-${idx}`}>
                  <span className="timeline-minute">{event.minute}</span>
                  <div>
                    <strong>{event.team || "-"}</strong>
                    <p>{event.type || "Evento"}{event.detail ? ` - ${event.detail}` : ""}</p>
                    {(event.player || event.assist || event.comments) && (
                      <small>
                        {[event.player, event.assist ? `assist ${event.assist}` : null, event.comments].filter(Boolean).join(" | ")}
                      </small>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </article>
      </div>

      {linePredictions.length > 0 && (
        <article className="detail-block">
          <h4>Corners / Cards (linee)</h4>
          <div className="decision-grid">
            {linePredictions.map(([marketKey, payload]) => (
              <div className="decision-card" key={marketKey}>
                <div className="decision-head">
                  <strong>{marketLabel(marketKey)}</strong>
                </div>
                <p className="decision-pick">{predictionLabel(marketKey, payload.prediction, fixture)}</p>
                {payload.line_market_signal?.signal && (
                  <span
                    className="bet-over-badge"
                    title={`Soglia ottimale (Youden): P(Over) >= ${formatPercent(payload.line_market_signal.threshold)} (accuracy attesa ${formatPercent(payload.line_market_signal.expected_accuracy)}). Segnale indipendente dal pick sopra.`}
                  >
                    OVER (soglia ottima)
                  </span>
                )}
                <div className="decision-metrics">
                  <span>Conf.: {formatPercent(pickProbability(payload.prediction, payload.probability))}</span>
                </div>
              </div>
            ))}
          </div>
          <small>Solo predizione ML: nessun motore fair-odds/EV per queste linee (ancora non esteso a Corners/Cards).</small>
        </article>
      )}

      <article className="detail-block">
        <h4>Quote medie bookmaker</h4>
        {oddsMarkets.length === 0 ? (
          <div className="empty-panel">Nessuna quota disponibile al momento.</div>
        ) : (
          <div className="odds-market-grid">
            {oddsMarkets.map((marketKey) => (
              <div className="odds-market-card" key={`odds-${marketKey}`}>
                <h5>{marketLabel(marketKey)}</h5>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Outcome</th>
                        <th>Avg</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Book</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(oddsSummary[marketKey] || []).map((odd, idx) => (
                        <tr key={`${marketKey}-${idx}`}>
                          <td>{odd.outcome}</td>
                          <td>{formatOdd(odd.avg_odd)}</td>
                          <td>{formatOdd(odd.min_odd)}</td>
                          <td>{formatOdd(odd.max_odd)}</td>
                          <td>{odd.bookmakers ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}

        {matchDetail?.odds_updated_at && (
          <small className="muted">Aggiornamento quote API: {matchDetail.odds_updated_at}</small>
        )}
      </article>
    </section>
  );
}
