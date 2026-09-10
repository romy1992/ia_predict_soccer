export function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

export function formatDateIt(iso) {
  if (!iso) {
    return "-";
  }
  const [y, m, d] = String(iso).split("-");
  if (!y || !m || !d) {
    return iso;
  }
  return `${d}/${m}/${y}`;
}

export function formatPercent(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "--";
  }
  return `${(num * 100).toFixed(1)}%`;
}

export function marketLabel(market) {
  const map = {
    h2h: "Vincitore partita",
    "1x2": "1X2",
    goal_no_goal: "Goal / No Goal",
    dc: "Doppia chance",
    corners: "Corners",
    cards: "Cards",
    under_over_1_5: "Over/Under 1.5",
    under_over_2_5: "Over/Under 2.5",
    under_over_3_5: "Over/Under 3.5",
    under_over_4_5: "Over/Under 4.5",
  };
  return map[market] || market;
}

export function predictionLabel(market, prediction, row) {
  if (market === "goal_no_goal") {
    return prediction === 1 ? "Goal" : "No Goal";
  }
  if (market === "dc") {
    return prediction === 1 ? "1X" : "Vittoria ospite";
  }
  if (market.startsWith("under_over_")) {
    const threshold = market.replace("under_over_", "").replace("_", ".");
    return prediction === 1 ? `Over ${threshold}` : `Under ${threshold}`;
  }
  if (market === "h2h") {
    return prediction === 1 ? row.home : "Non casa";
  }
  if (market === "corners") {
    return prediction === 1 ? "Over corners" : "Under corners";
  }
  if (market === "cards") {
    return prediction === 1 ? "Over cards" : "Under cards";
  }
  return String(prediction);
}

export function phaseLabel(phase) {
  if (phase === "live") {
    return "In diretta";
  }
  if (phase === "finished") {
    return "Finita";
  }
  return "Da giocare";
}

export function phaseClass(phase) {
  if (phase === "live") {
    return "badge-live";
  }
  if (phase === "finished") {
    return "badge-finished";
  }
  return "badge-upcoming";
}

export function confidenceClass(probability) {
  if (probability >= 0.8) {
    return "prediction-strong";
  }
  if (probability >= 0.65) {
    return "prediction-medium";
  }
  return "prediction-low";
}

// Colore per ESITO REALE (2026-09-10, richiesto esplicitamente
// dall'operatore: "quando una partita e' finita, colorami di verde le
// odds prese e in rosso quelle non prese") - distinto da `confidenceClass`
// (che colora per PROBABILITA' stimata, sempre disponibile anche prima del
// fischio d'inizio). `correct` e' `true`/`false` SOLO per partite concluse
// con risultato determinabile per quel mercato (vedi
// `DashboardService._annotate_prediction_correctness`); `null`/`undefined`
// altrimenti, nel qual caso il chiamante deve ricadere su `confidenceClass`.
export function outcomeClass(correct) {
  if (correct === true) {
    return "prediction-correct";
  }
  if (correct === false) {
    return "prediction-wrong";
  }
  return null;
}

export function formatOdd(value) {
  const num = Number(value);
  if (Number.isNaN(num) || num <= 0) {
    return "-";
  }
  return num.toFixed(2);
}

export function formatNumber(value, decimals = 2) {
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  return num.toFixed(decimals);
}

export function formatEdge(value) {
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  const pct = num * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

export function formatPercentagePoints(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(1)}%`;
}

export function formatSignedNumber(value, decimals = 2) {
  if (value === null || value === undefined) {
    return "-";
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    return "-";
  }
  return `${num > 0 ? "+" : ""}${num.toFixed(decimals)}`;
}

export function valueClass(valueLabel) {
  if (valueLabel === "PLAY") {
    return "value-play";
  }
  if (valueLabel === "BORDERLINE") {
    return "value-borderline";
  }
  if (valueLabel === "SENZA QUOTA" || valueLabel === "N/D") {
    return "value-unavailable";
  }
  return "value-no-bet";
}

/**
 * OPS-03 (Monitoring): mappa la severity di un alert (info/warning/
 * critical) sulla STESSA palette colori gia' usata da `valueClass`
 * (verde/giallo/rosso) - nessuna nuova classe CSS, riuso diretto.
 */
export function severityClass(severity) {
  if (severity === "critical") {
    return "value-no-bet";
  }
  if (severity === "warning") {
    return "value-borderline";
  }
  return "value-play";
}

/**
 * Filtra una riga (dayData.rows) per un singolo mercato SENZA una nuova
 * fetch al backend: la riga arriva gia' con le predizioni di TUTTI i
 * mercati (vedi `App.jsx::loadDashboardData`, che scarica sempre
 * fase+mercato "Tutti" una sola volta per data/ricerca) - il cambio tab
 * Mercato/Fase deve quindi essere istantaneo, solo un filtro locale sugli
 * oggetti gia' in memoria (`predictions`/`decision_cards`), mai un nuovo
 * giro di rete che ricalcola le predizioni ML per centinaia di fixture.
 */
export function filterRowByMarket(row, market) {
  if (!market || market === "all") {
    return row;
  }
  const predictions = row.predictions || {};
  const filteredPredictions = predictions[market] ? { [market]: predictions[market] } : {};
  const decisionCards = (row.decision_cards || []).filter((card) => card.market === market);
  return {
    ...row,
    predictions: filteredPredictions,
    decision_cards: decisionCards,
    best_decision: decisionCards.find((card) => card.is_market_best) || decisionCards[0] || null,
  };
}

