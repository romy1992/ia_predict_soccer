import { formatEdge, formatOdd, marketLabel, valueClass } from "../../shared/formatters";

/**
 * MATCH-01: badge decision (PLAY / BORDERLINE / NO BET) per la vista lista
 * (Match Center). Mostra SOLO dati gia' calcolati dal backend
 * (`row.best_decision`, da `DashboardService._build_decision_cards` +
 * `_select_best_decision_card`) - nessun calcolo di edge/EV/decisione qui,
 * solo formattazione (acceptance criteria "FE non ricalcola logica betting").
 *
 * `decision` puo' essere `null` quando la fixture non e' ancora nel DB
 * locale (nessuna quota disponibile senza una fetch dedicata): in quel
 * caso si mostra esplicitamente "N/D", mai un badge inventato.
 */
export default function ValueBadge({ decision }) {
  if (!decision) {
    return <span className="empty-state">N/D</span>;
  }

  return (
    <span className={`value-badge value-badge-table ${valueClass(decision.value_label)}`} title={decision.value_reason || ""}>
      <strong>{decision.value_label}</strong>
      <em>{marketLabel(decision.market)}</em>
      <small>
        quota {formatOdd(decision.odd)} · fair {formatOdd(decision.fair_odd)} · edge {formatEdge(decision.edge)}
      </small>
    </span>
  );
}
