import { useEffect, useState } from "react";
import { getModelLegend } from "../../api";
import { formatPercent } from "./formatters";

const DIRECTION_LABEL = {
  over: "Over",
  under: "Under",
};

function directionLabel(entry) {
  if (entry.policy_family === "generic_decision_policy") {
    return "PLAY/BORDERLINE";
  }
  return DIRECTION_LABEL[entry.direction] || "-";
}

/**
 * Legenda "quale soglia, quale direzione conviene giocare" per ogni
 * mercato (2026-09-19, richiesta esplicita operatore) - GET /models/legend,
 * calcolato dal backend (`src/api/main.py::model_legend`), espone
 * direttamente le policy di decisione gia' in uso (`over_signal_policy.py`,
 * `line_market_signal_policy.py`, `DecisionPolicy` generica): nessun numero
 * ricalcolato o duplicato qui, solo rendering.
 *
 * Collassata di default (`<details>` nativo, niente stato globale): la
 * mostrano sia Partite che Schedine, stesso componente, nessuna
 * duplicazione di markup/fetch.
 */
export default function ModelLegend() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (!isOpen || report || isLoading) {
      return;
    }
    setIsLoading(true);
    setError(null);
    getModelLegend()
      .then((data) => setReport(data))
      .catch((err) => setError(err?.message || "Errore di rete"))
      .finally(() => setIsLoading(false));
  }, [isOpen, report, isLoading]);

  return (
    <details className="detail-block model-legend" open={isOpen} onToggle={(e) => setIsOpen(e.target.open)}>
      <summary>Legenda modelli — soglie e direzione consigliata</summary>

      {isLoading && <div className="info-box">Carico la legenda...</div>}
      {error && <div className="error-box">Errore API: {error}</div>}

      {report && (
        <>
          <p className="dek">
            Per ogni mercato: la soglia di probabilita' che fa scattare il segnale "conviene giocare" e la direzione
            che il modello predilige (Over o Under). Indipendente dal pick mostrato in dashboard (soglia fissa 0.5).
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mercato</th>
                  <th>Direzione</th>
                  <th>Soglia</th>
                  <th>Precisione/Accuracy attesa</th>
                  <th>Stato</th>
                  <th>Nota</th>
                </tr>
              </thead>
              <tbody>
                {report.entries.map((entry) => (
                  <tr key={entry.market}>
                    <td>{entry.market_label}</td>
                    <td>{directionLabel(entry)}</td>
                    <td>{entry.threshold !== null && entry.threshold !== undefined ? formatPercent(entry.threshold) : "-"}</td>
                    <td>
                      {formatPercent(entry.expected_precision ?? entry.expected_accuracy)}
                      {entry.expected_recall !== null && entry.expected_recall !== undefined && (
                        <span className="match-sub"> (recall {formatPercent(entry.expected_recall)})</span>
                      )}
                    </td>
                    <td>
                      {entry.active_in_production ? (
                        <span className="legend-badge legend-badge-active">attivo</span>
                      ) : (
                        <span className="legend-badge legend-badge-reference">solo riferimento</span>
                      )}
                    </td>
                    <td className="match-sub">{entry.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </details>
  );
}
