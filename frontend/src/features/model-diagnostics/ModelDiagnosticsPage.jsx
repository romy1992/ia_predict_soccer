import RocComparisonChart from "./RocComparisonChart";
import ThresholdCard from "./ThresholdCard";
import "./model-diagnostics.css";

const STATUS_REASONS = {
  no_model: "nessun modello registrato",
  insufficient_data: "dati insufficienti per il walk-forward OOF",
  single_class_oof: "OOF con una sola classe osservata (confusion matrix non calcolabile)",
};

function statusReason(status) {
  if (STATUS_REASONS[status]) {
    return STATUS_REASONS[status];
  }
  return status || "sconosciuto";
}

/**
 * Model Diagnostics (2026-09-12): report out-of-fold (confusion matrix,
 * precision/recall/F1 per classe, ROC+AUC) per OGNI mercato con un modello
 * registrato - GET /models/diagnostics, calcolato dal backend
 * (`src/ml/evaluation/model_diagnostics_service.py`); questo componente fa
 * SOLO rendering, nessun ricalcolo di metriche lato client (stesso
 * principio gia' seguito da MonitoringPage/DataQualityPage).
 */
export default function ModelDiagnosticsPage({ report, isLoading, error, onRefresh }) {
  const markets = report?.markets || [];
  const okMarkets = markets.filter((entry) => entry.status === "ok");
  const skippedMarkets = markets.filter((entry) => entry.status !== "ok");

  return (
    <div className="model-diagnostics-page">
      <header className="top">
        <span className="eyebrow">Soccer Oracle · model diagnostics</span>
        <h1 className="display">
          Model Diagnostics <span className="line">— report out-of-fold per mercato</span>
        </h1>
        <p className="dek">
          Report di classificazione out-of-fold per ogni mercato con un modello registrato: confusion matrix,
          precision/recall/F1 per classe e curva ROC, alla soglia di decisione di default (p ≥ 0.5).
        </p>
      </header>

      <div className="diag-toolbar">
        <button onClick={onRefresh} disabled={isLoading}>
          {isLoading ? "Calcolo in corso..." : "Ricalcola"}
        </button>
        {report?.generated_at && (
          <span className="generated-at">Generato il {new Date(report.generated_at).toLocaleString("it-IT")}</span>
        )}
      </div>

      {error && <div className="error-box">Errore API: {error}</div>}
      {isLoading && !report && <div className="info-box">Calcolo diagnostica in corso (walk-forward OOF)...</div>}

      <div className="callout">
        <span className="mark">i</span>
        <p>
          <strong>Come vengono calcolati questi numeri.</strong> Ogni previsione qui sotto arriva da uno split
          walk-forward out-of-fold (finestra espandente, 5 fold): nessuna riga viene mai valutata da un modello che
          l'ha vista in training. La pipeline esatta di ogni champion (selector, calibratore, base learner) viene
          clonata e ri-addestrata fold per fold. Log loss, Brier ed ECE — le metriche usate per scegliere questi
          champion — restano in <code>ModelRegistry</code>; la vista per classe qui sotto (a soglia fissa{" "}
          <code>p ≥ 0.5</code>) e' complementare e puo' raccontare una storia diversa.
        </p>
      </div>

      {okMarkets.length > 0 && (
        <section id="roc-section">
          <div className="section-head">
            <h2 className="display">Discriminazione, tutti i mercati</h2>
            <span className="note">ROC · tasso di veri positivi vs falsi positivi</span>
          </div>
          <RocComparisonChart markets={okMarkets} />
        </section>
      )}

      {okMarkets.length > 0 && (
        <section id="cards-section">
          <div className="section-head">
            <h2 className="display">Dettaglio per soglia</h2>
            <span className="note">confusion matrix · precision / recall / F1</span>
          </div>
          <div className="grid4">
            {okMarkets.map((entry) => (
              <ThresholdCard key={entry.market} entry={entry} />
            ))}
          </div>
        </section>
      )}

      {skippedMarkets.length > 0 && (
        <div className="skipped-note">
          Mercati esclusi da questo report:{" "}
          {skippedMarkets.map((entry) => `${entry.market} (${statusReason(entry.status)})`).join(" · ")}
        </div>
      )}

      {!isLoading && okMarkets.length === 0 && skippedMarkets.length === 0 && (
        <div className="empty-state">Nessun mercato diagnosticabile al momento.</div>
      )}

      <footer>
        <p>
          Metriche calcolate on-demand sui modelli attualmente registrati (produzione se presente, altrimenti
          l'ultimo candidato) · esclude "1x2" (multiclasse) e i mercati a linea configurabile (corners_line_*/
          cards_line_*) · cache lato server 15 minuti.
        </p>
      </footer>
    </div>
  );
}
