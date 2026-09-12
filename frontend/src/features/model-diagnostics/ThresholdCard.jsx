import { marketClassLabels, marketLabel } from "../shared/formatters";

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function fmt(value) {
  return value.toFixed(3);
}

// Stesso schema di shading a intensita' relativa dell'Artifact di
// riferimento: la cella con piu' casi nella confusion matrix e' la piu'
// scura, le altre scalano linearmente - solo una guida visiva, i numeri
// esatti restano nelle celle.
function shade(n, maxN) {
  const t = maxN ? n / maxN : 0;
  const alpha = 0.08 + t * 0.42;
  return `color-mix(in oklab, var(--ink) ${Math.round(alpha * 100)}%, var(--surface))`;
}

export default function ThresholdCard({ entry }) {
  const { market, n_oof: nOof, champion, stage, accuracy, auc, cm, class0, class1, weighted } = entry;
  // Etichette lette dalla mappa mercato->etichette (mai hardcoded "Under"/
  // "Over": qui dentro convivono anche h2h/goal_no_goal/dc/corners/cards).
  const [label0, label1] = marketClassLabels(market);
  const actual0 = cm.tn + cm.fp;
  const actual1 = cm.fn + cm.tp;
  const maxN = Math.max(cm.tn, cm.fp, cm.fn, cm.tp);

  const metricRows = [
    { label: label0, color: "var(--under)", data: class0 },
    { label: label1, color: "var(--over)", data: class1 },
  ];

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h3 className="display">{marketLabel(market)}</h3>
          <div className="sub">
            n={nOof.toLocaleString("it-IT")} righe OOF · {champion}
          </div>
        </div>
        <span className="badge">{stage || "n/d"}</span>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="k">Accuracy</div>
          <div className="v">{pct(accuracy)}</div>
        </div>
        <div className="stat">
          <div className="k">ROC AUC</div>
          <div className="v">{fmt(auc)}</div>
        </div>
        <div className="stat">
          <div className="k">Weighted F1</div>
          <div className="v">{fmt(weighted.f1)}</div>
        </div>
      </div>

      <div className="cm-block">
        <div className="cm-xlabs">
          <span>previsto: {label0}</span>
          <span>previsto: {label1}</span>
        </div>
        <div className="cm-row">
          <div className="cm-rowlab">
            reale
            <br />
            {label0}
          </div>
          <div className="cm-cell correct" style={{ background: shade(cm.tn, maxN) }}>
            <span className="n">{cm.tn}</span>
            <span className="p">{pct(cm.tn / actual0)} del reale</span>
          </div>
          <div className="cm-cell" style={{ background: shade(cm.fp, maxN) }}>
            <span className="n">{cm.fp}</span>
            <span className="p">{pct(cm.fp / actual0)} del reale</span>
          </div>
        </div>
        <div className="cm-row">
          <div className="cm-rowlab">
            reale
            <br />
            {label1}
          </div>
          <div className="cm-cell" style={{ background: shade(cm.fn, maxN) }}>
            <span className="n">{cm.fn}</span>
            <span className="p">{pct(cm.fn / actual1)} del reale</span>
          </div>
          <div className="cm-cell correct" style={{ background: shade(cm.tp, maxN) }}>
            <span className="n">{cm.tp}</span>
            <span className="p">{pct(cm.tp / actual1)} del reale</span>
          </div>
        </div>
      </div>

      <div className="metric-table">
        <div className="row head">
          <span>Classe</span>
          <span>Precision</span>
          <span>Recall</span>
          <span>F1</span>
        </div>
        {metricRows.map((row) => (
          <div className="row" key={row.label}>
            <span className="name">
              <span className="sw" style={{ background: row.color }} />
              {row.label}
            </span>
            <span className="val">{fmt(row.data.precision)}</span>
            <span className="val">{fmt(row.data.recall)}</span>
            <span className="val">{fmt(row.data.f1)}</span>
          </div>
        ))}
        <div className="row" style={{ borderTop: "1px solid var(--line-soft)", paddingTop: "6px", marginTop: "1px" }}>
          <span className="name" style={{ color: "var(--muted)", fontWeight: 500 }}>
            Weighted
          </span>
          <span className="val" style={{ color: "var(--ink-soft)" }}>
            {fmt(weighted.precision)}
          </span>
          <span className="val" style={{ color: "var(--ink-soft)" }}>
            {fmt(weighted.recall)}
          </span>
          <span className="val" style={{ color: "var(--ink-soft)" }}>
            {fmt(weighted.f1)}
          </span>
        </div>
      </div>
    </div>
  );
}
