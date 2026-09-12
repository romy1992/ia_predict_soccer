import { marketLabel } from "../shared/formatters";

// Porting fedele dell'SVG costruito imperativamente nell'Artifact di
// riferimento (vedi PROMPT_model_diagnostics_web_section.md) in JSX
// dichiarativo - stesse dimensioni/margini/scale, cosi' il chart resta
// pixel-identico al mockup approvato.
const WIDTH = 560;
const HEIGHT = 380;
const MARGIN = { l: 46, r: 14, t: 14, b: 38 };
const PLOT_WIDTH = WIDTH - MARGIN.l - MARGIN.r;
const PLOT_HEIGHT = HEIGHT - MARGIN.t - MARGIN.b;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const SCALE_COLORS = ["var(--scale-1)", "var(--scale-2)", "var(--scale-3)", "var(--scale-4)"];

function scaleX(value) {
  return MARGIN.l + value * PLOT_WIDTH;
}

function scaleY(value) {
  return MARGIN.t + (1 - value) * PLOT_HEIGHT;
}

// Le 4 tonalita' predefinite (--scale-1..4) coprono il caso tipico di
// poche linee in confronto; oltre la quarta genera tonalita' aggiuntive
// ruotando l'hue HSL, cosi' il chart resta leggibile anche quando i
// mercati diagnosticabili cresceranno (nuovi mercati = nuove entry qui,
// mai un limite fisso a 4 come nel mockup originale).
function seriesColor(index) {
  if (index < SCALE_COLORS.length) {
    return SCALE_COLORS[index];
  }
  const hue = (210 + index * 47) % 360;
  return `hsl(${hue}deg 55% 45%)`;
}

export default function RocComparisonChart({ markets }) {
  return (
    <div className="roc-panel">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Curve ROC per mercato"
      >
        <rect x={MARGIN.l} y={MARGIN.t} width={PLOT_WIDTH} height={PLOT_HEIGHT} className="roc-frame" />
        {TICKS.map((t) => (
          <g key={`tick-${t}`}>
            <line
              x1={scaleX(t)}
              x2={scaleX(t)}
              y1={MARGIN.t}
              y2={MARGIN.t + PLOT_HEIGHT}
              className="roc-tick"
              opacity="0.5"
            />
            <line
              x1={MARGIN.l}
              x2={MARGIN.l + PLOT_WIDTH}
              y1={scaleY(t)}
              y2={scaleY(t)}
              className="roc-tick"
              opacity="0.5"
            />
            <text x={scaleX(t)} y={MARGIN.t + PLOT_HEIGHT + 18} textAnchor="middle" className="roc-axis-label">
              {t.toFixed(2)}
            </text>
            <text x={MARGIN.l - 10} y={scaleY(t) + 4} textAnchor="end" className="roc-axis-label">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={MARGIN.l + PLOT_WIDTH / 2} y={HEIGHT - 4} textAnchor="middle" className="roc-axis-label">
          tasso di falsi positivi
        </text>
        <text
          x={-(MARGIN.t + PLOT_HEIGHT / 2)}
          y={13}
          textAnchor="middle"
          className="roc-axis-label"
          transform="rotate(-90)"
        >
          tasso di veri positivi
        </text>
        <line x1={scaleX(0)} y1={scaleY(0)} x2={scaleX(1)} y2={scaleY(1)} className="roc-diag" />
        {markets.map((entry, index) => {
          const points = (entry.roc_fpr || [])
            .map((fpr, i) => `${scaleX(fpr)},${scaleY(entry.roc_tpr[i])}`)
            .join(" ");
          return <polyline key={entry.market} points={points} className="roc-line" stroke={seriesColor(index)} />;
        })}
      </svg>
      <div className="legend">
        {markets.map((entry, index) => (
          <div className="legend-row" key={entry.market}>
            <span className="swatch" style={{ background: seriesColor(index) }} />
            <span className="lab">{marketLabel(entry.market)}</span>
            <span className="auc">AUC {entry.auc.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
