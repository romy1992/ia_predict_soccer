import { phaseLabel } from "../../shared/formatters";

/**
 * Tab di filtro per fase partita (Tutte/Da giocare/In diretta/Finite),
 * riusato sia da Match Center (4 fasi) sia da Dashboard (senza "live",
 * gia' coperta dal pannello "Partite in diretta" dedicato).
 */
export default function PhaseTabs({ phases, value, onChange }) {
  return (
    <div className="tabs">
      {phases.map((item) => (
        <button
          key={item}
          type="button"
          className={value === item ? "tab active" : "tab"}
          onClick={() => onChange(item)}
        >
          {item === "all" ? "Tutte" : phaseLabel(item)}
        </button>
      ))}
    </div>
  );
}

