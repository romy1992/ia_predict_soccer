// Chip di navigazione per le pagine di dettaglio lunghe (solo mobile, vedi
// `.jump-nav` in theme.css): scorre fino alla sezione con l'id indicato.
export default function JumpNav({ items }) {
  return (
    <nav className="jump-nav" aria-label="Sezioni">
      {items.map(([id, label]) => (
        <button
          key={id}
          type="button"
          className="tab"
          onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" })}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}
