import { useState } from "react";

/**
 * Campo con "occhio". Se `onReveal` e' passato, il valore vero si chiede al
 * server solo al primo clic (e mai mostrato prima); altrimenti l'occhio
 * mostra/nasconde quello digitato.
 */
export default function SecretField({ label, value, onChange, onReveal, readOnly = false, placeholder = "", autoComplete = "off", disabled = false, required = false }) {
  const [visible, setVisible] = useState(false);
  const [revealed, setRevealed] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function toggle() {
    setError("");
    if (visible) {
      setVisible(false);
      return;
    }
    if (onReveal && revealed === null) {
      setLoading(true);
      try {
        const plain = await onReveal();
        if (plain === null || plain === undefined || plain === "") {
          setError("Non disponibile: reimposta il valore una volta per poterlo vedere.");
          return;
        }
        setRevealed(plain);
      } catch (err) {
        setError("Impossibile leggere il valore.");
        return;
      } finally {
        setLoading(false);
      }
    }
    setVisible(true);
  }

  const shown = onReveal ? (visible ? revealed ?? "" : value ?? "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022") : value;
  return (
    <label className="cred-field">
      <span className="cred-label">{label}</span>
      <span className="cred-field-row">
        <input
          type={onReveal || visible ? "text" : "password"}
          value={shown}
          onChange={onChange ? (e) => onChange(e.target.value) : undefined}
          readOnly={readOnly || Boolean(onReveal)}
          placeholder={placeholder}
          autoComplete={autoComplete}
          disabled={disabled}
          required={required}
        />
        <button type="button" className="btn-secondary cred-eye" onClick={toggle} disabled={loading} aria-label={visible ? `Nascondi ${label}` : `Mostra ${label}`} title={visible ? "Nascondi" : "Mostra"}>
          {visible ? "\u{1F648}" : "\u{1F441}"}
        </button>
      </span>
      {error && <small className="cred-error">{error}</small>}
    </label>
  );
}
