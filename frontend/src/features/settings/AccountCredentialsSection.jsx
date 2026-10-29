import { useState } from "react";
import { changeCredentials } from "../../api";

const MIN_PASSWORD_LENGTH = 8;

function readDetail(err) {
  const raw = err instanceof Error ? err.message : "";
  try {
    const detail = JSON.parse(raw)?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  } catch (_) {
    // messaggio non JSON: mostrato com'e'
  }
  return raw || "Operazione non riuscita.";
}

export default function AccountCredentialsSection() {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setMessage("");
    const wantedUsername = newUsername.trim();
    if (!wantedUsername && !newPassword) {
      setError("Inserisci un nuovo username e/o una nuova password.");
      return;
    }
    if (newPassword && newPassword.length < MIN_PASSWORD_LENGTH) {
      setError(`La nuova password deve avere almeno ${MIN_PASSWORD_LENGTH} caratteri.`);
      return;
    }
    if (newPassword && newPassword !== confirmPassword) {
      setError("Le due password non coincidono.");
      return;
    }
    setBusy(true);
    try {
      const result = await changeCredentials(currentPassword, wantedUsername, newPassword);
      setCurrentPassword("");
      setNewUsername("");
      setNewPassword("");
      setConfirmPassword("");
      setMessage(`Credenziali aggiornate. Account attuale: ${result?.username ?? ""}`);
    } catch (err) {
      setError(readDetail(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Credenziali di accesso</h2>
      </div>
      <p>Lascia vuoto ciò che non vuoi cambiare. Dopo il cambio, le altre sessioni aperte vengono disconnesse.</p>
      <form onSubmit={handleSubmit}>
        <label>
          Nuovo username
          <input type="text" value={newUsername} onChange={(e) => setNewUsername(e.target.value)} autoComplete="username" disabled={busy} />
        </label>
        <label>
          Nuova password
          <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} autoComplete="new-password" disabled={busy} />
        </label>
        <label>
          Conferma nuova password
          <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} autoComplete="new-password" disabled={busy} />
        </label>
        <label>
          Password attuale
          <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} autoComplete="current-password" required disabled={busy} />
        </label>
        <button type="submit" disabled={busy}>Aggiorna credenziali</button>
      </form>
      {message && <div className="success-box">{message}</div>}
      {error && <div className="error-box">Errore: {error}</div>}
    </section>
  );
}
