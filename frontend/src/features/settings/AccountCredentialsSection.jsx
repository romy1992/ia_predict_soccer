import { useEffect, useState } from "react";
import { changeCredentials, getProviderKeys, revealCredentials, revealProviderKey, updateProviderKey } from "../../api";
import SecretField from "./SecretField";

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

function CurrentCredentials({ refreshKey }) {
  const [username, setUsername] = useState("");
  useEffect(() => {
    revealCredentials().then((r) => setUsername(r?.username || "")).catch(() => setUsername(""));
  }, [refreshKey]);
  return (
    <div className="cred-grid" key={refreshKey}>
      <label className="cred-field">
        <span className="cred-label">Username attuale</span>
        <span className="cred-field-row cred-field-pad">
          <input type="text" value={username} readOnly />
        </span>
      </label>
      <SecretField label="Password salvata" onReveal={async () => (await revealCredentials())?.password} />
    </div>
  );
}

function ProviderKeysSection() {
  const [keys, setKeys] = useState([]);
  const [values, setValues] = useState({});
  const [currentPassword, setCurrentPassword] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [version, setVersion] = useState(0);

  useEffect(() => {
    getProviderKeys().then((r) => setKeys(r?.keys || [])).catch((err) => setError(readDetail(err)));
  }, [version]);

  async function save(name) {
    setError("");
    setMessage("");
    const value = (values[name] || "").trim();
    if (value.length < 8) {
      setError("La chiave deve avere almeno 8 caratteri.");
      return;
    }
    if (!currentPassword) {
      setError("Inserisci la password attuale per confermare.");
      return;
    }
    setBusy(name);
    try {
      await updateProviderKey(name, value, currentPassword);
      setValues((v) => ({ ...v, [name]: "" }));
      setMessage("Chiave aggiornata: vale subito per i job.");
      setVersion((n) => n + 1);
    } catch (err) {
      setError(readDetail(err));
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="panel cred-panel">
      <div className="panel-header">
        <h2>Chiavi dei provider</h2>
      </div>
      <p className="muted">Salvate cifrate nel database: le variabili d'ambiente servono solo come seme al primo avvio.</p>
      <div className="cred-block">
        <div className="cred-grid">
          <SecretField
            label="Password attuale (conferma per salvare)"
            value={currentPassword}
            onChange={setCurrentPassword}
            autoComplete="current-password"
            disabled={Boolean(busy)}
          />
        </div>
      </div>
      {keys.map((k) => (
        <div className="cred-block" key={`${k.name}-${version}`}>
          <h3>{k.label}</h3>
          <div className="cred-grid">
            <SecretField label="Chiave attuale" onReveal={async () => (await revealProviderKey(k.name))?.value} />
            <SecretField
              label="Nuova chiave"
              value={values[k.name] || ""}
              onChange={(v) => setValues((cur) => ({ ...cur, [k.name]: v }))}
              placeholder={k.configured ? "Incolla la nuova chiave" : "Non configurata: incolla la chiave"}
              disabled={Boolean(busy)}
            />
          </div>
          <div className="cred-actions">
            <button type="button" className="btn-primary" onClick={() => save(k.name)} disabled={Boolean(busy)}>
              Salva {k.label}
            </button>
          </div>
        </div>
      ))}
      {message && <div className="success-box">{message}</div>}
      {error && <div className="error-box">Errore: {error}</div>}
    </section>
  );
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
    <div className="stack">
      <section className="panel cred-panel">
        <div className="panel-header">
          <h2>Credenziali di accesso</h2>
        </div>
        <CurrentCredentials refreshKey={message} />
        <form className="cred-block" onSubmit={handleSubmit}>
          <h3>Cambia credenziali</h3>
          <p className="muted">
            Lascia vuoto ciò che non vuoi cambiare. Dopo il cambio, le altre sessioni aperte vengono disconnesse.
          </p>
          <div className="cred-grid">
            <label className="cred-field">
              <span className="cred-label">Nuovo username</span>
              <span className="cred-field-row cred-field-pad">
                <input type="text" value={newUsername} onChange={(e) => setNewUsername(e.target.value)} autoComplete="username" disabled={busy} />
              </span>
            </label>
            <SecretField label="Nuova password" value={newPassword} onChange={setNewPassword} autoComplete="new-password" disabled={busy} />
            <SecretField label="Conferma nuova password" value={confirmPassword} onChange={setConfirmPassword} autoComplete="new-password" disabled={busy} />
            <SecretField label="Password attuale" value={currentPassword} onChange={setCurrentPassword} autoComplete="current-password" required disabled={busy} />
          </div>
          <div className="cred-actions">
            <button type="submit" className="btn-primary" disabled={busy}>Aggiorna credenziali</button>
          </div>
        </form>
        {message && <div className="success-box">{message}</div>}
        {error && <div className="error-box">Errore: {error}</div>}
      </section>
      <ProviderKeysSection />
    </div>
  );
}
