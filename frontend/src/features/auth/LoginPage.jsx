import { useState } from "react";
import { login } from "../../api";

export default function LoginPage({ onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      const result = await login(username, password);
      onLoginSuccess(result.username);
    } catch (_) {
      setError("Credenziali non valide");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="stack" style={{ maxWidth: 360, margin: "10vh auto" }}>
      <section className="panel">
        <div className="panel-header">
          <h3>Accedi</h3>
        </div>
        <form className="inline-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
          </label>
          <label>
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
          <button className="btn-primary" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Accesso..." : "Accedi"}
          </button>
        </form>
        {error && <div className="error-box">{error}</div>}
      </section>
    </section>
  );
}
