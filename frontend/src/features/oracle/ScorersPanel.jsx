import { useEffect, useState } from "react";
import { getScorersForFixture } from "../../api";

const REFRESH_MS = 5 * 60 * 1000;
const LOW_PROBABILITY = 0.05;
const ROLE_LABEL = { F: "ATT", M: "CEN", D: "DIF" };

function formatDecimal(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(2).replace(".", ",");
}

function formatTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("it-IT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function groupByTeam(rows, home, away) {
  const map = new Map();
  rows.forEach((row) => {
    const key = row.team_id ?? row.team_name;
    if (!map.has(key)) map.set(key, { key, name: row.team_name, players: [] });
    map.get(key).players.push(row);
  });
  const teams = [...map.values()];
  teams.forEach((team) => team.players.sort((a, b) => Number(b.probability) - Number(a.probability)));
  const rank = (team) => {
    if (home && team.name === home) return 0;
    if (away && team.name === away) return 1;
    return 2;
  };
  teams.sort((a, b) => rank(a) - rank(b) || String(a.name).localeCompare(String(b.name)));
  return teams;
}

function expectedGoalsFor(team, index, home, away, homeXg, awayXg) {
  if (home && team.name === home && homeXg != null) return homeXg;
  if (away && team.name === away && awayXg != null) return awayXg;
  return null;
}

function ScoredMark({ scored }) {
  if (scored === true) return <span className="scorers-mark scorers-mark-yes" title="Ha segnato">✓</span>;
  if (scored === false) return <span className="scorers-mark scorers-mark-no" title="Non ha segnato">✗</span>;
  return null;
}

export default function ScorersPanel({ fixtureId, home, away, homeXg, awayXg, finished }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const rows = data?.rows || [];
  // `supported === false`: competizione fuori dal mercato (es. Champions League) -> il pannello non si mostra.
  const unsupported = data?.supported === false && rows.length === 0;
  const isEmpty = data !== null && rows.length === 0 && !unsupported;

  useEffect(() => {
    if (!fixtureId) return undefined;
    let cancelled = false;
    setLoading(true);
    setError("");
    getScorersForFixture(fixtureId)
      .then((payload) => { if (!cancelled) setData(payload); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fixtureId]);

  // Ricarica ogni 5 minuti solo finche' la formazione non e' ufficiale e la partita non e' finita.
  useEffect(() => {
    if (!fixtureId || finished || !isEmpty) return undefined;
    const id = setInterval(() => {
      getScorersForFixture(fixtureId)
        .then((payload) => { setData(payload); setError(""); })
        .catch(() => {});
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [fixtureId, finished, isEmpty]);

  if (unsupported) return null;

  const teams = groupByTeam(rows, home, away);
  const first = rows[0];

  return (
    <section className="panel" id="or-scorers">
      <div className="panel-header">
        <h4>Possibili marcatori</h4>
        {first && (
          <small className="scorers-meta">
            {first.model_version ? `modello ${first.model_version}` : ""}
            {first.created_at ? ` · formazione ${formatTime(first.created_at)}` : ""}
          </small>
        )}
      </div>

      {loading && !data && <div className="empty-panel">Caricamento possibili marcatori...</div>}
      {error && !data && <div className="empty-panel">Possibili marcatori non disponibili al momento.</div>}
      {isEmpty && (
        <div className="empty-panel">
          Formazione non ancora ufficiale: i possibili marcatori compaiono circa un'ora prima del calcio d'inizio.
        </div>
      )}

      {teams.length > 0 && (
        <div className="scorers-grid">
          {teams.map((team, idx) => {
            const sum = team.players.reduce((acc, p) => acc + Number(p.probability || 0), 0);
            const xg = expectedGoalsFor(team, idx, home, away, homeXg, awayXg);
            const max = Math.max(...team.players.map((p) => Number(p.probability) || 0), 0.0001);
            return (
              <div className="scorers-team" key={team.key}>
                <div className="scorers-team-head">
                  <span>{team.name}</span>
                  <span>P(segna)</span>
                </div>
                {team.players.map((p) => {
                  const prob = Number(p.probability) || 0;
                  return (
                    <div
                      className={`scorers-row${prob < LOW_PROBABILITY ? " scorers-row-low" : ""}`}
                      key={`${p.player_id}-${p.revision}`}
                    >
                      <span className="scorers-role">{ROLE_LABEL[p.position] || p.position}</span>
                      <span className="scorers-name">{p.player_name} <ScoredMark scored={p.scored} /></span>
                      <span className="scorers-pct">{Math.round(prob * 100)}%</span>
                      <div className="scorers-bar"><i style={{ width: `${Math.min(100, (prob / max) * 100)}%` }} /></div>
                    </div>
                  );
                })}
                <div className="scorers-sum">
                  Somma <b>{formatDecimal(sum)}</b>
                  {xg != null && <> · gol attesi <b>{formatDecimal(xg)}</b></>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <p className="scorers-note">
        Probabilità di segnare almeno un gol nei tempi regolamentari, autogol esclusi. Non è una quota.
      </p>
    </section>
  );
}
