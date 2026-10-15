import { useEffect, useRef, useState } from "react";
import { getPlayerCardsForFixture, getScorersForFixture } from "../../../api";
import { groupByTeam } from "../../oracle/ScorersPanel";

const ROLE_LABEL = { F: "ATT", M: "CEN", D: "DIF" };
const LOW_PROBABILITY = 0.05;

// Un mercato giocatore = una scheda: cosa chiedere al motore e come descriverlo.
const KINDS = {
  scorers: {
    label: "Marcatori",
    load: getScorersForFixture,
    hitField: "scored",
    hitYes: "Ha segnato",
    hitNo: "Non ha segnato",
    column: "P(segna)",
    pending: "Formazione non ancora ufficiale: i possibili marcatori compaiono circa un'ora prima del calcio d'inizio.",
    note: "Probabilità di segnare almeno un gol nei tempi regolamentari, autogol esclusi. Non è una quota.",
  },
  cards: {
    label: "Gialli",
    load: getPlayerCardsForFixture,
    hitField: "yellow",
    hitYes: "Ammonito",
    hitNo: "Non ammonito",
    column: "P(giallo)",
    pending: "Formazione non ancora ufficiale: i gialli probabili compaiono circa un'ora prima del calcio d'inizio.",
    note: "Probabilità di ricevere almeno un'ammonizione nei tempi regolamentari (secondo giallo incluso). Non è una quota.",
  },
};

function HitMark({ kind, value }) {
  if (value === true) return <span className="scorers-mark scorers-mark-yes" title={KINDS[kind].hitYes}>✓</span>;
  if (value === false) return <span className="scorers-mark scorers-mark-no" title={KINDS[kind].hitNo}>✗</span>;
  return null;
}

function PlayerList({ kind, state, home, away }) {
  const cfg = KINDS[kind];
  if (!state || state.loading) return <div className="empty-panel">Caricamento {cfg.label.toLowerCase()}...</div>;
  if (state.error) return <div className="empty-panel">{cfg.label} non disponibili al momento.</div>;
  const rows = state.data?.rows || [];
  if (state.data?.supported === false && rows.length === 0) {
    return <div className="empty-panel">Competizione non coperta da questo mercato.</div>;
  }
  if (rows.length === 0) return <div className="empty-panel">{cfg.pending}</div>;

  const teams = groupByTeam(rows, home, away);
  return (
    <>
      <div className="scorers-grid">
        {teams.map((team) => {
          const max = Math.max(...team.players.map((p) => Number(p.probability) || 0), 0.0001);
          const sum = team.players.reduce((acc, p) => acc + Number(p.probability || 0), 0);
          return (
            <div className="scorers-team" key={team.key}>
              <div className="scorers-team-head"><span>{team.name}</span><span>{cfg.column}</span></div>
              {team.players.map((p) => {
                const prob = Number(p.probability) || 0;
                return (
                  <div className={`scorers-row${prob < LOW_PROBABILITY ? " scorers-row-low" : ""}`} key={`${p.player_id}-${p.revision}`}>
                    <span className="scorers-role">{ROLE_LABEL[p.position] || p.position}</span>
                    <span className="scorers-name">{p.player_name} <HitMark kind={kind} value={p[cfg.hitField]} /></span>
                    <span className="scorers-pct">{Math.round(prob * 100)}%</span>
                    <div className="scorers-bar"><i style={{ width: `${Math.min(100, (prob / max) * 100)}%` }} /></div>
                  </div>
                );
              })}
              <div className="scorers-sum">Somma <b>{sum.toFixed(2).replace(".", ",")}</b></div>
            </div>
          );
        })}
      </div>
      <p className="scorers-note">{cfg.note}</p>
    </>
  );
}

/**
 * Modale "Giocatori" di una partita: probabilita' dei marcatori e dei gialli dei titolari (formazione
 * ufficiale). Ogni scheda carica i dati solo quando la si apre.
 */
export default function PlayersModal({ row, onClose }) {
  const [tab, setTab] = useState("scorers");
  const [states, setStates] = useState({});
  const fixtureId = row?.fixture_id;
  const requested = useRef(new Set());
  const current = useRef(fixtureId);

  useEffect(() => {
    current.current = fixtureId;
    requested.current = new Set();
    setStates({});
    setTab("scorers");
  }, [fixtureId]);

  // Ogni scheda si carica una volta sola, alla prima apertura (ref: nessuna richiesta annullata da un re-render).
  useEffect(() => {
    if (!fixtureId || requested.current.has(tab)) return;
    requested.current.add(tab);
    const forFixture = fixtureId;
    setStates((prev) => ({ ...prev, [tab]: { loading: true } }));
    KINDS[tab].load(forFixture)
      .then((data) => {
        if (current.current === forFixture) setStates((prev) => ({ ...prev, [tab]: { data } }));
      })
      .catch((err) => {
        if (current.current === forFixture) {
          setStates((prev) => ({ ...prev, [tab]: { error: err instanceof Error ? err.message : String(err) } }));
        }
      });
  }, [fixtureId, tab]);

  if (!row) return null;
  return (
    <section className="panel detail-panel players-modal" aria-label="Giocatori della partita">
      <div className="panel-header">
        <h3>{row.home} vs {row.away}</h3>
        <button className="btn-secondary" onClick={onClose}>Chiudi</button>
      </div>
      <div className="players-modal-tabs" role="tablist">
        {Object.entries(KINDS).map(([key, cfg]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            className={tab === key ? "btn-primary" : "btn-secondary"}
            onClick={() => setTab(key)}
          >
            {cfg.label}
          </button>
        ))}
      </div>
      <PlayerList kind={tab} state={states[tab]} home={row.home} away={row.away} />
    </section>
  );
}
