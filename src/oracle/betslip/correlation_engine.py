"""Correlation Engine per Schedina Oracle (SLIP-02, Fase SCHEDINA).

Le pick candidate selezionate dal Pick Pool (SLIP-01, `pick_pool.py`) sono
oggi trattate come fossero sempre STATISTICAMENTE INDIPENDENTI quando si
combinano piu' esiti della STESSA partita in un'unica schedina (es.
"Over 2.5" + "BTTS Yes" + "Home" sullo stesso match): moltiplicare
semplicemente le probabilita' modello (`p_model_a * p_model_b * ...`) e'
corretto SOLO se gli eventi sono davvero indipendenti, ma molte
combinazioni same-match NON lo sono affatto - alcune sono addirittura
LOGICAMENTE IMPOSSIBILI insieme (es. "Under 1.5" + "BTTS Yes": se entrambe
le squadre segnano il totale e' per definizione >=2, incompatibile con
"meno di 1.5 gol totali"), altre sono fortemente ridondanti (es.
"Over 3.5" + "Over 2.5": il primo implica matematicamente il secondo).

Questo modulo (PURO, nessun DB/IO, stesso principio di `pick_pool.py`)
implementa le "regole same-match" richieste dal task:

1. Regole same-match: le regole di correlazione si applicano SOLO tra
   pick della STESSA fixture - due pick su partite diverse sono SEMPRE
   considerate indipendenti (nessuna correlazione incrociata tra partite
   diverse e' supportata, ne' avrebbe senso).
2. Dipendenze goal/BTTS/1X2: copre esplicitamente le relazioni tra
   Totals (`under_over_*`, soglie gol 1.5/2.5/3.5/4.5), BTTS
   (`goal_no_goal`/`btts`, Yes/No) e 1X2/Double Chance (`h2h`/`dc`).
3. Penalita' o esclusione: ogni coppia viene classificata con una
   `severity` esplicita:
   - `EXCLUDE`: le due pick sono LOGICAMENTE incompatibili nella stessa
     partita (probabilita' congiunta = 0 per costruzione, es. "Over 2.5"
     e "Under 1.5") - una schedina con questa coppia non deve MAI essere
     proposta.
   - `PENALTY`: le due pick sono possibili insieme ma NON indipendenti
     (una implica l'altra, o si sovrappongono fortemente) - la
     probabilita' congiunta "ingenua" (prodotto semplice) sovrastima o
     sottostima quella reale e va corretta.
   - `INDEPENDENT`: nessuna regola nota si applica (o le pick sono su
     fixture diverse) - il prodotto semplice resta l'assunzione corretta.
4. Matrice/regole versionate: `CorrelationRuleSet` (stesso principio di
   `PickPoolPolicy`/`DecisionPolicy`) con una `version` esplicita - una
   nuova taratura (es. una banda di sovrapposizione piu' larga/stretta)
   richiede una nuova istanza con una nuova versione, mai un edit
   silenzioso.

Fondamento delle regole (deliberatamente SOLO logico/insiemistico, MAI un
numero di correlazione "inventato" o stimato staticamente - questo NON e'
un modello statistico, e' un motore di regole):
- Se l'esito A implica l'esito B (A e' un sottoinsieme di B, es.
  "Over 3.5" implica "Over 2.5"), allora per definizione di probabilita'
  P(A e B) = P(A) - un fatto matematico, non una stima. In questi casi la
  probabilita' congiunta corretta e' `min(p_model_a, p_model_b)` (che
  coincide esattamente con P(A) quando i due p_model sono coerenti).
- Se A e B sono mutuamente esclusivi nella stessa partita (es. "Over 2.5"
  e "Under 1.5", oppure due esiti 1X2 diversi), P(A e B) = 0 per
  costruzione: la combinazione e' impossibile e va esclusa, non solo
  penalizzata.
- Per QUALUNQUE coppia di eventi (anche solo parzialmente sovrapposti, non
  in relazione di implicazione stretta) vale SEMPRE la disuguaglianza
  P(A e B) <= min(P(A), P(B)): un fatto di teoria della probabilita', non
  un'approssimazione arbitraria. E' quindi corretto usare `min(p_a, p_b)`
  come stima "prudente" della probabilita' congiunta ogni volta che una
  coppia e' marcata `PENALTY`, al posto del prodotto semplice.

Limite noto (documentato, non un bug): la classificazione si basa sul
testo di `market`/`outcome` gia' normalizzato (case-insensitive). Per il
mercato "h2h" alcuni chiamanti (es. `dashboard_service.py`) possono usare
il NOME della squadra invece di "Home"/"Away" come label del pick: in quel
caso l'esito non e' riconosciuto (nessun match nella tabella token) e la
coppia e' classificata `INDEPENDENT` per default (fail-open: MEGLIO non
segnalare una correlazione reale che bloccare per errore una combinazione
valida per un'etichetta non riconosciuta). Le pick generate con le label
standard (usate da `pick_pool_test.py`/`pick_pool_service_test.py`:
"Home"/"Draw"/"Away", "Home/Draw"/"Draw/Away", "Yes"/"No"/"Goal"/"No Goal",
"Over X"/"Under X") sono invece sempre riconosciute correttamente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from src.oracle.betslip.pick_pool import CandidatePick, PoolPick

# ---------------------------------------------------------------------------
# Severita' (stessa convenzione a stringa di PLAY/BORDERLINE/NO_BET in
# value_engine.py): nessun Enum per restare coerenti con lo stile gia' in
# uso nel progetto per gli esiti/decisioni.
# ---------------------------------------------------------------------------

INDEPENDENT = "INDEPENDENT"
PENALTY = "PENALTY"
EXCLUDE = "EXCLUDE"


# ---------------------------------------------------------------------------
# Normalizzazione (market, outcome) grezzi -> semantica canonica.
# ---------------------------------------------------------------------------

_MATCH_RESULT_MARKETS = frozenset({"h2h", "1x2"})
_DOUBLE_CHANCE_MARKETS = frozenset({"dc", "double_chance"})
_BTTS_MARKETS = frozenset({"goal_no_goal", "btts"})

_BTTS_YES_TOKENS = frozenset({"yes", "goal", "gg", "si"})
_BTTS_NO_TOKENS = frozenset({"no", "nogoal", "no goal", "ng"})

_MATCH_RESULT_TOKENS = {"home": "HOME", "1": "HOME", "draw": "DRAW", "x": "DRAW", "away": "AWAY", "2": "AWAY"}

# Le tre Double Chance sono sottoinsiemi di 2 elementi dei 3 esiti 1X2 di
# base (MARKET-02, `market_double_chance.py::DOUBLE_CHANCE_OUTCOMES`).
DOUBLE_CHANCE_MEMBERS: dict[str, frozenset] = {
    "HOME_DRAW": frozenset({"HOME", "DRAW"}),
    "HOME_AWAY": frozenset({"HOME", "AWAY"}),
    "DRAW_AWAY": frozenset({"DRAW", "AWAY"}),
}
_DOUBLE_CHANCE_TOKENS = {
    "home/draw": "HOME_DRAW", "home draw": "HOME_DRAW", "1x": "HOME_DRAW",
    "home/away": "HOME_AWAY", "home away": "HOME_AWAY", "12": "HOME_AWAY",
    "draw/away": "DRAW_AWAY", "draw away": "DRAW_AWAY", "x2": "DRAW_AWAY",
}

_TOTALS_OUTCOME_RE = re.compile(r"^(over|under)\s*([\d.,]+)$")


def _normalize(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@dataclass(frozen=True)
class SemanticOutcome:
    """Rappresentazione semantica canonica di un (market, outcome) grezzo,
    usata SOLO per classificare le correlazioni - non sostituisce/duplica
    `CandidatePick` (che resta l'unica fonte per market/outcome originali)."""

    family: str  # "MATCH_RESULT" | "DOUBLE_CHANCE" | "BTTS" | "TOTALS"
    tag: str  # es. "HOME" / "HOME_DRAW" / "YES" / "OVER"
    threshold: Optional[float] = None  # SOLO per family TOTALS


def classify_outcome(market: str, outcome: str) -> Optional[SemanticOutcome]:
    """Da (market, outcome) grezzi a `SemanticOutcome`, se riconosciuto.

    Ritorna `None` quando il market/outcome non appartiene a nessuna delle
    family note (fail-open, vedi docstring di modulo): nessuna regola di
    correlazione puo' essere applicata a un outcome non riconosciuto, mai
    un falso positivo che bloccherebbe una combinazione valida."""
    market_norm = _normalize(market)
    outcome_norm = _normalize(outcome)

    if market_norm in _BTTS_MARKETS:
        if outcome_norm in _BTTS_YES_TOKENS:
            return SemanticOutcome(family="BTTS", tag="YES")
        if outcome_norm in _BTTS_NO_TOKENS:
            return SemanticOutcome(family="BTTS", tag="NO")
        return None

    if market_norm in _MATCH_RESULT_MARKETS:
        tag = _MATCH_RESULT_TOKENS.get(outcome_norm)
        return SemanticOutcome(family="MATCH_RESULT", tag=tag) if tag else None

    if market_norm in _DOUBLE_CHANCE_MARKETS:
        tag = _DOUBLE_CHANCE_TOKENS.get(outcome_norm)
        return SemanticOutcome(family="DOUBLE_CHANCE", tag=tag) if tag else None

    if market_norm.startswith("under_over_"):
        outcome_match = _TOTALS_OUTCOME_RE.match(outcome_norm)
        if not outcome_match:
            return None
        direction = outcome_match.group(1).upper()
        threshold = float(outcome_match.group(2).replace(",", "."))
        return SemanticOutcome(family="TOTALS", tag=direction, threshold=threshold)

    return None


# ---------------------------------------------------------------------------
# Matrice di regole versionata.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationRule:
    """Voce puramente descrittiva del catalogo (acceptance criteria
    "matrice versionata"): usata per introspezione/tracciabilita', la
    logica di classificazione vera e propria resta nelle funzioni pure
    sottostanti (mai duplicata qui)."""

    rule_id: str
    families: tuple[str, str]
    description: str


@dataclass(frozen=True)
class CorrelationRuleSet:
    """Parametri versionati della matrice (mai hardcoded inline nel corpo
    delle funzioni senza una versione tracciabile): una nuova taratura
    richiede una nuova istanza con una nuova `version`, mai un edit
    silenzioso di questa (stesso principio di `PickPoolPolicy`/
    `DecisionPolicy`)."""

    version: str = "correlation_ruleset_v1"
    # Scarto massimo (in gol) tra una soglia Over e una Under opposte (o
    # tra BTTS Yes, trattato come "Over 1.5" virtuale essendo BTTS Yes un
    # sottoinsieme stretto di "almeno 2 gol totali", e una soglia O/U)
    # sotto il quale la sovrapposizione e' considerata "stretta" (PENALTY)
    # invece che trascurabile (INDEPENDENT). Valore deliberatamente
    # prudente: le soglie reali sono 1.5/2.5/3.5/4.5, quindi 1.0 = soglie
    # adiacenti nella sequenza standard.
    totals_narrow_band_max_gap: float = 1.0


DEFAULT_CORRELATION_RULESET = CorrelationRuleSet()

CORRELATION_RULE_CATALOG: tuple[CorrelationRule, ...] = (
    CorrelationRule(
        "same_market_mutually_exclusive", ("*", "*"),
        "Stesso mercato, stessa fixture, esiti diversi: un solo esito puo' verificarsi.",
    ),
    CorrelationRule(
        "duplicate_pick", ("*", "*"),
        "Stesso mercato e stesso esito selezionati due volte per la stessa fixture.",
    ),
    CorrelationRule(
        "totals_contradiction", ("TOTALS", "TOTALS"),
        "Over/Under di soglie diverse logicamente incompatibili nella stessa partita.",
    ),
    CorrelationRule(
        "totals_nested_same_direction", ("TOTALS", "TOTALS"),
        "Stessa direzione (Over/Over o Under/Under), soglie diverse: un esito implica l'altro.",
    ),
    CorrelationRule(
        "totals_narrow_band", ("TOTALS", "TOTALS"),
        "Direzioni opposte compatibili ma con banda gol stretta (poco spazio tra le due soglie).",
    ),
    CorrelationRule(
        "btts_totals_contradiction", ("BTTS", "TOTALS"),
        "BTTS Yes (almeno 2 gol totali) incompatibile con una soglia Under troppo bassa.",
    ),
    CorrelationRule(
        "btts_totals_implied", ("BTTS", "TOTALS"),
        "BTTS Yes implica Over 1.5 (entrambe le squadre a segno => almeno 2 gol totali).",
    ),
    CorrelationRule(
        "btts_totals_narrow_band", ("BTTS", "TOTALS"),
        "BTTS Yes e soglia O/U compatibili ma con banda gol stretta.",
    ),
    CorrelationRule(
        "match_result_double_chance_implied", ("MATCH_RESULT", "DOUBLE_CHANCE"),
        "L'esito 1X2 scelto e' incluso nella Double Chance scelta (stesso match).",
    ),
    CorrelationRule(
        "match_result_double_chance_exclusive", ("MATCH_RESULT", "DOUBLE_CHANCE"),
        "L'esito 1X2 scelto NON e' incluso nella Double Chance scelta: mutuamente esclusivi.",
    ),
)


# ---------------------------------------------------------------------------
# Output della classificazione.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationFinding:
    """Esito della classificazione di UNA coppia di pick (acceptance
    criteria "tracciabile": mai una severity senza un `rule`/`reason`
    espliciti che la giustifichino)."""

    fixture_id: Optional[int]
    market_a: str
    outcome_a: str
    market_b: str
    outcome_b: str
    severity: str
    rule: str
    reason: str


@dataclass
class CorrelationReport:
    """Tutte le coppie NON indipendenti trovate tra le pick della stessa
    fixture (le coppie indipendenti, la maggioranza, non sono elencate per
    non appesantire il report - restano implicitamente indipendenti)."""

    ruleset_version: str
    findings: list = field(default_factory=list)

    @property
    def excluded(self) -> list:
        return [f for f in self.findings if f.severity == EXCLUDE]

    @property
    def penalized(self) -> list:
        return [f for f in self.findings if f.severity == PENALTY]

    @property
    def is_valid(self) -> bool:
        """`False` se esiste almeno una coppia EXCLUDE (contraddizione
        logica): quella combinazione non puo' MAI verificarsi nella
        realta' e non deve mai essere proposta come schedina."""
        return len(self.excluded) == 0


# ---------------------------------------------------------------------------
# Classificazione per coppia di family.
# ---------------------------------------------------------------------------


def _totals_pair_finding(a: SemanticOutcome, b: SemanticOutcome, ruleset: CorrelationRuleSet) -> tuple[str, str, str]:
    """a, b entrambi TOTALS (soglie diverse, garantito dal chiamante)."""
    if a.tag == b.tag:
        lower, higher = sorted((a, b), key=lambda x: x.threshold)
        implying, implied = (higher, lower) if a.tag == "OVER" else (lower, higher)
        reason = (
            f"{implying.tag.title()} {implying.threshold} implica {implied.tag.title()} {implied.threshold} "
            "nella stessa partita"
        )
        return "totals_nested_same_direction", PENALTY, reason

    over_out, under_out = (a, b) if a.tag == "OVER" else (b, a)
    if under_out.threshold <= over_out.threshold:
        reason = (
            f"Over {over_out.threshold} e Under {under_out.threshold} sono incompatibili nella stessa partita "
            f"(Under {under_out.threshold} richiede meno gol di quanti Over {over_out.threshold} ne richieda)"
        )
        return "totals_contradiction", EXCLUDE, reason

    gap = under_out.threshold - over_out.threshold
    if gap <= ruleset.totals_narrow_band_max_gap:
        reason = f"Over {over_out.threshold} e Under {under_out.threshold}: banda gol stretta (scarto {gap})"
        return "totals_narrow_band", PENALTY, reason

    return "totals_wide_band", INDEPENDENT, "Soglie compatibili con banda gol ampia, correlazione trascurabile"


def _btts_totals_finding(btts: SemanticOutcome, totals: SemanticOutcome, ruleset: CorrelationRuleSet) -> tuple[str, str, str]:
    """btts e' family BTTS, totals e' family TOTALS."""
    if btts.tag == "NO":
        # BTTS No (almeno una squadra non segna) non vincola in modo
        # monotono il totale gol (compatibile sia con 0-0 sia con 5-0):
        # nessuna implicazione ne' contraddizione dimostrabile.
        return "btts_no_totals_independent", INDEPENDENT, "BTTS No non vincola logicamente le soglie gol totali"

    threshold = totals.threshold
    if totals.tag == "UNDER":
        if threshold <= 1.5:
            reason = f"BTTS Yes richiede almeno 2 gol totali, incompatibile con Under {threshold}"
            return "btts_totals_contradiction", EXCLUDE, reason
        gap = threshold - 1.5
        if gap <= ruleset.totals_narrow_band_max_gap:
            reason = f"BTTS Yes e Under {threshold}: compatibili ma banda gol stretta (scarto {gap})"
            return "btts_totals_narrow_band", PENALTY, reason
        return "btts_totals_wide_band", INDEPENDENT, "Banda gol ampia, correlazione trascurabile"

    # totals.tag == "OVER"
    if threshold <= 1.5:
        reason = f"BTTS Yes implica Over {threshold} (entrambe le squadre a segno => almeno 2 gol totali)"
        return "btts_totals_implied", PENALTY, reason
    gap = threshold - 1.5
    if gap <= ruleset.totals_narrow_band_max_gap:
        reason = f"BTTS Yes e Over {threshold}: compatibili ma banda gol stretta (scarto {gap})"
        return "btts_totals_narrow_band", PENALTY, reason
    return "btts_totals_wide_band", INDEPENDENT, "Banda gol ampia, correlazione trascurabile"


def _match_result_double_chance_finding(mr: SemanticOutcome, dc: SemanticOutcome) -> tuple[str, str, str]:
    """mr e' family MATCH_RESULT, dc e' family DOUBLE_CHANCE."""
    members = DOUBLE_CHANCE_MEMBERS.get(dc.tag, frozenset())
    dc_label = dc.tag.replace("_", "/").title()
    if mr.tag in members:
        reason = f"{mr.tag.title()} e' incluso nella Double Chance {dc_label}: non sono indipendenti"
        return "match_result_double_chance_implied", PENALTY, reason
    reason = f"{mr.tag.title()} esclude la Double Chance {dc_label} nella stessa partita"
    return "match_result_double_chance_exclusive", EXCLUDE, reason


def classify_pick_pair(
    pick_a: CandidatePick,
    pick_b: CandidatePick,
    ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
) -> CorrelationFinding:
    """Classifica UNA coppia di pick. Regola "same-match" (acceptance
    criteria): fixture diverse => SEMPRE indipendenti, nessuna regola di
    correlazione incrociata tra partite diverse."""
    if pick_a.fixture_id != pick_b.fixture_id:
        return CorrelationFinding(
            fixture_id=None,
            market_a=pick_a.market, outcome_a=pick_a.outcome,
            market_b=pick_b.market, outcome_b=pick_b.outcome,
            severity=INDEPENDENT, rule="different_fixture",
            reason="Fixture diverse: eventi statisticamente indipendenti, nessuna regola same-match applicabile",
        )

    fixture_id = pick_a.fixture_id

    def _finding(rule: str, severity: str, reason: str) -> CorrelationFinding:
        return CorrelationFinding(
            fixture_id=fixture_id,
            market_a=pick_a.market, outcome_a=pick_a.outcome,
            market_b=pick_b.market, outcome_b=pick_b.outcome,
            severity=severity, rule=rule, reason=reason,
        )

    market_a_norm = _normalize(pick_a.market)
    market_b_norm = _normalize(pick_b.market)

    if market_a_norm == market_b_norm:
        if _normalize(pick_a.outcome) == _normalize(pick_b.outcome):
            return _finding("duplicate_pick", EXCLUDE, "Stesso mercato ed esito selezionati due volte per la stessa partita")
        return _finding(
            "same_market_mutually_exclusive", EXCLUDE,
            f"Mercato '{pick_a.market}': un solo esito puo' verificarsi nella stessa partita",
        )

    semantic_a = classify_outcome(pick_a.market, pick_a.outcome)
    semantic_b = classify_outcome(pick_b.market, pick_b.outcome)
    if semantic_a is None or semantic_b is None:
        return _finding("no_rule", INDEPENDENT, "Nessuna regola di correlazione nota per questa coppia di mercati/esiti")

    families = {semantic_a.family, semantic_b.family}

    if families == {"TOTALS"}:
        rule, severity, reason = _totals_pair_finding(semantic_a, semantic_b, ruleset)
        return _finding(rule, severity, reason)

    if families == {"BTTS", "TOTALS"}:
        btts, totals = (semantic_a, semantic_b) if semantic_a.family == "BTTS" else (semantic_b, semantic_a)
        rule, severity, reason = _btts_totals_finding(btts, totals, ruleset)
        return _finding(rule, severity, reason)

    if families == {"MATCH_RESULT", "DOUBLE_CHANCE"}:
        mr, dc = (semantic_a, semantic_b) if semantic_a.family == "MATCH_RESULT" else (semantic_b, semantic_a)
        rule, severity, reason = _match_result_double_chance_finding(mr, dc)
        return _finding(rule, severity, reason)

    return _finding("no_rule", INDEPENDENT, "Nessuna regola di correlazione nota per questa coppia di mercati/esiti")


def evaluate_same_match_correlations(
    picks: list[CandidatePick],
    ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
) -> CorrelationReport:
    """Valuta TUTTE le coppie della STESSA fixture tra `picks` (regole
    same-match): le coppie su fixture diverse non vengono nemmeno
    esaminate (indipendenti per costruzione, mai un confronto inutile)."""
    by_fixture: dict[int, list[CandidatePick]] = {}
    for p in picks:
        by_fixture.setdefault(p.fixture_id, []).append(p)

    findings: list[CorrelationFinding] = []
    for fixture_picks in by_fixture.values():
        if len(fixture_picks) < 2:
            continue
        for pick_a, pick_b in combinations(fixture_picks, 2):
            finding = classify_pick_pair(pick_a, pick_b, ruleset=ruleset)
            if finding.severity != INDEPENDENT:
                findings.append(finding)

    findings.sort(key=lambda f: (f.fixture_id, f.market_a, f.market_b))
    return CorrelationReport(ruleset_version=ruleset.version, findings=findings)


# ---------------------------------------------------------------------------
# Probabilita' congiunta: "ingenua" (prodotto, corretta SOLO se davvero
# indipendenti) vs corretta per la correlazione same-match.
# ---------------------------------------------------------------------------


def naive_independent_probability(picks: list[CandidatePick]) -> Optional[float]:
    """Prodotto semplice dei `p_model` (l'assunzione "ingenua" di
    indipendenza che questo motore esiste per NON lasciar passare
    inosservata quando le pick sono fortemente correlate). `None` se la
    lista e' vuota o manca almeno un `p_model`."""
    probabilities = [p.p_model for p in picks]
    if not probabilities or any(p is None for p in probabilities):
        return None
    result = 1.0
    for p in probabilities:
        result *= float(p)
    return result


@dataclass
class CombinationEvaluation:
    """Confronto esplicito tra probabilita' 'ingenua' (prodotto, corretta
    SOLO se le pick fossero davvero indipendenti) e probabilita' corretta
    per la correlazione same-match (acceptance criteria "Combinazioni
    fortemente correlate non passano come indipendenti"): quando esiste
    almeno una coppia PENALTY/EXCLUDE, `adjusted_probability` e' calcolata
    con una regola diversa dal semplice prodotto, mai la stessa formula
    "ingenua" spacciata per corretta."""

    is_valid: bool
    ruleset_version: str
    naive_probability: Optional[float]
    adjusted_probability: Optional[float]
    findings: list = field(default_factory=list)


def _adjusted_probability_by_clusters(picks: list[CandidatePick], penalty_findings: list[CorrelationFinding]) -> float:
    """Raggruppa le pick in componenti connesse (collegate da almeno una
    coppia PENALTY): dentro ogni componente la probabilita' congiunta vera
    e' SEMPRE <= al minimo dei `p_model` del gruppo (disuguaglianza valida
    per QUALSIASI coppia di eventi, mai un numero inventato) - ed E' QUELLA
    ESATTA nei casi di implicazione logica stretta (nested totals, BTTS
    Yes/Over 1.5, esito 1X2/Double Chance) che sono la maggioranza delle
    regole PENALTY di questo motore. Tra componenti diverse (nessuna
    coppia PENALTY che le colleghi) si continua a moltiplicare, come per
    pick davvero indipendenti."""
    n = len(picks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    index_by_key: dict[tuple, list[int]] = {}
    for idx, p in enumerate(picks):
        index_by_key.setdefault((p.fixture_id, p.market, p.outcome), []).append(idx)

    for finding in penalty_findings:
        keys_a = index_by_key.get((finding.fixture_id, finding.market_a, finding.outcome_a), [])
        keys_b = index_by_key.get((finding.fixture_id, finding.market_b, finding.outcome_b), [])
        for ia in keys_a:
            for ib in keys_b:
                if ia != ib:
                    union(ia, ib)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    adjusted = 1.0
    for members in clusters.values():
        cluster_probs = [float(picks[i].p_model) for i in members]
        adjusted *= min(cluster_probs)
    return adjusted


def evaluate_combination(
    picks: list[CandidatePick],
    ruleset: CorrelationRuleSet = DEFAULT_CORRELATION_RULESET,
) -> CombinationEvaluation:
    """Valuta un insieme di pick candidate a formare UNA schedina.

    - Se esiste una coppia EXCLUDE (contraddizione logica: le due pick non
      possono MAI verificarsi insieme nella stessa partita), la
      combinazione e' INVALIDA (`is_valid=False`, `adjusted_probability`
      forzata a 0.0 indipendentemente da quanto ottimistico sarebbe stato
      il calcolo ingenuo: una schedina cosi' non deve mai essere generata).
    - Altrimenti `adjusted_probability` e' calcolata raggruppando le pick
      correlate (PENALTY) in cluster e usando il minimo dei `p_model` per
      cluster (vedi `_adjusted_probability_by_clusters`), MAI il prodotto
      semplice per le coppie flaggate.
    """
    findings = evaluate_same_match_correlations(picks, ruleset=ruleset).findings
    exclude_findings = [f for f in findings if f.severity == EXCLUDE]
    penalty_findings = [f for f in findings if f.severity == PENALTY]

    naive = naive_independent_probability(picks)

    if exclude_findings:
        return CombinationEvaluation(
            is_valid=False,
            ruleset_version=ruleset.version,
            naive_probability=naive,
            adjusted_probability=0.0,
            findings=findings,
        )

    if naive is None:
        return CombinationEvaluation(
            is_valid=True,
            ruleset_version=ruleset.version,
            naive_probability=None,
            adjusted_probability=None,
            findings=findings,
        )

    adjusted = _adjusted_probability_by_clusters(picks, penalty_findings)
    return CombinationEvaluation(
        is_valid=True,
        ruleset_version=ruleset.version,
        naive_probability=naive,
        adjusted_probability=adjusted,
        findings=findings,
    )


def candidates_from_pool_picks(pool_picks: list[PoolPick]) -> list[CandidatePick]:
    """Comodo: estrae i `CandidatePick` da una lista di `PoolPick` (output
    di `build_pick_pool`/SLIP-01), cosi' il chiamante (tipicamente SLIP-03)
    puo' passare direttamente `PickPoolResult.picks` a questo motore."""
    return [p.candidate for p in pool_picks]

