"""Segnale a soglia OTTIMALE per Corners/Cards a linea configurabile
(MARKET-05/06, 2026-09-13) - stesso principio di `over_signal_policy.py`
per i gol, necessario perche' a soglia 0.5 diverse linee sono DEGENERI
(eventi rari: es. cards 6.5 non prediceva mai "Over" a soglia 0.5, corners
11.5 quasi mai - verificato sul training reale del 2026-09-12).

Distinto DELIBERATAMENTE dal pick/probabilita' esposti in `predictions`
(soglia fissa 0.5, MAI toccata da questo modulo, stesso principio di
`over_signal_policy.py`): qui la soglia e' quella di Youden (statistica
J = TPR - FPR massimizzata, calcolata da `classification_report.py`
durante il training reale), diversa dalla soglia orientata alla PRECISIONE
usata da `over_signal_policy.py` per i gol - scelta consistente con come
questo progetto valuta gia' Corners/Cards ("tutte le metriche possibili",
soglia ottimale sempre riportata nel report di classificazione).

Il segnale e' quindi INDIPENDENTE dal pick a 0.5: puo' attivarsi (P(Over)
>= soglia_linea) anche quando il pick mostrato resta "Under", e viceversa.

Valori corners (`corners_line_8_5..11_5`) dal training del 2026-09-12
(`best_models/corners_cards_from_export_summary.json`,
`classification_report_post.optimal_threshold`) - PRECEDONO il fix della
contaminazione bookmaker (BetMGM/BetRivers/Bovada, 2026-09-16) e la
riverifica che ha concluso di NON promuovere alcun modello corners NUOVO
(segnale troppo debole, AUC 0.53-0.57 anche sui dati puliti - vedi
`docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md`).

CORREZIONE 2026-09-19: contrariamente a quanto scritto qui fino a questa
data ("nessun modello corners mai promosso"), verificato sul registry
reale (`report_verifica_corners_production_esistente.md`) che questi 4
modelli SONO EFFETTIVAMENTE in `production` dal 2026-09-12 (stesso giorno/
batch dei vecchi modelli cards, stesso `actor: operator_request`, 73
feature incluse quelle quote legacy, AUC 0.51-0.54 - un segnale
debolissimo che il gate ha comunque lasciato passare, essendo la prima
promozione per quel mercato). L'affermazione precedente era una deduzione
mai verificata direttamente, non un dato controllato - `evaluate_line_
market_signal` quindi NON e' inerte su questi 4 mercati: e' attivo, con
soglie Youden calcolate su un modello quasi-casuale e probabilmente
contaminato dagli stessi bookmaker placeholder mai esclusi qui (creato
prima del fix del 16/09). Decisione su come procedere (lasciare, archiviare
la production corners, o altro) lasciata all'operatore - non presa
unilateralmente in questa correzione.

Valori cards (`cards_line_3_5..6_5`) RISCRITTI il 2026-09-19 dopo il
training reale + verifica ROI con IC bootstrap sui champion effettivi
(`report_cards_champion_verifica.md`): a differenza di corners, qui la
soglia scelta e' quella con ROI positivo e intervallo di confidenza
bootstrap (2000 resample, 95%) che NON include zero, sulla direzione
UNDER (l'unica con edge verificato - la direzione OVER non ha mai
superato il criterio su nessuna linea, vedi `report_cards_training_step_
a_b.md`). Tra le soglie robuste disponibili per ciascuna linea si e'
scelta quella col volume maggiore (piu' partite coperte), non la piu'
aggressiva - vedi il report per le alternative.

Una ritaratura futura (nuovo training) richiede una nuova versione qui,
MAI un edit silenzioso di questi valori (stesso principio di
`DecisionPolicy`/BET-04 e di `over_signal_policy.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

LINE_MARKET_SIGNAL_POLICY_VERSION = "line_market_signal_policy_v2"


@dataclass(frozen=True)
class LineMarketSignalThreshold:
    market: str
    probability_threshold: float
    direction: Literal["over", "under"]
    expected_accuracy: float
    expected_precision: float
    expected_recall: float
    n_oof: int


# fmt: off
LINE_MARKET_SIGNAL_THRESHOLDS: dict[str, LineMarketSignalThreshold] = {
    # Corners: soglie Youden del 12/09, ATTIVE - un modello e' davvero in
    # production per queste 4 linee dal 12/09 (verificato 19/09, vedi il
    # commento in testa al file), AUC 0.51-0.54, probabilmente ancora
    # contaminato dai bookmaker placeholder (fix del 16/09 mai riapplicato
    # a questo run). Non correggerle qui in silenzio: serve una decisione
    # esplicita (nuovo training pulito + repromozione, o disattivazione).
    "corners_line_8_5": LineMarketSignalThreshold("corners_line_8_5", 0.6140, "over", 0.5277, 0.6185, 0.5538, n_oof=8804),
    "corners_line_9_5": LineMarketSignalThreshold("corners_line_9_5", 0.4906, "over", 0.5298, 0.5182, 0.4923, n_oof=8804),
    "corners_line_10_5": LineMarketSignalThreshold("corners_line_10_5", 0.3438, "over", 0.4600, 0.3894, 0.7947, n_oof=8804),
    "corners_line_11_5": LineMarketSignalThreshold("corners_line_11_5", 0.2786, "over", 0.5209, 0.2774, 0.5034, n_oof=8804),
    # Cards: direzione UNDER (p_over <= soglia), soglia robusta col volume
    # maggiore tra quelle con IC95% ROI bootstrap che non include zero.
    "cards_line_3_5": LineMarketSignalThreshold("cards_line_3_5", 0.40, "under", 0.5930, 0.704, 0.62, n_oof=2706),
    "cards_line_4_5": LineMarketSignalThreshold("cards_line_4_5", 0.45, "under", 0.6300, 0.706, 0.90, n_oof=3141),
    "cards_line_5_5": LineMarketSignalThreshold("cards_line_5_5", 0.45, "under", 0.7560, 0.755, 0.99, n_oof=2843),
    "cards_line_6_5": LineMarketSignalThreshold("cards_line_6_5", 0.25, "under", 0.8530, 0.862, 0.93, n_oof=2106),
}
# fmt: on


def evaluate_line_market_signal(market: str, p_over: Optional[float]) -> Optional[dict]:
    """`None` se il mercato non ha una soglia ottima calcolata, o se
    `p_over` non e' disponibile (mai un segnale inventato). Altrimenti un
    dict con `signal` (bool) + i valori di riferimento della soglia, cosi'
    il chiamante puo' sempre mostrare "a quale accuracy/precision/recall
    attesi corrisponde" invece di un booleano nudo.

    `direction` decide il verso del confronto: "over" (storico, corners)
    segnala quando `p_over >= soglia`; "under" (cards, 2026-09-19) segnala
    quando `p_over <= soglia` - l'edge verificato su cards e' scommettere
    Under quando il modello e' CONFIDENTE che il totale resti basso, non
    quando prevede Over."""
    spec = LINE_MARKET_SIGNAL_THRESHOLDS.get(market)
    if spec is None or p_over is None:
        return None
    p = float(p_over)
    signal = p >= spec.probability_threshold if spec.direction == "over" else p <= spec.probability_threshold
    return {
        "signal": bool(signal),
        "direction": spec.direction,
        "threshold": spec.probability_threshold,
        "expected_accuracy": spec.expected_accuracy,
        "expected_precision": spec.expected_precision,
        "expected_recall": spec.expected_recall,
        "policy_version": LINE_MARKET_SIGNAL_POLICY_VERSION,
    }
