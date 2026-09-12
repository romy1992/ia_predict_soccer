> **STATO: IMPLEMENTATO (2026-09-12)**. Vedi `IMPLEMENTATION_LOG.md`, voce
> "Model Diagnostics: report out-of-fold per mercato nel frontend web", per
> il dettaglio completo di backend/API/frontend consegnati. Decisioni prese
> sui due punti lasciati aperti qui sotto: (1) menu — nuova voce top-level
> "Model Diagnostics" (confermata con l'operatore); (2) identita' visiva —
> mantenuta quella distinta dell'Artifact (teal/amber), scoped sotto
> `.model-diagnostics-page` per non toccare la palette generale dell'app.
> Questo file resta come riferimento storico del design/dei requisiti
> originali, non piu' come lavoro da fare.

# Prompt: sezione "Model Diagnostics" nel frontend web

Richiesto esplicitamente dall'operatore (prima il 2026-09-08 come nota per il
futuro, poi confermato il 2026-09-10): portare nel web app la stessa vista
di diagnostica modelli oggi pubblicata SOLO come Artifact one-shot
("Under/Over Champions"), come sezione permanente e viva (dati ricalcolati,
non uno snapshot statico), navigabile dalla UI.

## Riferimento visivo (fonte di verità per il design)

Artifact gia' pubblicato (privato, di proprieta' dell'operatore):
**https://claude.ai/code/artifact/46107f9a-0aac-4d07-bebb-20c4e3c6d71f**

E' la versione interattiva/scrollabile di quello che si vede negli screenshot
condivisi in chat il 2026-09-10 (grafico ROC comparativo a 4 linee + griglia
di card per soglia con matrice di confusione e tabella precision/recall/F1).
Aprire quel link per vedere il risultato reale prima di implementare - qui
sotto sono comunque riportati per intero i design token e la struttura HTML/
CSS/JS sorgente dell'artifact, cosi' da poter replicare ESATTAMENTE lo stesso
look (palette, tipografia, layout) nel componente React reale, senza dover
indovinare dai pixel di uno screenshot.

## Obiettivo

Una pagina (o una nuova tab dentro "ML Lab") che mostri, per ciascun mercato
con un modello registrato (oggi 5: i 4 Under/Over + goal_no_goal - MAI solo i
4 Under/Over come nell'artifact originale, quello era limitato dal task del
momento):

1. **Discrimination overview**: un grafico ROC comparativo con una linea per
   mercato (stesso stile: `<svg>` con `<polyline>`, non una libreria di
   chart esterna - vedi sorgente sotto) + legenda con AUC per mercato.
2. **Per-market breakdown**: una card per mercato con:
   - header (nome mercato leggibile, `n` righe OOF, pipeline del champion,
     badge stage registry - `candidate`/`production`)
   - 3 stat tile: Accuracy, ROC AUC, Weighted F1
   - matrice di confusione 2x2 (predicted vs actual), celle shadate per
     intensita' in base al conteggio, percentuale "of actual" per cella
   - tabella precision/recall/F1 per classe + riga "Weighted"
3. Un callout in cima che spiega la metodologia (walk-forward OOF,
   nessuna riga mai scorata da un modello allenato su di essa, soglia di
   decisione fissa p>=0.5) - stesso tono/contenuto del callout nell'artifact.

## Dati: nuovo endpoint necessario, MAI dati statici in JS

L'artifact ha i numeri hardcoded in un oggetto `DATA` JS (era un report
one-shot). La pagina web reale deve invece leggerli da un endpoint nuovo,
che ricalcoli (o serva una cache aggiornata al bisogno) esattamente questa
forma dati, riusando la logica GIA' ESISTENTE in
`scripts/analysis/evaluate_champions_detailed.py` (walk-forward OOF, clone
del pipeline registrato, stessa garanzia anti-leakage di
`temporal_oof_probabilities` - NON reinventare quel calcolo, solo esporlo
via API invece che come script CLI).

Forma dati per mercato (uguale alla shape di `DATA[market]` nell'artifact,
vedi sorgente completo sotto per i nomi esatti dei campi):
```
{
  "n_oof": int,
  "champion": str,           // nome pipeline/classe sklearn del champion
  "accuracy": float,
  "auc": float,
  "cm": {"tn": int, "fp": int, "fn": int, "tp": int},
  "class0": {"precision": float, "recall": float, "f1": float, "support": int},
  "class1": {"precision": float, "recall": float, "f1": float, "support": int},
  "weighted": {"precision": float, "recall": float, "f1": float},
  "roc_fpr": [float, ...],   // punti curva ROC, ordinati
  "roc_tpr": [float, ...]
}
```

Proposta endpoint: `GET /models/diagnostics?markets=under_over_1_5,...`
(o senza query param = tutti i mercati con un run registrato), schema
Pydantic dedicato in `src/api/schemas.py`, servizio nuovo (o funzione
riusata da `evaluate_champions_detailed.py`, refactorizzata per essere
chiamabile sia da CLI sia da endpoint - stesso principio "una sola
funzione, riusata da manuale e API" gia' seguito ovunque nel progetto).

**Nota su `class0`/`class1`**: per i mercati Under/Over la label e' overo
"Under"/"Over" (vedi `LABELS`/CHAMPIONS nel JS sorgente); per `goal_no_goal`
sara' "No Goal"/"Goal" - il componente React deve leggere l'etichetta di
classe dal payload (o da una mappa mercato->etichette lato frontend), MAI
hardcoded a "Under"/"Over" come nell'artifact originale (che copriva solo
4 mercati Under/Over).

## Design tokens (da riusare identici)

```css
:root{
  --paper:#F1F5F3; --surface:#FFFFFF; --surface-2:#E7EEEB;
  --ink:#0F211E; --ink-soft:#3C534E; --muted:#647A75;
  --line:#D6E1DC; --line-soft:#E4ECE8;
  --under:#1C7A72; --under-soft:#E4F2EF;
  --over:#C96A26; --over-soft:#FBEADB;
  --scale-1:#C9DCEE; --scale-2:#8DB4DC; --scale-3:#4F7FBE; --scale-4:#1F4C8F;
  --grid:#122E29; --focus:#1C7A72;
}
/* dark mode (prefers-color-scheme + [data-theme="dark"]) */
:root:not([data-theme="light"]) /* dark media query */, :root[data-theme="dark"]{
  --paper:#0C1917; --surface:#12211F; --surface-2:#182E2B;
  --ink:#E8F2EF; --ink-soft:#B9CBC6; --muted:#7E9C96;
  --line:#243F3A; --line-soft:#1C332E;
  --under:#4FC7BB; --under-soft:#12312D;
  --over:#F0A15A; --over-soft:#2E2013;
  --scale-1:#3E6E9E; --scale-2:#5C93C6; --scale-3:#82B3DE; --scale-4:#ADD3F2;
  --grid:#DCEDE9;
}
```
Tipografia: `Barlow Semi Condensed` (700, display/titoli) + `IBM Plex Sans`
(400/500/600, corpo testo) + `IBM Plex Mono` (400/500, numeri/label tecniche
- font-variant-numeric tabular-nums per gli stat). Da Google Fonts, stesso
link gia' usato: `https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap`.

Il tema scuro dell'artifact usa teal (`--under`)/amber (`--over`) invece del
blu/grigio gia' in uso nel resto della Dashboard - va deciso se questa
pagina mantiene una sua identita' visiva distinta (come nell'artifact,
coerente con "design system dedicato" gia' menzionato in
IMPLEMENTATION_LOG.md 2026-09-08) o se si allinea alla palette generale
dell'app (`frontend/src/styles.css`) - lasciare la scelta a chi implementa,
non e' stato deciso esplicitamente dall'operatore.

## Sorgente HTML/CSS/JS completo dell'artifact di riferimento

Vedi l'artifact stesso (link sopra) per il sorgente completo aggiornato -
include la costruzione del grafico ROC via SVG puro (nessuna libreria di
chart) e la costruzione delle card via template string JS, entrambe da
riscrivere come componenti React (`ModelDiagnosticsPage.jsx`, con
sottocomponenti tipo `RocComparisonChart`/`ThresholdCard`) che leggono da
`DATA` recuperato via fetch invece che hardcoded.

## Non-obiettivi

- Nessuna modifica al training/registry/serving - solo una nuova vista
  di sola lettura.
- Nessun refresh automatico richiesto esplicitamente per ora (va bene
  ricalcolare al caricamento della pagina, o con un bottone "Aggiorna" -
  il walk-forward OOF su tutte le righe storiche non e' istantaneo, va
  verificato il tempo di risposta reale prima di decidere se serve una
  cache/job dedicato).
- Nessuna decisione ancora presa su dove esporre il link nel menu (nuova
  voce top-level "Diagnostics" vs nuova tab dentro "ML Lab" gia'
  esistente) - proporre un'opzione e chiedere conferma prima di
  implementare, stesso principio gia' seguito per le altre feature di
  questa sessione.
