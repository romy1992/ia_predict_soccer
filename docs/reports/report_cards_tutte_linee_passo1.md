# Mercato cards (cartellini) — Passo 1, tutte le linee

Stesso identico giro fatto su corners (`report_corners_8_5_passo1.md`,
`docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md`). Dato letto dal
DB reale, sola lettura: nessuna quota modificata, nessuno script di
training/pipeline toccato. Il fix per `LINE_MARKETS`/`_build_line_dataset`
(gia' presente sul branch, usato per corners) copre gia' anche
`cards_line_*`: **Passo 1 NON bloccato**, nessuna modifica necessaria.

## Passo 1 — estrazione grezza (`fill_missing=False`)

| Linea | Shape | NaN totali | y mean (P(over)) |
|---|---:|---:|---:|
| cards_line_3_5 | (8423, 453) | 1.967.239 | 0.6105 |
| cards_line_4_5 | (8423, 453) | 1.967.239 | 0.4351 |
| cards_line_5_5 | (8423, 453) | 1.967.239 | 0.2780 |
| cards_line_6_5 | (8423, 453) | 1.967.239 | 0.1663 |

CSV prodotti in `scripts/analysis/_export/cards_line_{3_5,4_5,5_5,6_5}_raw.csv`.

## Passo 2 — controllo bookmaker (quote placeholder)

Query diretta `select(Match).options(selectinload(Match.odds))`, filtro sul
bucket JSON `Odds.cards` isolando le chiavi con linea == 3.5 (linea piu'
liquida, stesso criterio usato per corners_8.5).

| Metrica | Valore |
|---|---:|
| Match totali a DB | 47.719 |
| Fixture distinte con bucket `cards` non vuoto | 10.257 |
| Quote totali con linea == 3.5 (grezze) | 32.554 |
| Bookmaker che quotano Over/Under 3.5 cards | 8 |

### Quota Over 3.5 per bookmaker (fixture distinte, valore piu' comune)

| Bookmaker | N fixture | N valori distinti | Valore piu' comune | % su valore comune |
|---|---:|---:|---:|---:|
| **Bovada** | **6.411** | **1** | **1.61** | **100.0%** |
| Betano | 2.462 | 87 | 1.65 | 6.0% |
| Unibet | 2.398 | 128 | 1.50 | 3.1% |
| Superbet | 1.738 | 111 | 1.35 | 3.9% |
| Bet365 | 1.184 | 22 | 1.62 | 17.3% |
| 1xBet | 971 | 34 | 1.67 | 16.9% |
| Pinnacle | 682 | 145 | 1.83 | 2.3% |
| 10Bet | 431 | 24 | 1.62 | 10.4% |

**PATTERN TROVATO: Bovada.** Quota Over 3.5 = 1.61 e Under 3.5 = 2.23,
IDENTICHE su tutte le 6.411 fixture distinte (100%) — stesso pattern
placeholder gia' trovato sui corner (dove Bovada era uno dei tre bookmaker
contaminati). Copertura: 6.411/10.257 = 62.5% delle fixture quotate su
questa linea, stessa entita' del caso corners (~60%).

**BetMGM e BetRivers (contaminati sui corner) qui NON quotano affatto il
mercato cards** — non compaiono tra i bookmaker con chiavi cards a DB,
quindi non serve escluderli qui (non c'e' nulla da escludere: la loro
assenza e' gia' l'esclusione).

Gli altri 7 bookmaker mostrano quote variabili (decine/centinaia di valori
distinti su centinaia/migliaia di fixture): nessun altro pattern sospetto.

## Conclusione operativa

- Passo 1: OK per tutte e 4 le linee, nessun blocco, nessuna modifica di
  codice necessaria.
- Passo 2: **Bovada va escluso dalle feature per linea cards**, stesso
  trattamento gia' applicato a BetMGM/BetRivers/Bovada sui corners (vedi
  `corners_market.py`, gia' presente sul branch — va replicato in
  `cards_market.py` se non gia' fatto, ma questo esula da questo
  incarico di sola analisi).

## Dopo esclusione Bovada

Fix applicato in `cards_market.py` (`CONTAMINATED_BOOKMAKERS = {"Bovada"}`,
stesso pattern dei corner) e ri-estrazione identica al Passo 1.

| Linea | Shape | NaN totali | y mean (P(over)) |
|---|---:|---:|---:|
| cards_line_3_5 | (8423, 453) | 2.275.039 | 0.6105 |
| cards_line_4_5 | (8423, 453) | 2.275.039 | 0.4351 |
| cards_line_5_5 | (8423, 453) | 2.275.039 | 0.2780 |
| cards_line_6_5 | (8423, 453) | 2.275.039 | 0.1663 |

Shape e y mean identici a prima (stesse 8.423 fixture, nessun dato DB
cambiato nel frattempo — confermato anche dalla colonna aggregata
`odds_count`, generale e non filtrata per linea, invariata a mean=14.858
prima/dopo). L'unico effetto e' sulle feature per linea, dove Bovada viene
ora escluso dal conteggio/media.

### Confronto `odds_count_line_<linea>` prima/dopo

| Linea | Mean prima | Mean dopo | Median prima | Median dopo | Righe non-NaN prima | Righe non-NaN dopo |
|---|---:|---:|---:|---:|---:|---:|
| 3_5 | 3.485 | 6.333 | 2.0 | 6.0 | 7.804 | 2.674 |
| 4_5 | 3.639 | 6.342 | 2.0 | 6.0 | 8.240 | 3.110 |
| 5_5 | 4.447 | 4.447 | 4.0 | 4.0 | 2.812 | 2.812 |
| 6_5 | 3.798 | 3.798 | 4.0 | 4.0 | 2.077 | 2.077 |

**La media SALE, non scende** — apparentemente controintuitivo, ma corretto:
non e' un valore che cresce riga per riga, e' un effetto di composizione.
Verificato riga per riga (merge su `id_fixture`): nessuna riga ha un
`odds_count_line_3_5`/`_4_5` più alto dopo il fix; per ogni fixture il
conteggio resta identico o diventa `NaN`. Sulla linea 3.5, 5.130 fixture
(5.573 → 443 al bucket `odds_count=2`) avevano **Bovada come UNICO
quotante**: escluso Bovada, quelle righe passano da "2 quote fittizie" a
`NaN` (nessuna quota reale disponibile) e escono dal calcolo della media
(`.mean()` ignora i NaN). Le righe con conteggio ≥4 (altri bookmaker
presenti) restano **esattamente invariate** — segno che Bovada, quando
compare, non coesiste quasi mai con altri bookmaker sulla stessa
fixture/linea. Il risultato netto: la media sale perché il denominatore si
restringe alle sole righe con quote reali, ma la **copertura** (righe
non-NaN) crolla — da 7.804 a 2.674 su 3.5 (-66%), da 8.240 a 3.110 su 4.5
(-62%). Le linee 5.5 e 6.5 sono **identiche in ogni cifra**: Bovada non
quota quelle linee (coerente con l'indagine originale, concentrata sulla
linea 3.5 piu' liquida), quindi l'esclusione non le tocca.
