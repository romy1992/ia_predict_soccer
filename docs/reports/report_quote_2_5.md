# Indagine contaminazione quote — mercato `under_over_2_5`

Dati letti dal DB reale (Postgres su Railway, stesso `DATABASE_URL` usato dall'app), query SQLAlchemy diretta su `Match` con `selectinload(Match.odds)`, filtrando le fixture con almeno una riga `Odds.under_over_2_5` non nulla. Nessun numero in questo report e' stimato: tutto proviene dall'esecuzione riportata sotto.

## Portata dei dati

- Fixture con almeno una riga `odds.under_over_2_5` non nulla (colonna JSON): **20.759**
- Di queste, righe `Odds` con il dizionario `under_over_2_5` effettivamente non vuoto: **19.991** (le restanti hanno la colonna valorizzata a `{}`, non `NULL` — passano il filtro SQL ma non contengono quote)
- Quote singole estratte (una per ogni coppia esito_bookmaker dentro il JSON): **255.578**
- Sorgenti (`odds_from`) su queste quote: `sports-api` = 109.534, `odds-api` = 146.044 (nessun'altra sorgente presente)

Nota sul parsing: le chiavi del JSON (prodotte da `map_odds()`, formato `f"{esito}_{bookmaker}"`) si presentano in **due varianti** a seconda della sorgente/periodo di ingestion: `"over 2.5_Bet365"` (spazio, tipico di `sports-api`) e `"over_2.5_Unibet"` (underscore-punto, tipico di `odds-api`), oltre alla variante `"alternate over 2.5_..."` per alcuni bookmaker USA/odds-api. Nella tabella sotto le ho tenute distinte com'erano a DB (colonna "Esito (chiave grezza)"), senza normalizzarle, proprio per non mascherare eventuali differenze sistematiche tra formati/sorgenti.

## Tabella per bookmaker x esito (tutte le fixture disponibili)

| Bookmaker | Esito (chiave grezza) | N quote | Min | Mediana | Max | P1 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| 10Bet | over 2.5 | 4658 | 1.10 | 1.85 | 3.85 | 1.28 | 2.72 |
| 10Bet | under 2.5 | 4658 | 1.25 | 1.90 | 6.50 | 1.42 | 3.65 |
| 188Bet | over 2.5 | 1304 | 1.61 | 1.94 | 2.17 | 1.76 | 2.10 |
| 188Bet | under 2.5 | 1304 | 1.61 | 1.93 | 2.18 | 1.74 | 2.09 |
| 1xBet | alternate_over_2_5 | 318 | 1.21 | 1.65 | 2.45 | 1.26 | 2.35 |
| 1xBet | alternate_under_2_5 | 318 | 1.58 | 2.30 | 4.71 | 1.62 | 4.00 |
| 1xBet | over 2.5 | 5665 | 1.07 | 1.89 | 5.60 | 1.22 | 2.67 |
| 1xBet | over_2.5 | 12735 | 1.46 | 1.96 | 2.77 | 1.59 | 2.56 |
| 1xBet | under 2.5 | 5665 | 1.15 | 1.98 | 5.90 | 1.41 | 3.73 |
| 1xBet | under_2.5 | 12735 | 1.45 | 1.96 | 2.65 | 1.57 | 2.51 |
| 888Sport | over 2.5 | 3268 | 1.10 | 1.80 | 3.60 | 1.22 | 2.72 |
| 888Sport | under 2.5 | 3268 | 1.25 | 1.95 | 7.00 | 1.40 | 3.80 |
| Bet365 | over 2.5 | 5591 | 1.11 | 1.85 | 4.33 | 1.25 | 2.75 |
| Bet365 | under 2.5 | 5591 | 1.22 | 1.95 | 6.50 | 1.44 | 4.00 |
| Betano | over 2.5 | 5164 | 1.12 | 1.87 | 5.00 | 1.28 | 2.75 |
| Betano | under 2.5 | 5164 | 1.16 | 1.95 | 6.30 | 1.45 | 3.67 |
| Betfair | over 2.5 | 4869 | 1.08 | 1.85 | 3.80 | 1.22 | 2.75 |
| Betfair | under 2.5 | 4869 | 1.22 | 1.87 | 7.00 | 1.40 | 4.00 |
| **BetMGM** | **alternate_over_2_5** | **621** | **1.18** | **1.69** | **23.00** | **1.25** | **10.90** |
| BetMGM | alternate_under_2_5 | 621 | 1.00 | 2.05 | 4.50 | 1.04 | 3.72 |
| BetOnline.ag | alternate_over_2_5 | 5 | 1.38 | 1.38 | 1.69 | 1.38 | 1.68 |
| BetOnline.ag | alternate_under_2_5 | 5 | 2.15 | 3.05 | 3.15 | 2.18 | 3.15 |
| BetRivers | alternate_over_2_5 | 1819 | 1.09 | 1.73 | 4.00 | 1.22 | 3.00 |
| BetRivers | alternate_under_2_5 | 1819 | 1.23 | 2.08 | 6.75 | 1.36 | 4.29 |
| BetVictor | over 2.5 | 1761 | 1.06 | 1.72 | 3.00 | 1.20 | 2.46 |
| BetVictor | under 2.5 | 1761 | 1.32 | 1.97 | 7.50 | 1.46 | 4.50 |
| Betway | over 2.5 | 135 | 1.20 | 1.80 | 2.80 | 1.26 | 2.57 |
| Betway | under 2.5 | 135 | 1.38 | 1.91 | 4.33 | 1.45 | 3.83 |
| Bovada | alternate_over_2_5 | 1294 | 1.19 | 1.70 | 3.65 | 1.28 | 3.05 |
| Bovada | alternate_under_2_5 | 1294 | 1.29 | 2.18 | 4.80 | 1.39 | 3.75 |
| Bwin | over 2.5 | 135 | 1.22 | 1.78 | 2.80 | 1.26 | 2.52 |
| Bwin | under 2.5 | 135 | 1.37 | 1.90 | 3.90 | 1.45 | 3.60 |
| Caesars | alternate_over_2_5 | 501 | 1.17 | 1.71 | 3.10 | 1.22 | 2.65 |
| Caesars | alternate_under_2_5 | 501 | 1.33 | 2.10 | 4.80 | 1.45 | 4.00 |
| Coolbet | alternate_over_2_5 | 4269 | 1.13 | 1.80 | 6.70 | 1.25 | 2.90 |
| Coolbet | alternate_under_2_5 | 4269 | 1.11 | 2.05 | 6.20 | 1.44 | 4.15 |
| Dafabet | over 2.5 | 65 | 1.56 | 1.80 | 2.40 | 1.57 | 2.33 |
| Dafabet | under 2.5 | 65 | 1.58 | 2.02 | 2.36 | 1.59 | 2.34 |
| DraftKings | alternate_over_2_5 | 394 | 1.54 | 1.83 | 2.50 | 1.56 | 2.45 |
| DraftKings | alternate_under_2_5 | 394 | 1.48 | 1.87 | 2.30 | 1.49 | 2.30 |
| FanDuel | alternate_over_2_5 | 1150 | 1.13 | 1.67 | 3.60 | 1.22 | 2.78 |
| FanDuel | alternate_under_2_5 | 1150 | 1.29 | 2.18 | 6.00 | 1.44 | 4.03 |
| Fonbet | over 2.5 | 135 | 1.20 | 1.85 | 2.85 | 1.28 | 2.62 |
| Fonbet | under 2.5 | 135 | 1.42 | 1.95 | 4.15 | 1.46 | 3.73 |
| LiveScore Bet (EU) | alternate_over_2_5 | 2196 | 1.09 | 1.73 | 3.55 | 1.18 | 2.80 |
| LiveScore Bet (EU) | alternate_under_2_5 | 2196 | 1.26 | 2.00 | 6.10 | 1.39 | 4.25 |
| Marathonbet | over 2.5 | 5664 | 1.06 | 1.85 | 3.74 | 1.21 | 2.66 |
| Marathonbet | under 2.5 | 5664 | 1.22 | 1.94 | 6.10 | 1.41 | 3.72 |
| Pinnacle | alternate_over_2_5 | 5274 | 1.12 | 1.83 | 4.04 | 1.30 | 3.04 |
| Pinnacle | alternate_under_2_5 | 5274 | 1.26 | 2.07 | 6.41 | 1.41 | 3.60 |
| Pinnacle | over 2.5 | 5542 | 1.13 | 1.90 | 4.01 | 1.32 | 2.84 |
| Pinnacle | over_2.5 | 11864 | 1.72 | 1.93 | 5.16 | 1.79 | 2.09 |
| Pinnacle | under 2.5 | 5542 | 1.26 | 1.98 | 6.05 | 1.42 | 3.45 |
| Pinnacle | under_2.5 | 11864 | 1.18 | 1.93 | 2.13 | 1.79 | 2.08 |
| SBO | over 2.5 | 2967 | 1.49 | 1.92 | 2.78 | 1.56 | 2.58 |
| SBO | under 2.5 | 2967 | 1.46 | 1.94 | 2.58 | 1.52 | 2.49 |
| Superbet | over 2.5 | 2906 | 1.11 | 1.88 | 3.90 | 1.27 | 2.82 |
| Superbet | under 2.5 | 2906 | 1.26 | 1.90 | 6.70 | 1.42 | 3.70 |
| Suprabets | alternate_over_2_5 | 4853 | 1.15 | 1.83 | 6.20 | 1.26 | 2.98 |
| Suprabets | alternate_under_2_5 | 4853 | 1.15 | 2.07 | 6.40 | 1.44 | 4.15 |
| Tipico | over 2.5 | 41 | 1.37 | 1.87 | 2.85 | 1.37 | 2.75 |
| Tipico | under 2.5 | 41 | 1.40 | 1.87 | 3.00 | 1.43 | 2.98 |
| Unibet | over 2.5 | 4897 | 1.10 | 1.82 | 3.90 | 1.25 | 2.70 |
| **Unibet** | **over_2.5** | **10081** | **1.43** | **1.89** | **21.00** | **1.52** | **2.60** |
| Unibet | under 2.5 | 4897 | 1.24 | 1.91 | 6.50 | 1.40 | 3.80 |
| Unibet | under_2.5 | 10081 | 1.01 | 1.89 | 2.80 | 1.47 | 2.50 |
| Unibet (IT) | over_2.5 | 19 | 1.36 | 1.70 | 2.32 | 1.39 | 2.30 |
| Unibet (IT) | under_2.5 | 19 | 1.56 | 2.06 | 2.85 | 1.56 | 2.76 |
| Unibet (NL) | over_2.5 | 22 | 1.36 | 1.73 | 2.32 | 1.40 | 2.30 |
| Unibet (NL) | under_2.5 | 22 | 1.56 | 2.00 | 2.85 | 1.56 | 2.74 |
| William Hill | alternate_over_2_5 | 4692 | 1.12 | 1.73 | 4.20 | 1.22 | 2.75 |
| William Hill | alternate_under_2_5 | 4692 | 1.25 | 2.00 | 6.00 | 1.40 | 4.00 |
| William Hill | over_2.5 | 9923 | 1.12 | 1.80 | 3.25 | 1.29 | 2.62 |
| William Hill | under_2.5 | 9923 | 1.33 | 1.91 | 5.50 | 1.44 | 3.50 |
| WynnBET | alternate_over_2_5 | 992 | 1.16 | 1.81 | 5.50 | 1.26 | 2.90 |
| WynnBET | alternate_under_2_5 | 992 | 1.15 | 2.05 | 5.60 | 1.44 | 4.02 |

Le due righe in **grassetto** sono le uniche in cui il massimo si stacca in modo abnorme sia dalla mediana sia dal P99 (che per il resto della tabella resta sempre sotto ~4.5): **Unibet / `over_2.5`** (max 21.00 contro P99 2.60, mediana 1.89) e **BetMGM / `alternate_over_2_5`** (max 23.00, ma qui anche il P99 e' gia' a 10.90 — non e' solo un singolo outlier isolato, la coda destra e' sistematicamente gonfiata).

## a) Le 20 quote piu' alte per "Over 2.5" (qualunque variante di nome esito)

| # | Bookmaker | Esito | Valore | id_fixture | Data partita |
|---:|---|---|---:|---|---|
| 1 | Unibet | over_2.5 | 21.00 | (nullo — vedi nota sotto) | 2024-11-28 |
| 2 | Unibet | over_2.5 | 18.00 | (nullo) | 2024-11-07 |
| 3 | Unibet | over_2.5 | 18.00 | (nullo) | 2024-11-28 |
| 4 | Unibet | over_2.5 | 18.00 | 1063342 | 2024-05-10 |
| 5 | Unibet | over_2.5 | 18.00 | 1299216 | 2024-11-28 |
| 6 | Unibet | over_2.5 | 18.00 | 1299213 | 2024-11-28 |
| 7 | Unibet | over_2.5 | 17.00 | (nullo) | 2024-11-28 |
| 8 | Unibet | over_2.5 | 17.00 | (nullo) | 2024-11-28 |
| 9 | Unibet | over_2.5 | 17.00 | 1049181 | 2024-05-18 |
| 10 | Unibet | over_2.5 | 17.00 | 1299359 | 2024-11-28 |
| 11 | Unibet | over_2.5 | 17.00 | 1299340 | 2024-11-07 |
| 12 | Unibet | over_2.5 | 16.00 | (nullo) | 2024-11-28 |
| 13 | Unibet | over_2.5 | 16.00 | 1299214 | 2024-11-28 |
| 14 | Unibet | over_2.5 | 16.00 | 1299361 | 2024-11-28 |
| 15 | Unibet | over_2.5 | 15.00 | (nullo) | 2024-11-28 |
| 16 | Unibet | over_2.5 | 15.00 | 1234828 | 2024-12-29 |
| 17 | Unibet | over_2.5 | 14.00 | (nullo) | 2024-12-19 |
| 18 | Unibet | over_2.5 | 14.00 | 1063337 | 2024-05-10 |
| 19 | Unibet | over_2.5 | 14.00 | 1125232 | 2024-05-10 |
| 20 | Unibet | over_2.5 | 14.00 | 1299217 | 2024-11-28 |

Nota su "id_fixture nullo": alcune righe `Odds` hanno `id_fixture` a `NULL` sulla `Match` collegata (2.546 fixture su 20.759, ~12%, hanno `id_fixture` NULL nell'intero dataset `under_over_2_5` — non e' un problema specifico di queste 20 righe). Sono verosimilmente fixture arrivate SOLO dall'ingestion quote (`odds-api`, identificata da `id_events`/`id_alternate_events`) senza mai essere abbinata a una fixture `api-sports` con `id_fixture` numerico. Per queste ho riportato comunque bookmaker/valore/data, presi dalla stessa riga `Odds`.

## b) Concentrazione: bookmaker e periodo

Le 20 quote piu' alte sono **tutte** sullo stesso bookmaker (**Unibet**) e sulla stessa variante di chiave esito (**`over_2.5`**, formato underscore-punto). Non e' un caso isolato: allargando la soglia a "qualunque quota Over 2.5 > 10" (non solo la top 20) si trovano **37 quote** in totale su tutto il dataset, cosi' ripartite:

- **Per bookmaker**: Unibet 29/37, BetMGM 8/37 — nessun altro bookmaker supera mai 10.00 su un esito "over 2.5" in tutto il dataset.
- **Per periodo**: 31/37 datate 2024 (in particolare un cluster molto fitto attorno al 28-29 novembre 2024 e un secondo attorno al 10 maggio 2024), le restanti 3 nel 2023 e 3 nel 2025. Non e' quindi un fenomeno distribuito uniformemente nel tempo: si concentra in poche date del 2024.

Conclusione punto (b): la contaminazione e' imputabile in modo netto a **un bookmaker (Unibet, e in misura minore BetMGM)** e a **un periodo circoscritto (soprattutto fine 2024)**, non e' un problema diffuso su tutti i bookmaker o costante nel tempo.

## c) Sorgente (`odds_from`) delle righe coinvolte

Tutte e 37 le quote anomale (>10 su un esito "over 2.5") hanno `odds_from = 'odds-api'`, **mai** `'sports-api'`. Questo e' coerente con quanto visto nella tabella per bookmaker: `sports-api` alimenta solo la variante di chiave `"over 2.5"` (spazio) per bookmaker come Bet365, Betfair, Marathonbet, Pinnacle, ecc. — e su NESSUNA di queste il massimo supera mai ~5.6. La variante `"over_2.5"` (underscore-punto) e `"alternate_over_2_5"`, prodotte invece da `odds-api`, sono le uniche a contenere valori anomali.

Va pero' precisato che `odds-api` di per se' non e' "rotta": William Hill (9.923 quote, tutte da `odds-api`) ha un massimo del tutto plausibile (3.25 su `over_2.5`), cosi' come Pinnacle-via-odds-api (max 5.16). Il problema non e' quindi la sorgente `odds-api` in blocco, ma specificamente **le righe Unibet/BetMGM ingerite da `odds-api`**, concentrate soprattutto a fine 2024 — compatibile con un problema puntuale di una vecchia ingestion/normalizzazione per quei bookmaker, non con un difetto strutturale della sorgente `odds-api` in generale.

Conclusione punto (c): confermato — le righe contaminate provengono esclusivamente da `odds_from='odds-api'`, mai da `'sports-api'`, e dentro `odds-api` sono isolabili a due bookmaker specifici (Unibet, BetMGM) in un periodo circoscritto (soprattutto novembre-dicembre 2024).

## Raccomandazione

Prima di usare `under_over_2_5` per training/serving, filtrare o cappare le quote con `odds_from='odds-api'` per bookmaker `Unibet`/`BetMGM` quando `valore > ~6` (il P99 di tutti gli altri bookmaker/sorgenti non supera mai questa soglia), oppure investigare direttamente presso il fornitore `odds-api` la finestra novembre-dicembre 2024 per capire se e' un bug di parsing (es. quota decimale letta come intera, o linea sbagliata mappata su "over 2.5") lato ingestion storica.
