# Costo di rimuovere le quote `odds_from='odds-api'` dal training di `under_over_2_5`

Misure eseguite sul DB reale (stesso `DATABASE_URL` dell'app, Railway `dev`), sulla popolazione selezionata da `FilterMarketService._search_matches` (filtri: `mean_statistics` non nullo, `odds` non nullo, `statistics` non nullo, `status='FT'`), ristretta alle partite con dizionario `under_over_2_5` non vuoto in almeno una riga Odds.

Script usato per la misura: `_scratch_costo_oddsapi.py` (temporaneo, eseguito e poi rimosso — non committato). Riusa `FilterMarketService._search_matches`/`build_dataset` senza modifiche permanenti al codice sorgente.

## 0. Popolazione di base

| metrica | valore |
|---|---|
| Fixture candidate da `_search_matches` (status FT, mean_statistics/odds/statistics non nulli) | 16127 |
| ...di cui con `under_over_2_5` vuoto su ENTRAMBI i provider (escluse dal training comunque) | 371 (2,30%) |
| **Popolazione che entra nel training `under_over_2_5`** (`under_over_2_5` non vuoto in almeno un provider) | **15756** |

## 1. Copertura per provider, a livello di partita

| gruppo | partite | % sulla popolazione (15756) |
|---|---:|---:|
| Hanno almeno una riga `sports-api` con `under_over_2_5` non vuoto | 3735 | 23,70% |
| Hanno almeno una riga `odds-api` con `under_over_2_5` non vuoto | 12021 | 76,30% |
| Hanno **ENTRAMBI** i provider (con `under_over_2_5` non vuoto su entrambi) | **0** | **0,00%** |
| Hanno **SOLO** `odds-api` (si perderebbero togliendo `odds-api`) | 12021 | 76,30% |
| Hanno **SOLO** `sports-api` (resterebbero) | 3735 | 23,70% |

**Risultato chiave: nessuna partita ha entrambi i provider con `under_over_2_5` popolato.** I due provider non si sovrappongono affatto su questo mercato: sono temporalmente disgiunti (vedi punto 3). Togliere `odds-api` non elimina un duplicato silenzioso — elimina il 76,30% delle partite disponibili per il training di `under_over_2_5`.

## 2. Quale riga sta in posizione `odds_list[0]` oggi (partite con entrambi i provider)

Non misurabile: **il gruppo "entrambi i provider" è vuoto (0 partite)**. Di conseguenza il bug descritto in `filter_market_service.py:388` (`_build_row` usa solo `odds_list[0]`, quindi se una partita avesse sia una riga `sports-api` sia una `odds-api` una delle due verrebbe ignorata) non ha impatto pratico sul mercato `under_over_2_5`: non esiste oggi nessuna partita in questa situazione. Il problema resta teoricamente valido per l'architettura, ma su questo mercato specifico è un non-evento empirico.

## 3. Distribuzione temporale delle partite "solo odds-api"

Le 12021 partite "solo `odds-api`" coincidono esattamente con le 12021 partite "ha `odds-api`" (essendo il gruppo "entrambi" vuoto): rappresentano quindi **tutta** la copertura storica del mercato precedente ad agosto 2025.

### Per anno (`date_match`)

| anno | partite solo odds-api |
|---|---:|
| 2020 | 1096 |
| 2021 | 2506 |
| 2022 | 2166 |
| 2023 | 2471 |
| 2024 | 2455 |
| 2025 | 1327 |

### Per stagione (`season`)

| stagione | partite solo odds-api |
|---|---:|
| 2019 | 267 |
| 2020 | 2206 |
| 2021 | 2430 |
| 2022 | 2034 |
| 2023 | 2589 |
| 2024 | 2495 |

**Il buco non è sparso: è un blocco temporale continuo.** Togliere `odds-api` cancella integralmente le stagioni 2019–2024 (giugno 2020 – agosto 2025) e lascia solo la copertura `sports-api`, che parte dal 15/08/2025 (vedi punto 4). Non resterebbe alcuna storia pre-2025/26 per addestrare o validare il modello su finestre temporali passate.

## 4. Confronto dei due dataset grezzi esportati

`build_dataset(market="under_over_2_5", fill_missing=False)` — NaN lasciati come tali (non azzerati).

| metrica | `under_over_2_5_raw.csv` (baseline, esistente) | `under_over_2_5_raw_no_oddsapi.csv` (nuovo, solo sports-api) |
|---|---:|---:|
| Shape (righe, colonne) | 15340 × 90 | 3735 × 90 |
| NaN totali | 9990 | 4860 |
| % celle NaN | 0,72% | 1,45% |
| Base rate di `y` (Over 2.5) | 0,5288 | 0,5414 |
| Periodo coperto (`prediction_at`, min) | 2020-06-11 20:00 UTC | 2025-08-15 14:45 UTC |
| Periodo coperto (`prediction_at`, max) | 2026-09-13 16:30 UTC | 2026-09-13 19:30 UTC |
| Bookmaker medi per partita (`odds_count`) | 12,97 | 19,95 |

Nota sulle 15340 righe della baseline vs le 15756 partite del punto 1: la differenza (416 partite) è dovuta a filtri successivi dentro `_build_row` non legati al provider — soprattutto partite `FT` senza `score_ft` risolvibile (target non calcolabile) o senza feature utili (`len(row) <= 3`). Sul sottoinsieme `sports-api`-only queste 416 non incidono: le 3735 righe del CSV coincidono esattamente con le 3735 partite "solo sports-api" del punto 1, segno che su questo sottoinsieme più recente il punteggio finale è sempre disponibile.

I bookmaker medi per partita sono più alti (quasi 20 vs ~13) sul sottoinsieme `sports-api`-only: le quote correnti dell'ingestion attiva coprono più bookmaker per singola partita rispetto agli snapshot storici `odds-api`.

## Sintesi

Rimuovere `odds_from='odds-api'` da `under_over_2_5`:
- non recupera nessun duplicato (0 partite con entrambi i provider oggi popolati su questo mercato — il bug su `odds_list[0]` è quindi irrilevante qui);
- riduce il dataset di training da 15340 a 3735 righe (-75,65%);
- comprime il periodo storico coperto da 6+ anni (giugno 2020) a poco più di 13 mesi (agosto 2025 – oggi);
- lascia un dataset con più bookmaker medi per partita ma nessuna copertura di stagioni passate per validazione temporale (`expanding_window_splits` su più stagioni, backtest storici, ecc. diventerebbero impraticabili con questi soli dati).

Il problema di qualità dato in `df_odds_service.py:258` (linea non filtrata per `point`, quote Over 2.5 fino a 23.00) riguarda quindi l'**unica fonte storica disponibile** per questo mercato prima di agosto 2025: non è possibile scartare `odds-api` senza perdere quel periodo, ma nemmeno tenerlo così com'è senza correggere il bug di parsing della linea.
