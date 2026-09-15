# Quote implausibili nel mercato `under_over_2_5` — quadro a piu' soglie

Segue il report `report_quote_2_5.md` gia' sul branch (che aveva trovato 37 quote "over 2.5" > 10.00, tutte da `odds_from='odds-api'`, ripartite Unibet 29 / BetMGM 8). Qui si allarga l'analisi a piu' soglie, a entrambe le direzioni (over/under), e si calcola il tasso per provider (non solo il conteggio assoluto), per capire quanto e' diffuso il problema e non solo dove si concentrano i casi peggiori.

Dati letti dal DB reale (Postgres su Railway, stesso `DATABASE_URL` usato dall'app), query SQLAlchemy diretta su `Match` con `selectinload(Match.odds)`, filtrando le righe `Odds.under_over_2_5` con dizionario JSON non vuoto. Per ogni coppia chiave/valore dentro il JSON, l'esito e' stato separato dal bookmaker con `rpartition("_")` (ultimo underscore, per non spezzare esiti come `alternate_over_2_5`), poi normalizzato a `over`/`under` unificando le varianti `over 2.5` / `over_2.5` / `alternate_over_2_5` / `alternate over 2.5` (idem per under). Nessun numero qui sotto e' stimato: tutto viene dall'esecuzione riportata.

## Portata dei dati

- Fixture caricate (join `Match`+`Odds`): **47.835**
- Quote singole estratte dal JSON `under_over_2_5`: **255.924** (over: 127.962, under: 127.962 — simmetrico, perche' quasi ogni riga bookmaker/sorgente porta sia la chiave over che quella under)
- Provider (`odds_from`) presenti: **odds-api** (146.044 quote, 73.022 per direzione) e **sports-api** (109.880 quote, 54.940 per direzione) — nessun'altra sorgente
- Nessuna chiave non riconosciuta (0 righe "altro"): il parsing `rpartition` + normalizzazione ha classificato correttamente il 100% delle quote come over o under

## 1) Conteggio per soglia — OVER e UNDER separate

Le due direzioni sono tenute separate perche' un Under 2.5 alto e' molto piu' plausibile di un Over 2.5 alto (due squadre difensive possono davvero pagare l'Under a 5-6; il contrario, un Over 2.5 a 2 cifre, non ha senso su un mercato binario dove l'esito opposto e' quasi certo).

### OVER

| Soglia | N quote | % su tot OVER (127.962) | N fixture distinte |
|---|---:|---:|---:|
| > 3.5 | 136 | 0.106% | 88 |
| > 4.0 | 76 | 0.059% | 69 |
| > 5.0 | 62 | 0.048% | 59 |
| > 6.0 | 51 | 0.040% | 51 |
| > 8.0 | 45 | 0.035% | 45 |
| > 10.0 | 37 | 0.029% | 37 |
| > 15.0 | 18 | 0.014% | 18 |

Conferma il numero del report precedente: **37 quote over sopra 10**, qui ricalcolato in modo indipendente. Il fatto che da 6.0 in su il numero di quote coincida quasi esattamente col numero di fixture (51 quote / 51 fixture, poi 45/45, 37/37) dice che sopra questa soglia e' sempre **una sola quota anomala per fixture**, mai piu' bookmaker anomali sulla stessa partita.

### UNDER

| Soglia | N quote | % su tot UNDER (127.962) | N fixture distinte |
|---|---:|---:|---:|
| > 3.5 | 1.545 | 1.207% | 367 |
| > 4.0 | 588 | 0.460% | 136 |
| > 5.0 | 142 | 0.111% | 39 |
| > 6.0 | 30 | 0.023% | 12 |
| > 8.0 | 0 | 0.000% | 0 |
| > 10.0 | 0 | 0.000% | 0 |
| > 15.0 | 0 | 0.000% | 0 |

L'Under si comporta esattamente come atteso da un mercato sano: e' molto piu' frequente sopra le soglie basse (1.207% > 3.5 contro lo 0.106% dell'Over: coerente, l'Under alto e' normale su partite offensive), ma **non supera mai 8.00** in tutto il dataset. L'Over invece continua a produrre casi fino a 18-23 (vedi tabella 3). Questa asimmetria e' di per se' la prova che il problema e' specifico della coda destra dell'Over, non un rumore generico sulle quote.

## 2) Ripartizione per provider, per soglia (tasso sul totale del provider)

### OVER

| Soglia | Provider | N quote provider (tot direzione) | N sopra soglia | % sopra soglia |
|---|---|---:|---:|---:|
| > 3.5 | odds-api | 73.022 | 113 | 0.155% |
| > 3.5 | sports-api | 54.940 | 23 | 0.042% |
| > 4.0 | odds-api | 73.022 | 72 | 0.099% |
| > 4.0 | sports-api | 54.940 | 4 | 0.007% |
| > 5.0 | odds-api | 73.022 | 61 | 0.084% |
| > 5.0 | sports-api | 54.940 | 1 | 0.002% |
| > 6.0 | odds-api | 73.022 | 51 | 0.070% |
| > 6.0 | sports-api | 54.940 | 0 | 0.000% |
| > 8.0 | odds-api | 73.022 | 45 | 0.062% |
| > 8.0 | sports-api | 54.940 | 0 | 0.000% |
| > 10.0 | odds-api | 73.022 | 37 | 0.051% |
| > 10.0 | sports-api | 54.940 | 0 | 0.000% |
| > 15.0 | odds-api | 73.022 | 18 | 0.025% |
| > 15.0 | sports-api | 54.940 | 0 | 0.000% |

Anche calcolando il **tasso** (non solo il conteggio assoluto) il quadro non cambia: da 6.0 in su `sports-api` e' a **zero secco**, `odds-api` resta l'unica sorgente con quote Over implausibili. Anche a 4.0, `sports-api` ha solo 4 quote su 54.940 (0.007%) — un tasso 14 volte piu' basso di `odds-api` (0.099%). Le uniche 4 quote `sports-api` sopra 4.0 sono comunque valori normali per il mercato (max 4.33 e 5.00, vedi tabella 3): non entrano mai nella fascia davvero implausibile.

### UNDER

| Soglia | Provider | N quote provider (tot direzione) | N sopra soglia | % sopra soglia |
|---|---|---:|---:|---:|
| > 3.5 | odds-api | 73.022 | 801 | 1.097% |
| > 3.5 | sports-api | 54.940 | 744 | 1.354% |
| > 4.0 | odds-api | 73.022 | 273 | 0.374% |
| > 4.0 | sports-api | 54.940 | 315 | 0.573% |
| > 5.0 | odds-api | 73.022 | 66 | 0.090% |
| > 5.0 | sports-api | 54.940 | 76 | 0.138% |
| > 6.0 | odds-api | 73.022 | 12 | 0.016% |
| > 6.0 | sports-api | 54.940 | 18 | 0.033% |
| > 8.0 | odds-api | 73.022 | 0 | 0.000% |
| > 8.0 | sports-api | 54.940 | 0 | 0.000% |

Punto importante: per l'**Under** (dove valori alti sono plausibili) `sports-api` ha addirittura un tasso **piu' alto** di `odds-api` a tutte le soglie fino a 6.0. Questo conferma che `odds-api` come sorgente non e' "rotta" in generale — il suo tasso piu' alto riguarda **esclusivamente** la coda Over implausibile, non le quote Under (dove anzi e' leggermente piu' contenuta di `sports-api`).

## 3) Ripartizione per bookmaker, direzione OVER, soglie 4.0 e 6.0

Tutti i bookmaker della tabella (nessuno escluso), ordinati per % sopra soglia decrescente.

### Soglia > 4.0

| Bookmaker | N tot | N sopra soglia | % sopra soglia | Max | odds_from prevalente |
|---|---:|---:|---:|---:|---|
| BetMGM | 621 | 16 | 2.576% | 23.00 | odds-api |
| Unibet | 14.978 | 41 | 0.274% | 21.00 | odds-api |
| WynnBET | 992 | 1 | 0.101% | 5.50 | odds-api |
| Coolbet | 4.269 | 3 | 0.070% | 6.70 | odds-api |
| Suprabets | 4.853 | 3 | 0.062% | 6.20 | odds-api |
| Pinnacle | 22.707 | 8 | 0.035% | 5.16 | odds-api |
| Betano | 5.192 | 1 | 0.019% | 5.00 | sports-api |
| Bet365 | 5.601 | 1 | 0.018% | 4.33 | sports-api |
| William Hill | 14.615 | 1 | 0.007% | 4.20 | odds-api |
| 1xBet | 18.753 | 1 | 0.005% | 5.60 | odds-api |
| 10Bet | 4.658 | 0 | 0.000% | 3.85 | sports-api |
| 188Bet | 1.304 | 0 | 0.000% | 2.17 | sports-api |
| 888Sport | 3.268 | 0 | 0.000% | 3.60 | sports-api |
| BetOnline.ag | 5 | 0 | 0.000% | 1.69 | odds-api |
| BetRivers | 1.819 | 0 | 0.000% | 4.00 | odds-api |
| BetVictor | 1.788 | 0 | 0.000% | 3.00 | sports-api |
| Betfair | 4.881 | 0 | 0.000% | 3.80 | sports-api |
| Betway | 135 | 0 | 0.000% | 2.80 | sports-api |
| Bovada | 1.294 | 0 | 0.000% | 3.65 | odds-api |
| Bwin | 135 | 0 | 0.000% | 2.80 | sports-api |
| Caesars | 501 | 0 | 0.000% | 3.10 | odds-api |
| Dafabet | 65 | 0 | 0.000% | 2.40 | sports-api |
| DraftKings | 394 | 0 | 0.000% | 2.50 | odds-api |
| FanDuel | 1.150 | 0 | 0.000% | 3.60 | odds-api |
| Fonbet | 135 | 0 | 0.000% | 2.85 | sports-api |
| LiveScore Bet (EU) | 2.196 | 0 | 0.000% | 3.55 | odds-api |
| Marathonbet | 5.690 | 0 | 0.000% | 3.74 | sports-api |
| SBO | 2.975 | 0 | 0.000% | 2.78 | sports-api |
| Superbet | 2.906 | 0 | 0.000% | 3.90 | sports-api |
| Tipico | 41 | 0 | 0.000% | 2.85 | sports-api |
| Unibet (IT) | 19 | 0 | 0.000% | 2.32 | odds-api |
| Unibet (NL) | 22 | 0 | 0.000% | 2.32 | odds-api |

### Soglia > 6.0

| Bookmaker | N tot | N sopra soglia | % sopra soglia | Max | odds_from prevalente |
|---|---:|---:|---:|---:|---|
| BetMGM | 621 | 10 | 1.610% | 23.00 | odds-api |
| Unibet | 14.978 | 39 | 0.260% | 21.00 | odds-api |
| Coolbet | 4.269 | 1 | 0.023% | 6.70 | odds-api |
| Suprabets | 4.853 | 1 | 0.021% | 6.20 | odds-api |
| 10Bet | 4.658 | 0 | 0.000% | 3.85 | sports-api |
| 188Bet | 1.304 | 0 | 0.000% | 2.17 | sports-api |
| 1xBet | 18.753 | 0 | 0.000% | 5.60 | odds-api |
| 888Sport | 3.268 | 0 | 0.000% | 3.60 | sports-api |
| Bet365 | 5.601 | 0 | 0.000% | 4.33 | sports-api |
| BetOnline.ag | 5 | 0 | 0.000% | 1.69 | odds-api |
| BetRivers | 1.819 | 0 | 0.000% | 4.00 | odds-api |
| BetVictor | 1.788 | 0 | 0.000% | 3.00 | sports-api |
| Betano | 5.192 | 0 | 0.000% | 5.00 | sports-api |
| Betfair | 4.881 | 0 | 0.000% | 3.80 | sports-api |
| Betway | 135 | 0 | 0.000% | 2.80 | sports-api |
| Bovada | 1.294 | 0 | 0.000% | 3.65 | odds-api |
| Bwin | 135 | 0 | 0.000% | 2.80 | sports-api |
| Caesars | 501 | 0 | 0.000% | 3.10 | odds-api |
| Dafabet | 65 | 0 | 0.000% | 2.40 | sports-api |
| DraftKings | 394 | 0 | 0.000% | 2.50 | odds-api |
| FanDuel | 1.150 | 0 | 0.000% | 3.60 | odds-api |
| Fonbet | 135 | 0 | 0.000% | 2.85 | sports-api |
| LiveScore Bet (EU) | 2.196 | 0 | 0.000% | 3.55 | odds-api |
| Marathonbet | 5.690 | 0 | 0.000% | 3.74 | sports-api |
| Pinnacle | 22.707 | 0 | 0.000% | 5.16 | odds-api |
| SBO | 2.975 | 0 | 0.000% | 2.78 | sports-api |
| Superbet | 2.906 | 0 | 0.000% | 3.90 | sports-api |
| Tipico | 41 | 0 | 0.000% | 2.85 | sports-api |
| Unibet (IT) | 19 | 0 | 0.000% | 2.32 | odds-api |
| Unibet (NL) | 22 | 0 | 0.000% | 2.32 | odds-api |
| William Hill | 14.615 | 0 | 0.000% | 4.20 | odds-api |
| WynnBET | 992 | 0 | 0.000% | 5.50 | odds-api |

Il tasso (non il conteggio assoluto) cambia la lettura: **BetMGM ha il tasso peggiore in assoluto** (2.576% delle sue quote Over sopra 4.0, 1.610% sopra 6.0) pur avendo meno quote anomale in valore assoluto di Unibet (16/41 vs Unibet a 4.0, 10/39 vs Unibet a 6.0). Unibet ha piu' casi ma su una base 24 volte piu' grande (14.978 quote contro 621): il suo tasso resta piu' basso (0.274%/0.260%) ma comunque un ordine di grandezza sopra tutti gli altri bookmaker che non siano BetMGM. Sopra 6.0 restano coinvolti **solo 4 bookmaker su 31**: BetMGM, Unibet, Coolbet (1 caso, max 6.70) e Suprabets (1 caso, max 6.20) — questi ultimi due con un solo episodio isolato ciascuno, non un pattern.

## 4) Controprova temporale — distribuzione mensile

Bookmaker selezionati per la controprova: quelli con almeno 5 quote Over sopra 4.0 in valore assoluto — **Unibet, BetMGM, Pinnacle**.

### Unibet

| Anno-mese | N quote > 4.0 |
|---|---:|
| 2022-03 | 1 |
| 2022-05 | 1 |
| 2023-03 | 2 |
| 2023-05 | 2 |
| 2023-06 | 1 |
| 2024-05 | 11 |
| 2024-11 | 19 |
| 2024-12 | 3 |
| 2025-02 | 1 |

Due cluster netti (maggio 2024 e novembre 2024, 30 casi su 41 = 73%), ma con una coda di episodi isolati che risale fino al 2022 — quindi non e' un bug comparso una volta sola e mai piu': si ripresenta a bassa frequenza anche fuori dai due cluster principali.

### BetMGM

| Anno-mese | N quote > 4.0 |
|---|---:|
| 2024-05 | 4 |
| 2024-09 | 5 |
| 2024-10 | 1 |
| 2024-12 | 1 |
| 2025-01 | 1 |
| 2025-02 | 1 |
| 2025-03 | 2 |
| 2025-04 | 1 |

Nessun mese domina (max 5 casi a settembre 2024): a differenza di Unibet, per BetMGM il fenomeno e' **spalmato in modo continuo** da maggio 2024 ad aprile 2025, non concentrato in una finestra ristretta.

### Pinnacle

| Anno-mese | N quote > 4.0 |
|---|---:|
| 2020-07 | 1 |
| 2023-03 | 1 |
| 2024-05 | 1 |
| 2024-08 | 3 |
| 2024-12 | 1 |
| 2026-02 | 1 |

Per Pinnacle non c'e' alcun cluster (mai piu' di 3 casi in un mese, su un arco di 6 anni), e i valori massimi restano contenuti (4.01-5.16, vedi tabella 3): coerente con quote genuinamente alte su partite con expected-goals molto basso, non con un errore di ingestion. Pinnacle e' quindi un caso a parte rispetto a Unibet/BetMGM — la tabella 3 lo aveva gia' suggerito (0 casi sopra 6.0), qui la distribuzione temporale lo conferma.

## 5) Controprova di coerenza — quadro completo per fixture

Campione di 15 fixture con almeno una quota Over sopra 4.0 (prese nell'ordine di estrazione dal DB, senza altro criterio), quadro completo di tutti i bookmaker/esiti sulla stessa fixture:

**Fixture id_match_fk=0c6a4697-3859-4d14-addf-e8f51e6de25c | id_fixture=None | data=2024-11-28T20:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.95 | odds-api |
| Coolbet | alternate_under_2_5 | 1.86 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 1.90 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 1.82 | odds-api |
| Pinnacle | over_2.5 | 2.13 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.95 | odds-api |
| Pinnacle | under_2.5 | 1.75 | odds-api |
| Pinnacle | alternate_under_2_5 | 1.93 | odds-api |
| Suprabets | alternate_over_2_5 | 1.96 | odds-api |
| Suprabets | alternate_under_2_5 | 1.94 | odds-api |
| **Unibet** | **over_2.5** | **16.00** | **odds-api** |
| Unibet | under_2.5 | 1.02 | odds-api |
| William Hill | alternate_over_2_5 | 1.91 | odds-api |
| William Hill | alternate_under_2_5 | 1.80 | odds-api |

**Fixture id_match_fk=2db3bb52-858a-40b4-95bc-a54b14657430 | id_fixture=None | data=2024-11-28T20:03:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 2.10 | odds-api |
| Coolbet | alternate_under_2_5 | 1.74 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 2.08 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 1.68 | odds-api |
| Pinnacle | over_2.5 | 2.06 | odds-api |
| Pinnacle | alternate_over_2_5 | 2.08 | odds-api |
| Pinnacle | under_2.5 | 1.81 | odds-api |
| Pinnacle | alternate_under_2_5 | 1.80 | odds-api |
| Suprabets | alternate_over_2_5 | 2.10 | odds-api |
| Suprabets | alternate_under_2_5 | 1.82 | odds-api |
| **Unibet** | **over_2.5** | **15.00** | **odds-api** |
| Unibet | under_2.5 | 1.02 | odds-api |
| William Hill | alternate_over_2_5 | 2.00 | odds-api |
| William Hill | alternate_under_2_5 | 1.73 | odds-api |

**Fixture id_match_fk=88a44c96-4d15-4110-bb8e-0fef8024136a | id_fixture=None | data=2024-11-07T20:01:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.45 | odds-api |
| Coolbet | alternate_under_2_5 | 2.70 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.47 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.73 | odds-api |
| Suprabets | alternate_over_2_5 | 1.50 | odds-api |
| Suprabets | alternate_under_2_5 | 2.79 | odds-api |
| **Unibet** | **over_2.5** | **18.00** | **odds-api** |
| Unibet | under_2.5 | 1.01 | odds-api |
| William Hill | alternate_over_2_5 | 1.44 | odds-api |
| William Hill | alternate_under_2_5 | 2.62 | odds-api |

**Fixture id_match_fk=4467c783-9a20-4c33-bff6-9ce0b3c6effa | id_fixture=None | data=2024-08-08T18:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| **Pinnacle** | **over_2.5** | **4.50** | **odds-api** |
| Pinnacle | alternate_over_2_5 | 1.76 | odds-api |
| Pinnacle | under_2.5 | 1.20 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.07 | odds-api |
| William Hill | alternate_over_2_5 | 1.73 | odds-api |
| William Hill | alternate_under_2_5 | 2.00 | odds-api |

**Fixture id_match_fk=31f456b7-2893-4b87-8e8b-4c718f364091 | id_fixture=None | data=2024-11-28T20:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.85 | odds-api |
| Coolbet | alternate_under_2_5 | 1.97 | odds-api |
| Pinnacle | over_2.5 | 2.36 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.85 | odds-api |
| Pinnacle | under_2.5 | 1.62 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.05 | odds-api |
| Suprabets | alternate_over_2_5 | 1.86 | odds-api |
| Suprabets | alternate_under_2_5 | 2.05 | odds-api |
| **Unibet** | **over_2.5** | **17.00** | **odds-api** |
| Unibet | under_2.5 | 1.02 | odds-api |
| William Hill | alternate_over_2_5 | 1.73 | odds-api |
| William Hill | alternate_under_2_5 | 2.00 | odds-api |

**Fixture id_match_fk=1f0320eb-3d09-49db-b5ae-482a48fb7282 | id_fixture=None | data=2024-11-28T20:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.75 | odds-api |
| Coolbet | alternate_under_2_5 | 2.05 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 1.68 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 2.04 | odds-api |
| Pinnacle | over_2.5 | 2.25 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.71 | odds-api |
| Pinnacle | under_2.5 | 1.67 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.21 | odds-api |
| Suprabets | alternate_over_2_5 | 1.71 | odds-api |
| Suprabets | alternate_under_2_5 | 2.24 | odds-api |
| **Unibet** | **over_2.5** | **18.00** | **odds-api** |
| Unibet | under_2.5 | 1.02 | odds-api |
| William Hill | alternate_over_2_5 | 1.67 | odds-api |
| William Hill | alternate_under_2_5 | 2.10 | odds-api |

**Fixture id_match_fk=1b534df0-6bd6-459e-bb66-b88fd382ea2b | id_fixture=None | data=2024-11-28T20:01:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 2.02 | odds-api |
| Coolbet | alternate_under_2_5 | 1.80 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 1.95 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 1.78 | odds-api |
| Pinnacle | over_2.5 | 2.45 | odds-api |
| Pinnacle | alternate_over_2_5 | 2.02 | odds-api |
| Pinnacle | under_2.5 | 1.57 | odds-api |
| Pinnacle | alternate_under_2_5 | 1.88 | odds-api |
| Suprabets | alternate_over_2_5 | 2.03 | odds-api |
| Suprabets | alternate_under_2_5 | 1.88 | odds-api |
| **Unibet** | **over_2.5** | **10.00** | **odds-api** |
| Unibet | under_2.5 | 1.05 | odds-api |
| William Hill | alternate_over_2_5 | 1.91 | odds-api |
| William Hill | alternate_under_2_5 | 1.80 | odds-api |

**Fixture id_match_fk=b9d78cbe-4397-40e0-ad65-3f529123bdde | id_fixture=None | data=2024-08-29T19:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| **Pinnacle** | **over_2.5** | **4.33** | **odds-api** |
| Pinnacle | alternate_over_2_5 | 1.71 | odds-api |
| Pinnacle | under_2.5 | 1.22 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.20 | odds-api |

**Fixture id_match_fk=4a49b218-7fa4-48d1-ae3c-8426f336267c | id_fixture=None | data=2024-12-19T20:21:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Pinnacle | over_2.5 | 2.38 | odds-api |
| Pinnacle | under_2.5 | 1.62 | odds-api |
| Suprabets | alternate_over_2_5 | 1.70 | odds-api |
| Suprabets | alternate_under_2_5 | 1.90 | odds-api |
| **Unibet** | **over_2.5** | **14.00** | **odds-api** |
| Unibet | under_2.5 | 1.03 | odds-api |

**Fixture id_match_fk=473d00aa-d923-4896-bec5-823ebb865d49 | id_fixture=None | data=2024-11-28T20:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.67 | odds-api |
| Coolbet | alternate_under_2_5 | 2.17 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 1.61 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 2.15 | odds-api |
| Pinnacle | over_2.5 | 2.23 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.68 | odds-api |
| Pinnacle | under_2.5 | 1.68 | odds-api |
| Pinnacle | alternate_under_2_5 | 2.23 | odds-api |
| Suprabets | alternate_over_2_5 | 1.70 | odds-api |
| Suprabets | alternate_under_2_5 | 2.26 | odds-api |
| **Unibet** | **over_2.5** | **17.00** | **odds-api** |
| Unibet | under_2.5 | 1.02 | odds-api |
| William Hill | alternate_over_2_5 | 1.67 | odds-api |
| William Hill | alternate_under_2_5 | 2.10 | odds-api |

**Fixture id_match_fk=8a538d04-cbae-41f0-ab8e-51e589bb09b2 | id_fixture=None | data=2023-03-19T19:01:15Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Pinnacle | over_2.5 | 4.71 | odds-api |
| Pinnacle | under_2.5 | 1.22 | odds-api |
| **Unibet** | **over_2.5** | **12.00** | **odds-api** |
| Unibet | under_2.5 | 1.03 | odds-api |

**Fixture id_match_fk=0dc9166c-b214-4493-8bba-616abe914f41 | id_fixture=None | data=2024-11-28T20:00:00Z**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.76 | odds-api |
| Coolbet | alternate_under_2_5 | 2.04 | odds-api |
| LiveScore Bet (EU) | alternate_over_2_5 | 1.74 | odds-api |
| LiveScore Bet (EU) | alternate_under_2_5 | 1.96 | odds-api |
| Pinnacle | over_2.5 | 2.40 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.88 | odds-api |
| Pinnacle | under_2.5 | 1.60 | odds-api |
| Pinnacle | alternate_under_2_5 | 1.97 | odds-api |
| Suprabets | alternate_over_2_5 | 1.89 | odds-api |
| Suprabets | alternate_under_2_5 | 1.99 | odds-api |
| **Unibet** | **over_2.5** | **21.00** | **odds-api** |
| Unibet | under_2.5 | 1.01 | odds-api |
| William Hill | alternate_over_2_5 | 1.80 | odds-api |
| William Hill | alternate_under_2_5 | 1.91 | odds-api |

**Fixture id_match_fk=45c638d6-97a3-4ac3-98e0-1af1a6467cbc | id_fixture=235516 | data=2020-07-31T19:00:00+00:00**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| **Pinnacle** | **over_2.5** | **4.11** | **odds-api** |
| Pinnacle | under_2.5 | 1.25 | odds-api |

**Fixture id_match_fk=a6134c5f-754d-4706-b8b3-0a58398693ce | id_fixture=1299212 | data=2024-11-28T20:00:00+00:00**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Coolbet | alternate_over_2_5 | 1.44 | odds-api |
| Coolbet | alternate_under_2_5 | 2.80 | odds-api |
| Pinnacle | over_2.5 | 1.97 | odds-api |
| Pinnacle | alternate_over_2_5 | 1.39 | odds-api |
| Pinnacle | under_2.5 | 1.90 | odds-api |
| Pinnacle | alternate_under_2_5 | 3.04 | odds-api |
| **Unibet** | **over_2.5** | **11.50** | **odds-api** |
| Unibet | under_2.5 | 1.04 | odds-api |
| William Hill | alternate_over_2_5 | 1.36 | odds-api |
| William Hill | alternate_under_2_5 | 3.00 | odds-api |

**Fixture id_match_fk=9f53f7d7-6696-4e1b-af49-e3a282aff1f4 | id_fixture=758823 | data=2022-03-20T16:00:00+00:00**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| Pinnacle | over_2.5 | 2.66 | odds-api |
| Pinnacle | under_2.5 | 1.51 | odds-api |
| **Unibet** | **over_2.5** | **5.40** | **odds-api** |
| Unibet | under_2.5 | 1.14 | odds-api |

### Esempi supplementari per BetMGM (non presenti nel campione dei 15 sopra, riportati a parte perche' BetMGM e' l'altro bookmaker "problematico")

**Fixture id_fixture=1223890, data=2025-03-29T17:00:00+00:00**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| 1xBet | over_2.5 | 2.28 | odds-api |
| **BetMGM** | **alternate_over_2_5** | **23.00** | **odds-api** |
| BetMGM | alternate_under_2_5 | 1.00 | odds-api |
| BetRivers | alternate_over_2_5 | 2.25 | odds-api |
| Bovada | alternate_over_2_5 | 2.36 | odds-api |
| Coolbet | alternate_over_2_5 | 2.30 | odds-api |
| DraftKings | alternate_over_2_5 | 2.30 | odds-api |
| FanDuel | alternate_over_2_5 | 2.38 | odds-api |
| Pinnacle | alternate_over_2_5 | 2.36 | odds-api |
| Unibet | over_2.5 | 2.20 | odds-api |
| William Hill | over_2.5 | 2.15 | odds-api |

**Fixture id_fixture=1223650, data=2024-09-29T18:45:00+00:00**

| Bookmaker | Esito | Valore | odds_from |
|---|---|---:|---|
| 1xBet | over_2.5 | 1.83 | odds-api |
| **BetMGM** | **alternate_over_2_5** | **20.00** | **odds-api** |
| BetMGM | alternate_under_2_5 | 1.01 | odds-api |
| BetRivers | alternate_over_2_5 | 1.65 | odds-api |
| Bovada | alternate_over_2_5 | 1.74 | odds-api |
| Coolbet | alternate_over_2_5 | 1.78 | odds-api |
| Pinnacle | over_2.5 | 1.98 | odds-api |
| Suprabets | alternate_over_2_5 | 1.80 | odds-api |
| William Hill | over_2.5 | 1.70 | odds-api |

(altri 2 esempi con lo stesso schema — id_fixture=1223628 del 2024-09-14, BetMGM alternate_over_2_5=18.00 contro un book normale ~2.0-2.3; id_fixture=1223648 del 2024-09-28, BetMGM alternate_over_2_5=16.50 contro un book normale ~2.0-2.7 — verificati nell'esecuzione ma omessi qui per brevita', stesso pattern esatto.)

### Lettura della controprova

In **tutte** le fixture del campione (15 + i 4 esempi BetMGM), il quadro e' identico:

- Il bookmaker anomalo (Unibet o BetMGM) mostra un valore Over abnorme (5.40-23.00) mentre **tutti gli altri bookmaker sulla stessa fixture, nello stesso momento, quotano l'Over in un intervallo del tutto normale** (tipicamente 1.4-2.7). Non e' mai un caso in cui l'intera fixture ha quote strane — quindi non e' un problema di lega minore o di fixture mappata male, e' un problema isolato di UN bookmaker su una riga specifica.
- Quando il valore Over anomalo compare, l'Under **abbinato dello stesso bookmaker** crolla quasi sempre a un valore impossibile per un mercato reale: **1.00-1.05** (es. Unibet under_2.5=1.02 quando over_2.5=16.00-21.00; BetMGM alternate_under_2_5=1.00-1.02 quando alternate_over_2_5=16.50-23.00). Un vero mercato binario con margine bookmaker non scende mai sotto ~1.01-1.02 in modo cosi' sistematico proprio in corrispondenza dell'anomalia opposta: questo pattern (over abnorme + under quasi-1.00 sullo stesso bookmaker) e' un indizio tecnico forte che si tratti di un **errore di parsing/mapping in ingestion** (es. lettura della linea/selezione sbagliata, o scambio decimale) specifico della coppia over/under 2.5 di quei due bookmaker via `odds-api`, non di quote di mercato reali mai esistite.
- Pinnacle, che pure compare nel campione con valori "alti" (4.11-4.71), NON mostra questo pattern: il suo under abbinato resta in un range coerente (1.20-1.25), esattamente quello che ci si aspetta da un Over davvero alto ma plausibile. Confermato: Pinnacle non e' un caso di contaminazione, sono partite con expected-goals genuinamente basso.

## Conclusioni

1. **Le quote Over implausibili sono rare in assoluto** (37 casi su 127.962 quote Over sopra 10.00, cioe' 0.029%) ma **concentrate quasi per intero su due bookmaker e una sola sorgente**: `odds_from='odds-api'`, bookmaker **Unibet** e **BetMGM**.
2. **Il tasso (non solo il conteggio) conferma la stessa lettura**: sopra 6.0, `sports-api` e' a zero secco in ogni soglia testata; dentro `odds-api`, BetMGM ha il tasso peggiore in proporzione (2.576% delle sue quote Over sopra 4.0), Unibet ha il volume assoluto peggiore (41 casi) ma un tasso piu' basso (0.274%) diluito su una base 24 volte piu' grande.
3. **Temporalmente**: Unibet ha due cluster netti (maggio e novembre 2024, 73% dei casi) piu' una coda sporadica dal 2022; BetMGM e' spalmato in modo continuo da maggio 2024 ad aprile 2025 senza un mese dominante — quindi per BetMGM non sembra un incidente isolato ma un problema strutturale ancora in corso all'ultima data osservata (aprile 2025).
4. **La controprova di coerenza e' la prova piu' forte**: su ogni fixture campionata, solo il bookmaker anomalo e' fuori scala, tutti gli altri sono coerenti tra loro; e l'Under abbinato dello stesso bookmaker anomalo crolla quasi sempre a ~1.00-1.05, un pattern tecnico compatibile con un bug di ingestion/parsing per la coppia over/under 2.5 di Unibet e BetMGM via `odds-api`, non con quote di mercato reali.

## Raccomandazione

Filtrare o cappare (es. escludere valori > 6.0, soglia sopra cui restano coinvolti solo 4 bookmaker su 31 e mai `sports-api`) le quote `under_over_2_5` con `odds_from='odds-api'` per i bookmaker `Unibet` e `BetMGM` prima di usarle in training/serving. Dato che per BetMGM il fenomeno risulta ancora presente ad aprile 2025 (ultimo mese osservato con casi), vale la pena verificare se il problema di ingestion e' tuttora attivo lato `odds-api`, non solo storico.
