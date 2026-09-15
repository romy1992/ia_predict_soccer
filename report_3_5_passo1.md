# Rifacimento mercato `under_over_3_5` — Passo 1 (dato grezzo dal DB)

Dati letti dal DB reale (stesso `DATABASE_URL` usato dall'app), via
`FilterMarketService().build_dataset(market="under_over_3_5", fill_missing=False)`
per il Passo 1, e query SQLAlchemy diretta `select(Match).options(selectinload(Match.odds))`
per i Passi 2-3. Nessun numero in questo report e' stimato: tutto proviene
dall'esecuzione riportata. Sola lettura: nessuna quota modificata, nessun
modello riaddestrato.

## Passo 1 — estrazione grezza

Controllo bloccante superato: con `fill_missing=False` i valori mancanti
**non** sono azzerati (NaN totali > 0). Il codice in esecuzione e' quello
aggiornato.

| Metrica | Valore |
|---|---:|
| Shape dataset | 8.443 righe x 90 colonne |
| NaN totali | 6.048 |
| Base rate `y` (Over 3.5) | 31.55% |
| `prediction_at` min | 2023-05-03T16:00:00+00:00 |
| `prediction_at` max | 2026-09-13T19:30:00+00:00 |
| `odds_count` medio | 12.89 |
| Leghe distinte | 14 |
| Stagioni distinte | 5 (2022, 2023, 2024, 2025, 2026) |

CSV scritto in `scripts/analysis/_export/under_over_3_5_raw.csv` (8.99 MB,
sotto la soglia dei 50 MB).

## Passo 2 — quadro quote per bookmaker

Righe di quota estratte dal bucket JSON `Odds.under_over_3_5` di tutti i
`Match` a DB (47.835 match totali), chiave separata con
`FilterMarketService._split_outcome_and_bookmaker` (split sull'ultimo
underscore): **145.881** quote singole totali.

### Statistiche per bookmaker x esito (normalizzato: `alternate_over_3_5` -> `over_3_5`, `over 3.5` -> `over_3_5`)

| Bookmaker | Esito | N | Min | Mediana | Max | P1 | P99 | `odds_from` prevalente |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 10Bet | over_3_5 | 4666 | 1.30 | 3.20 | 8.50 | 1.67 | 5.50 | sports-api |
| 10Bet | under_3_5 | 4666 | 1.06 | 1.33 | 3.35 | 1.14 | 2.15 | sports-api |
| 188Bet | over_3_5 | 189 | 1.67 | 1.96 | 2.15 | 1.7976 | 2.1412 | sports-api |
| 188Bet | under_3_5 | 189 | 1.71 | 1.93 | 2.23 | 1.75 | 2.1024 | sports-api |
| 1xBet | over_3_5 | 6017 | 1.28 | 2.93 | 8.60 | 1.69 | 5.15 | sports-api |
| 1xBet | under_3_5 | 6017 | 1.01 | 1.33 | 3.30 | 1.10 | 2.25 | sports-api |
| 888Sport | over_3_5 | 3268 | 1.30 | 3.00 | 9.00 | 1.65 | 5.75 | sports-api |
| 888Sport | under_3_5 | 3268 | 1.06 | 1.33 | 3.40 | 1.13 | 2.15 | sports-api |
| Bet365 | over_3_5 | 5272 | 1.30 | 3.25 | 10.00 | 1.67 | 5.50 | sports-api |
| Bet365 | under_3_5 | 5272 | 1.06 | 1.33 | 3.50 | 1.14 | 2.20 | sports-api |
| **BetMGM** | **over_3_5** | **621** | **1.48** | **2.65** | **67.00** | **1.612** | **30.00** | **odds-api** |
| BetMGM | under_3_5 | 621 | 1.00 | 1.43 | 2.50 | 1.03 | 2.19 | odds-api |
| BetOnline.ag | over_3_5 | 152 | 1.48 | 2.65 | 4.65 | 1.57 | 4.6245 | odds-api |
| BetOnline.ag | under_3_5 | 152 | 1.20 | 1.49 | 2.70 | 1.2051 | 2.4245 | odds-api |
| BetRivers | over_3_5 | 1819 | 1.30 | 2.90 | 10.00 | 1.62 | 6.50 | odds-api |
| BetRivers | under_3_5 | 1819 | 1.04 | 1.40 | 3.30 | 1.10 | 2.2964 | odds-api |
| BetVictor | over_3_5 | 1788 | 1.25 | 2.75 | 8.50 | 1.56 | 5.75 | sports-api |
| BetVictor | under_3_5 | 1784 | 1.06 | 1.38 | 3.60 | 1.1266 | 2.2217 | sports-api |
| Betano | over_3_5 | 5192 | 1.32 | 3.10 | 10.25 | 1.70 | 5.20 | sports-api |
| Betano | under_3_5 | 5192 | 1.04 | 1.38 | 3.40 | 1.15 | 2.1818 | sports-api |
| Betfair | over_3_5 | 4785 | 1.29 | 3.25 | 8.00 | 1.62 | 5.50 | sports-api |
| Betfair | under_3_5 | 4785 | 1.05 | 1.30 | 3.70 | 1.11 | 2.20 | sports-api |
| Betway | over_3_5 | 135 | 1.55 | 3.00 | 5.50 | 1.685 | 4.75 | sports-api |
| Betway | under_3_5 | 135 | 1.11 | 1.36 | 2.40 | 1.15 | 2.1674 | sports-api |
| Bovada | over_3_5 | 1458 | 1.27 | 2.88 | 12.00 | 1.6457 | 4.35 | odds-api |
| Bovada | under_3_5 | 1458 | 1.03 | 1.43 | 3.85 | 1.22 | 2.2886 | odds-api |
| Bwin | over_3_5 | 135 | 1.57 | 2.95 | 5.75 | 1.6578 | 4.915 | sports-api |
| Bwin | under_3_5 | 135 | 1.12 | 1.36 | 2.25 | 1.1534 | 2.115 | sports-api |
| Caesars | over_3_5 | 501 | 1.42 | 2.65 | 6.50 | 1.56 | 5.00 | odds-api |
| Caesars | under_3_5 | 501 | 1.10 | 1.45 | 2.75 | 1.14 | 2.35 | odds-api |
| Coolbet | over_3_5 | 4271 | 1.32 | 3.00 | 11.00 | 1.647 | 6.00 | odds-api |
| Coolbet | under_3_5 | 4271 | 1.05 | 1.41 | 3.40 | 1.14 | 2.313 | odds-api |
| Dafabet | over_3_5 | 15 | 1.85 | 2.21 | 2.85 | 1.8514 | 2.836 | sports-api |
| Dafabet | under_3_5 | 15 | 1.45 | 1.65 | 1.98 | 1.4542 | 1.9786 | sports-api |
| DraftKings | over_3_5 | 174 | 1.57 | 2.05 | 2.35 | 1.6046 | 2.30 | odds-api |
| DraftKings | under_3_5 | 174 | 1.54 | 1.69 | 2.25 | 1.54 | 2.20 | odds-api |
| FanDuel | over_3_5 | 1148 | 1.37 | 2.70 | 8.40 | 1.64 | 5.653 | odds-api |
| FanDuel | under_3_5 | 1149 | 1.08 | 1.48 | 3.10 | 1.1448 | 2.28 | odds-api |
| Fonbet | over_3_5 | 135 | 1.53 | 2.95 | 6.20 | 1.685 | 5.364 | sports-api |
| Fonbet | under_3_5 | 135 | 1.12 | 1.40 | 2.40 | 1.1334 | 2.214 | sports-api |
| LiveScore Bet (EU) | over_3_5 | 2198 | 1.22 | 2.90 | 15.00 | 1.56 | 5.75 | odds-api |
| LiveScore Bet (EU) | under_3_5 | 2198 | 1.01 | 1.37 | 3.55 | 1.1097 | 2.3006 | odds-api |
| Marathonbet | over_3_5 | 5690 | 1.27 | 2.95 | 8.50 | 1.6589 | 5.1555 | sports-api |
| Marathonbet | under_3_5 | 5690 | 1.01 | 1.32 | 3.36 | 1.10 | 2.19 | sports-api |
| Pinnacle | over_3_5 | 8937 | 1.23 | 2.86 | 7.67 | 1.6136 | 5.14 | odds-api |
| Pinnacle | under_3_5 | 8937 | 1.11 | 1.44 | 4.17 | 1.18 | 2.3564 | odds-api |
| SBO | over_3_5 | 702 | 1.56 | 2.16 | 2.53 | 1.61 | 2.5098 | sports-api |
| SBO | under_3_5 | 702 | 1.51 | 1.73 | 2.49 | 1.55 | 2.3798 | sports-api |
| Superbet | over_3_5 | 2906 | 1.32 | 3.15 | 8.80 | 1.6805 | 5.595 | sports-api |
| Superbet | under_3_5 | 2906 | 1.07 | 1.35 | 3.45 | 1.13 | 2.169 | sports-api |
| Suprabets | over_3_5 | 4854 | 1.27 | 3.07 | 13.00 | 1.6553 | 6.40 | odds-api |
| Suprabets | under_3_5 | 4854 | 1.05 | 1.42 | 4.00 | 1.15 | 2.3141 | odds-api |
| Tipico | over_3_5 | 41 | 1.85 | 3.10 | 5.20 | 1.858 | 5.08 | sports-api |
| Tipico | under_3_5 | 41 | 1.13 | 1.33 | 1.87 | 1.138 | 1.862 | sports-api |
| Unibet | over_3_5 | 4881 | 1.32 | 3.05 | 9.50 | 1.68 | 5.42 | sports-api |
| Unibet | under_3_5 | 4881 | 1.05 | 1.34 | 3.40 | 1.12 | 2.15 | sports-api |
| WynnBET | over_3_5 | 992 | 1.43 | 3.00 | 7.80 | 1.6791 | 6.00 | odds-api |
| WynnBET | under_3_5 | 992 | 1.10 | 1.41 | 2.95 | 1.14 | 2.2609 | odds-api |

### Overround per bookmaker

Calcolato come `1/quota_over + 1/quota_under` sulle righe dove il
bookmaker ha **sia** l'Over **sia** l'Under 3.5 **nello stesso record
`Odds`** (stesso `id_odds_fk`, cioe' lo stesso snapshot/evento — non solo
lo stesso `id_fixture`+bookmaker, perche' un match puo' avere piu' righe
`Odds` prese in momenti diversi e accoppiare quote di snapshot diversi
produrrebbe overround falsati).

| Bookmaker | N righe (snapshot) | N fixture distinte | Overround mediano | Overround min | Overround max | N overround < 1.00 |
|---|---:|---:|---:|---:|---:|---:|
| 10Bet | 4666 | 4666 | 1.0617 | 1.0500 | 1.1250 | 0 |
| 188Bet | 189 | 189 | 1.0278 | 1.0204 | 1.0587 | 0 |
| 1xBet | 6017 | 5959 | 1.0874 | 1.0130 | 1.1101 | 0 |
| 888Sport | 3268 | 3268 | 1.0760 | 1.0506 | 1.1030 | 0 |
| Bet365 | 5272 | 5235 | 1.0633 | 1.0434 | 1.0929 | 0 |
| **BetMGM** | **621** | **621** | **1.0779** | **0.8639** | **1.0944** | **7** |
| BetOnline.ag | 152 | 152 | 1.0480 | 1.0382 | 1.0700 | 0 |
| BetRivers | 1819 | 1792 | 1.0560 | 1.0422 | 1.0924 | 0 |
| BetVictor | 1784 | 1768 | 1.0887 | 1.0516 | 1.0932 | 0 |
| Betano | 5192 | 5158 | 1.0474 | 1.0353 | 1.1143 | 0 |
| Betfair | 4785 | 4784 | 1.0779 | 1.0252 | 1.1587 | 0 |
| Betway | 135 | 135 | 1.0750 | 1.0530 | 1.0859 | 0 |
| Bovada | 1458 | 1435 | 1.0470 | 1.0412 | 1.0542 | 0 |
| Bwin | 135 | 135 | 1.0794 | 1.0668 | 1.0902 | 0 |
| Caesars | 501 | 489 | 1.0670 | 1.0530 | 1.0772 | 0 |
| Coolbet | 4271 | 3875 | 1.0410 | 1.0267 | 1.0842 | 0 |
| Dafabet | 15 | 15 | 1.0520 | 1.0393 | 1.0648 | 0 |
| DraftKings | 174 | 168 | 1.0812 | 1.0700 | 1.0929 | 0 |
| FanDuel | 1148 | 1118 | 1.0499 | 1.0417 | 1.0988 | 0 |
| Fonbet | 135 | 135 | 1.0625 | 1.0432 | 1.0946 | 0 |
| LiveScore Bet (EU) | 2198 | 1894 | 1.0750 | 1.0063 | 1.1398 | 0 |
| Marathonbet | 5690 | 5653 | 1.0880 | 1.0031 | 1.1098 | 0 |
| Pinnacle | 8937 | 8298 | 1.0381 | 1.0221 | 1.1255 | 0 |
| SBO | 702 | 700 | 1.0388 | 1.0256 | 1.0743 | 0 |
| Superbet | 2906 | 2906 | 1.0507 | 1.0455 | 1.1153 | 0 |
| Suprabets | 4854 | 4286 | 1.0262 | 1.0050 | 1.1195 | 0 |
| Tipico | 41 | 41 | 1.0753 | 1.0694 | 1.0799 | 0 |
| Unibet | 4881 | 4881 | 1.0668 | 1.0472 | 1.1329 | 0 |
| WynnBET | 992 | 977 | 1.0401 | 1.0352 | 1.0632 | 0 |

**Conferma**: il mercato e' sano ovunque tranne **BetMGM**, esattamente
come previsto. BetMGM ha 621 quote, Over max 67.00, e **7 righe** con
overround < 1.00 (impossibile per un bookmaker reale). Dettaglio delle 7
righe anomale:

| `id_odds_fk` | `id_fixture` | Quota Over 3.5 | Quota Under 3.5 | Overround |
|---|---:|---:|---:|---:|
| 66c9bfb1-b472-438d-9883-8675de2d2d46 | 1052628 | 26.00 | 1.18 | 0.8859 |
| 3ddbb084-05d6-47d8-9bc6-ddc1f2ebc558 | 1223650 | 61.00 | 1.18 | 0.8639 |
| e9c3d609-c19c-4187-b9ad-193d56d4add0 | 1223648 | 51.00 | 1.03 | 0.9905 |
| 420b1bc5-b55f-4bc0-a669-b0cdb8950db3 | 1223890 | 67.00 | 1.07 | 0.9495 |
| cc0c9e9b-55dd-4c1f-900a-ad597d22dc08 | 1223619 | 31.00 | 1.06 | 0.9757 |
| 0033e096-dc0f-4ed6-bd64-22ecc84354ae | 1223628 | 51.00 | 1.06 | 0.9630 |
| 43d032a4-69df-443a-aca3-98c20e57fc6f | 1223769 | 31.00 | 1.05 | 0.9846 |

Tutte le quote Over di BetMGM coinvolte sono estreme (26.00-67.00) contro
Under vicine alla parita' (1.03-1.18): un pattern compatibile con la stessa
contaminazione gia' vista su Under/Over 2.5 (quota di un'altra linea o di
un altro mercato finita nel bucket 3.5).

## Passo 3 — varianti di chiave (esito grezzo)

| Esito (chiave grezza) | N quote | `odds_from` |
|---|---:|---|
| `alternate_over_3_5` | 23.100 | odds-api |
| `alternate_under_3_5` | 23.101 | odds-api |
| `over 3.5` | 49.842 | sports-api |
| `under 3.5` | 49.838 | sports-api |

Nessuna variante inattesa: solo le due forme gia' note e gestite da
`_OUTCOME_ALIAS_PREFIXES` (`alternate_over_3_5`/`alternate_under_3_5` da
`odds-api`, `over 3.5`/`under 3.5` con lo spazio da `sports-api`). La somma
delle 4 righe (145.881) coincide con il totale delle quote estratte al
Passo 2: nessuna chiave persa dal parsing.

## Conclusione operativa

- Il dato grezzo e' sano: base rate 31.55%, 8.443 righe, nessun problema di
  parsing sulle chiavi quote.
- L'unico bookmaker contaminato su Under/Over 3.5 e' **BetMGM** (7 righe su
  621, overround min 0.8639), da escludere o correggere prima di
  procedere all'EDA (Passo 2 della procedura generale) — stessa cautela
  gia' presa su Under/Over 2.5.
