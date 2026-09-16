# Rifacimento mercato `goal_no_goal` — Passo 1 (dato grezzo dal DB)

Dati letti dal DB reale (stesso `DATABASE_URL` usato dall'app), via
`FilterMarketService().build_dataset(market="goal_no_goal", fill_missing=False)`
per il Passo 1, e query SQLAlchemy diretta `select(Match).options(selectinload(Match.odds))`
per i Passi 2-3, stesso metodo gia' usato su Under/Over 3.5 (vedi
`report_3_5_passo1.md`). Nessun numero in questo report e' stimato: tutto
proviene dall'esecuzione riportata. Sola lettura: nessuna quota modificata,
nessun modello riaddestrato.

## Passo 1 — estrazione grezza

Controllo bloccante superato: con `fill_missing=False` i valori mancanti
**non** sono azzerati (NaN totali > 0). Il codice in esecuzione e' quello
aggiornato.

| Metrica | Valore |
|---|---:|
| Shape dataset | 4.596 righe x 90 colonne |
| NaN totali | 30.211 |
| Base rate `y` (Goal/BTTS Si) | 53.59% |
| `prediction_at` min | 2023-05-03T16:00:00+00:00 |
| `prediction_at` max | 2026-09-15T19:30:00+00:00 |
| `odds_count` medio | 10.34 |
| Leghe distinte | 14 |
| Stagioni distinte | 5 (2022, 2023, 2024, 2025, 2026) |

CSV scritto in `scripts/analysis/_export/goal_no_goal_raw.csv` (4.66 MB,
sotto la soglia dei 50 MB).

Nota: rispetto a Under/Over 3.5 (8.443 righe) il dataset `goal_no_goal` ha
molte meno righe (4.596) a parita' di universo match (47.719 match FT a DB,
vedi Passo 2) — il bucket `Odds.goal_no_goal` e' popolato su una frazione
minore delle partite rispetto a `Odds.under_over_3_5`, non un problema di
parsing: `_build_row` scarta una riga se il bucket richiesto e' vuoto o
assente, e qui lo e' piu' spesso.

## Passo 2 — quadro quote per bookmaker

Righe di quota estratte dal bucket JSON `Odds.goal_no_goal` di tutti i
`Match` a DB (47.719 match totali): **6.581** righe `Odds` hanno il bucket
`goal_no_goal` non vuoto, per **66.440** quote singole totali (valore > 0).

### Statistiche per bookmaker x esito (normalizzato: `no_goal_` -> `no_goal`, `goal_` -> `goal`)

| Bookmaker | Esito | N | Min | Mediana | Max | P1 | P99 | `odds_from` prevalente |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 10Bet | goal | 34 | 1.45 | 1.770 | 2.50 | 1.4500 | 2.5000 | sports-api |
| 10Bet | no_goal | 4586 | 1.30 | 1.910 | 3.60 | 1.5000 | 2.8800 | sports-api |
| 188Bet | no_goal | 3944 | 1.25 | 1.930 | 3.70 | 1.4900 | 2.9200 | sports-api |
| 1xBet | goal | 1168 | 1.25 | 1.830 | 2.52 | 1.3734 | 2.3300 | odds-api |
| 1xBet | no_goal | 6517 | 1.21 | 1.940 | 3.74 | 1.5100 | 2.9500 | sports-api |
| 888Sport | goal | 20 | 1.44 | 1.530 | 2.70 | 1.4400 | 2.5860 | sports-api |
| 888Sport | no_goal | 3264 | 1.30 | 1.950 | 3.60 | 1.5300 | 2.9000 | sports-api |
| Bet365 | goal | 337 | 1.30 | 1.800 | 2.50 | 1.3600 | 2.2500 | sports-api |
| Bet365 | no_goal | 5300 | 1.30 | 1.910 | 3.50 | 1.5000 | 2.7500 | sports-api |
| BetMGM | goal | 824 | 1.44 | 1.870 | 4.20 | 1.5400 | 2.4385 | odds-api |
| BetMGM | no_goal | 824 | 1.18 | 1.800 | 2.55 | 1.4800 | 2.3000 | odds-api |
| BetVictor | goal | 337 | 1.25 | 1.700 | 2.45 | 1.3000 | 2.2500 | sports-api |
| BetVictor | no_goal | 1829 | 1.33 | 2.000 | 3.75 | 1.5000 | 3.2000 | sports-api |
| Betano | goal | 283 | 1.28 | 1.720 | 2.45 | 1.3164 | 2.3344 | sports-api |
| Betano | no_goal | 5194 | 1.27 | 1.950 | 3.50 | 1.5200 | 2.9000 | sports-api |
| Betclic | goal | 50 | 1.52 | 1.865 | 2.30 | 1.5690 | 2.3000 | odds-api |
| Betclic | no_goal | 50 | 1.55 | 1.850 | 2.35 | 1.5647 | 2.2520 | odds-api |
| Betfair | goal | 239 | 1.25 | 1.730 | 2.50 | 1.3300 | 2.3800 | sports-api |
| Betfair | no_goal | 4879 | 1.29 | 1.910 | 3.80 | 1.5000 | 3.0000 | sports-api |
| Betway | no_goal | 135 | 1.50 | 1.910 | 2.80 | 1.5568 | 2.7388 | sports-api |
| Bovada | goal | 4 | 1.74 | 1.815 | 1.93 | 1.7412 | 1.9276 | odds-api |
| Bovada | no_goal | 4 | 1.82 | 1.935 | 2.05 | 1.8221 | 2.0479 | odds-api |
| Bwin | no_goal | 128 | 1.47 | 1.850 | 2.75 | 1.5154 | 2.3700 | sports-api |
| Caesars | goal | 282 | 1.53 | 1.950 | 2.50 | 1.5843 | 2.3595 | odds-api |
| Caesars | no_goal | 282 | 1.50 | 1.800 | 2.40 | 1.5543 | 2.3095 | odds-api |
| Dafabet | no_goal | 89 | 1.54 | 1.990 | 2.90 | 1.5840 | 2.7680 | sports-api |
| FanDuel | goal | 1 | 1.93 | 1.930 | 1.93 | 1.9300 | 1.9300 | odds-api |
| FanDuel | no_goal | 1 | 1.85 | 1.850 | 1.85 | 1.8500 | 1.8500 | odds-api |
| Fonbet | no_goal | 135 | 1.48 | 1.970 | 2.75 | 1.5170 | 2.6500 | sports-api |
| LiveScore Bet (EU) | goal | 339 | 1.50 | 1.850 | 4.20 | 1.5400 | 2.3620 | odds-api |
| LiveScore Bet (EU) | no_goal | 339 | 1.19 | 1.860 | 2.55 | 1.5376 | 2.4000 | odds-api |
| Marathonbet | goal | 346 | 1.25 | 1.715 | 2.46 | 1.2980 | 2.2555 | sports-api |
| Marathonbet | no_goal | 5695 | 1.30 | 1.920 | 3.64 | 1.4900 | 2.9300 | sports-api |
| Pinnacle | no_goal | 3648 | 1.36 | 1.980 | 3.72 | 1.5200 | 2.9453 | sports-api |
| Superbet | goal | 34 | 1.44 | 1.810 | 2.55 | 1.4565 | 2.5500 | sports-api |
| Superbet | no_goal | 2906 | 1.33 | 1.920 | 3.45 | 1.4900 | 2.8675 | sports-api |
| Tipico | goal | 1 | 3.60 | 3.600 | 3.60 | 3.6000 | 3.6000 | odds-api |
| Tipico | no_goal | 42 | 1.25 | 1.950 | 2.50 | 1.3935 | 2.5000 | sports-api |
| Unibet | goal | 28 | 1.42 | 1.680 | 2.60 | 1.4281 | 2.5244 | sports-api |
| Unibet | no_goal | 4581 | 1.38 | 1.940 | 3.55 | 1.4800 | 2.9500 | sports-api |
| William Hill | goal | 1175 | 1.30 | 1.850 | 2.60 | 1.4000 | 2.3000 | odds-api |
| William Hill | no_goal | 6358 | 1.33 | 1.910 | 3.60 | 1.5300 | 2.9000 | sports-api |
| William Hill (US) | goal | 104 | 1.48 | 1.870 | 2.60 | 1.5312 | 2.3000 | odds-api |
| William Hill (US) | no_goal | 104 | 1.50 | 1.870 | 2.65 | 1.5900 | 2.3985 | odds-api |

Nessun massimo estremo (il piu' alto e' 4.20, BetMGM/LiveScore Bet lato
`goal`): a differenza di Under/Over 3.5, dove BetMGM arrivava a 67.00 sul
lato Over, qui non compare nessun valore fuori scala.

Molti bookmaker (188Bet, Betway, Bwin, Dafabet, Fonbet, Pinnacle) hanno solo
il lato `no_goal` popolato, mai `goal`: non e' un'anomalia di parsing (le
chiavi sono lette correttamente, vedi Passo 3) ma copertura mancante alla
fonte per quel lato — quelle righe non entrano nel calcolo dell'overround
sotto, che richiede ENTRAMBI i lati nello stesso snapshot.

### Overround per bookmaker

Calcolato come `1/quota_goal + 1/quota_no_goal` sulle righe dove il
bookmaker ha **sia** `goal` **sia** `no_goal` **nello stesso record
`Odds`** (stesso `id_odds_fk`, cioe' lo stesso snapshot/evento — stessa
cautela presa su Under/Over 3.5, per non accoppiare quote di snapshot
diversi).

| Bookmaker | N righe (snapshot) | N fixture distinte | Overround mediano | Overround min | Overround max | N overround < 1.00 | N overround > 1.15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10Bet | 34 | 34 | 1.0718 | 1.0644 | 1.0997 | 0 | 0 |
| 1xBet | 1168 | 1165 | 1.0542 | 1.0497 | 1.0928 | 0 | 0 |
| 888Sport | 20 | 20 | 1.0657 | 1.0606 | 1.1057 | 0 | 0 |
| Bet365 | 337 | 337 | 1.0738 | 1.0471 | 1.0929 | 0 | 0 |
| BetMGM | 824 | 820 | 1.0885 | 1.0747 | 1.1054 | 0 | 0 |
| BetVictor | 337 | 337 | 1.0842 | 1.0779 | 1.0945 | 0 | 0 |
| Betano | 283 | 283 | 1.0713 | 1.0652 | 1.1124 | 0 | 0 |
| Betclic | 50 | 50 | 1.0805 | 1.0582 | 1.0842 | 0 | 0 |
| Betfair | 239 | 239 | 1.0700 | 1.0556 | 1.1011 | 0 | 0 |
| Bovada | 4 | 0 | 1.0672 | 1.0625 | 1.0696 | 0 | 0 |
| Caesars | 282 | 282 | 1.0650 | 1.0533 | 1.0703 | 0 | 0 |
| FanDuel | 1 | 1 | 1.0587 | 1.0587 | 1.0587 | 0 | 0 |
| LiveScore Bet (EU) | 339 | 339 | 1.0646 | 1.0528 | 1.1245 | 0 | 0 |
| Marathonbet | 346 | 346 | 1.0859 | 1.0795 | 1.1018 | 0 | 0 |
| Superbet | 34 | 34 | 1.0622 | 1.0577 | 1.0834 | 0 | 0 |
| Tipico | 1 | 1 | 1.0778 | 1.0778 | 1.0778 | 0 | 0 |
| Unibet | 28 | 28 | 1.0668 | 1.0600 | 1.0990 | 0 | 0 |
| William Hill | 1175 | 1171 | 1.0760 | 1.0405 | 1.1120 | 0 | 0 |
| William Hill (US) | 104 | 104 | 1.0650 | 1.0513 | 1.0703 | 0 | 0 |

(`N fixture distinte` per Bovada risulta 0 perche' tutti e 4 i suoi snapshot
hanno `Match.id_fixture` NULL a DB — 15 righe su 5.606 nel totale
overround, su 4 bookmaker (William Hill 4, BetMGM 4, Bovada 4, 1xBet 3),
`nunique()` non conta i NaN. Non incide sull'overround, calcolato per
snapshot (`id_odds_fk`) non per fixture; segnalato solo come osservazione
sui dati, nessuna correzione applicata.)

**Nessuna anomalia**: 0 righe con overround < 1.00 e 0 righe con overround
> 1.15 su tutti i 19 bookmaker che hanno almeno uno snapshot con entrambi i
lati. Il mercato `goal_no_goal` e' sano — **non riproduce** il bug visto su
Under/Over 2.5/3.5 (quota di un'altra linea finita nel bucket sbagliato di
`totals`).

## Passo 3 — varianti di chiave (esito grezzo)

| Esito (chiave grezza) | N quote | `odds_from` |
|---|---:|---|
| `no_goal_` | 57.567 | sports-api |
| `goal` | 3.267 | odds-api |
| `no_goal` | 3.267 | odds-api |
| `goal_` | 2.339 | sports-api |

Nessuna variante inattesa: solo le due forme gia' documentate nel codice
(`FilterMarketService._split_outcome_and_bookmaker`, docstring) — `goal_`/
`no_goal_` con l'underscore finale da `sports-api` (l'esito grezzo del
provider include gia' l'underscore prima del nome bookmaker) e `goal`/
`no_goal` senza da `odds-api` (`get_btts` in `df_odds_service.py` produce
`f'goal_{title}'`/`f'no_goal_{title}'`, un solo underscore, consumato dallo
split). Entrambe normalizzate correttamente a `goal`/`no_goal` da
`_normalize_outcome_name` (che fa `strip("_")`). La somma delle 4 righe
(66.440) coincide con il totale delle quote estratte al Passo 2: nessuna
chiave persa dal parsing.

A differenza di Under/Over 3.5 non compare nessuna variante `alternate_*`:
coerente con l'aspettativa dell'operatore — `goal_no_goal`/BTTS non ha
linee multiple (non e' un mercato "a soglia" come i Totali), quindi
`odds-api` non ha bisogno di un bucket "alternate" per punti diversi dalla
linea principale. Confermato sul dato reale a DB: la causa strutturale del
bug 2.5/3.5 (piu' `point` diversi mappati sulla stessa chiave piatta) non
si applica qui.

## Conclusione operativa

- Il dato grezzo e' sano: base rate 53.59%, 4.596 righe, nessun problema di
  parsing sulle chiavi quote.
- Overround sano su **tutti** i 19 bookmaker osservati (nessuno sotto 1.00,
  nessuno sopra 1.15): a differenza di Under/Over 2.5/3.5, qui non emerge
  nessun bookmaker contaminato (BetMGM compreso, che su 3.5 era l'unico
  problematico — qui il suo massimo e' 4.20, non 67.00).
- Percorso `get_btts` confermato diverso da quello di `totals` (nessun
  filtro per `point`/linea, perche' BTTS non ne ha) e, coerentemente,
  esente dal bug visto sui mercati Under/Over: nessuna correzione
  necessaria prima di procedere alla EDA (Passo 2 della procedura
  generale).
