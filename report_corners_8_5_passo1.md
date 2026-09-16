# Mercato `corners_line_8_5` — Passo 1 (dato grezzo dal DB)

Dati letti dal DB reale (stesso `DATABASE_URL` usato dall'app). Sola
lettura: nessuna quota modificata, nessun modello riaddestrato, nessuno
script modificato. Nessun numero in questo report e' stimato: tutto
proviene dall'esecuzione riportata.

## Passo 1 — estrazione grezza: BLOCCATA

Il comando richiesto

```python
FilterMarketService().build_dataset(market="corners_line_8_5", fill_missing=False)
```

fallisce con:

```
ValueError: Mercato non supportato: corners_line_8_5
```

Causa (verificata leggendo il codice, non modificato):
`FilterMarketService.build_dataset` (righe 467-498) accetta solo mercati
in `SUPPORTED_MARKETS` (`h2h`, `under_over_1_5..4_5`, `goal_no_goal`,
`corners`, `cards`, `dc`). `corners_line_8_5` appartiene invece a
`LINE_MARKETS` (righe 34-43, mercati a linea configurabile
8.5/9.5/10.5/11.5), che `build_dataset` non controlla mai — `build_dataset`
non e' stato esteso quando sono stati introdotti i mercati a linea.

Esiste un percorso dedicato per i mercati a linea,
`build_corners_frame_from_records` (`src/ml/markets/corners/
corners_market.py`, righe 149-183), che costruisce feature + target reale
per TUTTE le linee (8.5/9.5/10.5/11.5) in un colpo solo — ma fa
incondizionatamente `.fillna(0)` (riga 179): non esiste, ad oggi, alcun
parametro equivalente a `fill_missing=False` per i mercati a linea. Non e'
quindi possibile estrarre il dataset GREZZO (con i NaN preservati) per
`corners_line_8_5` senza scrivere codice nuovo — cosa esplicitamente
esclusa da questo incarico ("non modificare script").

**Nessun CSV prodotto per il Passo 1** (nessun file
`corners_line_8_5_raw.csv` in `scripts/analysis/_export/`): non c'e' un
dataset grezzo da esportare finche' questo gap non viene colmato con una
modifica di codice deliberata (fuori scope qui).

## Passo 2 — quadro quote per bookmaker (query diretta Match+Odds)

Query SQLAlchemy diretta `select(Match).options(selectinload(Match.odds))`,
poi filtro sul bucket JSON `Odds.corners` (mercato "corners" pooled, TUTTE
le linee/esiti/bookmaker in un unico dict — non separato per linea come
`under_over_*`), isolando le chiavi la cui linea estratta (
`FilterMarketService._extract_line_from_odds_key`) e' 8.5 (tolleranza
0.01).

| Metrica | Valore |
|---|---:|
| Match totali a DB | 47.719 |
| Record `Odds` totali | 20.822 |
| Record `Odds` con bucket `corners` non vuoto | 12.053 |
| Quote totali con linea == 8.5 (grezze) | 107.459 |
| Bookmaker con Over+Under 8.5 nello stesso `id_odds_fk` | 19 |
| Righe (snapshot) overround accoppiate Over+Under | 53.729 |

### Varianti di chiave trovate (esito grezzo, prima della normalizzazione)

| Esito (chiave grezza) | N quote | `odds_from` |
|---|---:|---|
| `over 8.5` | 27.995 | sports-api |
| `under 8.5` | 27.995 | sports-api |
| `corner_Over_8.5` | 25.735 | odds-api |
| `corner_Under_8.5` | 25.734 | odds-api |

Solo due forme, entrambe gia' gestite da `_OUTCOME_ALIAS_PREFIXES`
(`corner_...` normalizzato a `over_8_5`/`under_8_5` come le altre) e da
`_extract_line_from_odds_key`. La somma (107.459) coincide esattamente col
totale di quote-linea-8.5 estratte sopra: nessuna chiave persa dal
parsing.

### Overround per bookmaker (`1/quota_over + 1/quota_under`, stesso `id_odds_fk`)

| Bookmaker | N righe (snapshot) | N fixture distinte | Overround mediano | Min | Max | N overround < 1.00 |
|---|---:|---:|---:|---:|---:|---:|
| 10Bet | 650 | 650 | 1.0778 | 1.0669 | 1.1186 | 0 |
| 1xBet | 5.477 | 5.477 | 1.0810 | 1.0491 | 1.1390 | 0 |
| Bet365 | 1.553 | 1.553 | 1.0750 | 1.0526 | 1.0929 | 0 |
| **BetMGM** | **6.411** | **5.363** | **1.0915** | **1.0915** | **1.0915** | 0 |
| **BetRivers** | **6.411** | **5.363** | **1.0922** | **1.0922** | **1.0922** | 0 |
| Betano | 4.844 | 4.844 | 1.0711 | 1.0456 | 1.0846 | 0 |
| Betfair | 1.675 | 1.675 | 1.0818 | 1.0542 | 1.1310 | 0 |
| Betway | 80 | 80 | 1.0811 | 1.0718 | 1.0859 | 0 |
| **Bovada** | **6.411** | **5.363** | **1.0675** | **1.0675** | **1.0675** | 0 |
| Bwin | 73 | 73 | 1.0916 | 1.0754 | 1.1051 | 0 |
| Dafabet | 21 | 21 | 1.0720 | 1.0678 | 1.0814 | 0 |
| DraftKings | 72 | 72 | 1.0794 | 1.0741 | 1.0885 | 0 |
| FanDuel | 18 | 18 | 1.0723 | 1.0665 | 1.0768 | 0 |
| Fonbet | 83 | 83 | 1.0811 | 1.0779 | 1.0842 | 0 |
| Marathonbet | 2.803 | 2.803 | 1.0984 | 1.0795 | 1.1091 | 0 |
| Pinnacle | 10.546 | 9.498 | 1.0683 | 1.0363 | 1.1142 | 0 |
| Superbet | 2.732 | 2.732 | 1.0778 | 1.0656 | 1.0846 | 0 |
| Tipico | 24 | 24 | 1.0819 | 1.0795 | 1.0833 | 0 |
| Unibet | 3.845 | 3.845 | 1.0875 | 1.0747 | 1.1039 | 0 |

**Nessuna riga con overround < 1.00, nessuna riga con rapporto
max/mean quota > 3** (stesso criterio usato per `under_over_3_5`): a
differenza di quel mercato (dove BetMGM era contaminato), qui il criterio
non segnala nulla.

### Anomalia trovata (non catturata dal criterio overround<1 / ratio>3)

**BetMGM, BetRivers e Bovada hanno overround IDENTICO (min = mediana =
max) su migliaia di righe distinte** (6.411 snapshot, 5.363 fixture
diverse ciascuno). Verificato campionando le quote grezze: per questi tre
bookmaker la coppia Over/Under 8.5 e' **letteralmente identica, cifra per
cifra, su fixture completamente diverse**:

| Bookmaker | Quota Over 8.5 | Quota Under 8.5 | Esempio fixture (`id_fixture`) |
|---|---:|---:|---|
| BetMGM | 1.57 | 2.20 | 1299212, 1048883, 872518, 1044915, 868320, ... |
| BetRivers | 1.50 | 2.35 | (stesse fixture di sopra) |
| Bovada | 1.59 | 2.28 | (stesse fixture di sopra) |

Queste tre quote sono costanti su TUTTE le partite campionate (8/8 su
ciascun bookmaker, fixture id diversi, `id_odds_fk` diversi). Non e' un
bookmaker reale che quota sempre uguale per caso: e' un pattern
compatibile con una quota PLACEHOLDER/sintetica finita nel bucket
`corners`, non con dati di mercato reali — il criterio overround da solo
non la cattura perche' il margine risultante (1.0915/1.0922/1.0675) e'
plausibile, solo che e' sempre lo stesso numero.

## Conclusione operativa

- **Passo 1 bloccato**: `build_dataset` non supporta `corners_line_8_5`
  (ne' alcun mercato in `LINE_MARKETS`) — serve una modifica di codice per
  estrarre un dataset grezzo con NaN preservati per questo mercato. Nessun
  CSV prodotto.
- **Passo 2 completato**: 19 bookmaker con quote Over+Under 8.5
  accoppiate, 107.459 quote grezze, 2 varianti di chiave (entrambe gia'
  gestite correttamente dal parsing esistente).
- Il criterio overround<1/ratio>3 non segnala nulla, ma un controllo
  supplementare (valori costanti su fixture diverse) rivela che
  **BetMGM, BetRivers e Bovada** hanno quote Over/Under 8.5 fisse e
  identiche indipendentemente dalla partita: dati da escludere/investigare
  prima di qualunque training su questa linea, esattamente come BetMGM lo
  era (per un motivo diverso: overround<1, non costanza) su
  `under_over_3_5`.
