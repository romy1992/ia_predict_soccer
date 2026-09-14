# Riscrittura quote Under/Over 2.5 (odds-api) — report esecuzione

Data esecuzione: 2026-09-14
Branch: `feature/soccer-oracle-v2`
Commit di partenza: `35e2df5` (fix `df_odds_service.py` + script `fix_odds_2_5_lines.py`)

## Passo 1 — Simulazione

Comando: `python scripts/maintenance/fix_odds_2_5_lines.py` (senza `--apply`)

```
eventi nel CSV          : 15,499
quote Over lette prima  : 77,316
di linea SBAGLIATA      : 24,880  (32.18%)
eventi senza 2.5 vero   : 1,453

partite con id_events a DB: 15,053

righe_modificate               13,263
chiavi_rimosse                 85,218
chiavi_riscritte               99,040
righe_rimaste_vuote               470
righe_gia_corrette              1,022
righe_senza_bucket                768
partite_senza_payload               0
righe_non_odds_api_saltate          0
```

Controlli attesi: tutti confermati.
- quote Over lette prima = 77.316 ✓
- di linea sbagliata = 24.880 (32,18%) ✓
- righe_modificate = 13.263, nel range 10.000-15.000 ✓
- partite_senza_payload = 0 → il join su `Match.id_events` funziona correttamente ✓

## Passo 2 — Applicazione

Comando: `python scripts/maintenance/fix_odds_2_5_lines.py --apply`

**Primo tentativo (fallito).** Dopo 34 minuti di esecuzione (commit unico per tutte le 13.263 righe),
la connessione al Postgres remoto (Railway) si è chiusa a metà dell'`UPDATE` massivo:

```
psycopg2.OperationalError: server closed the connection unexpectedly
```

Verifica di sola lettura eseguita subito dopo: il valore a DB per una riga campione
(`id_odds_fk = 00032d6c-5928-44b9-867e-81c661887b26`) coincideva esattamente col "prima"
registrato nel backup, non con il valore nuovo che lo script stava scrivendo. Postgres ha
fatto rollback pulito della transazione: **nessuna scrittura parziale**, DB rimasto intatto.

**Causa**: un'unica transazione con ~13.263 `UPDATE` eseguiti riga per riga dall'ORM (un
round-trip di rete a testa) su una connessione remota è fragile — dopo mezz'ora la connessione
è caduta, probabilmente per un timeout lato Railway.

**Modifica allo script**, autorizzata esplicitamente dall'operatore in deroga alla regola di
non toccare `fix_odds_2_5_lines.py`:
- commit a lotti da 500 righe invece di un'unica transazione (`BATCH_COMMIT = 500`), con un
  commit finale per il resto;
- scrittura del backup incrementale (una riga JSONL alla volta) invece che in un unico blocco
  a fine ciclo, cosi' il backup non dipende dal completamento del commit;
- `sessione.expire_on_commit = False` sulla sessione locale dello script, per evitare che ogni
  commit parziale invalidasse tutti gli oggetti ORM gia' caricati (che avrebbe causato una
  query di refresh aggiuntiva per ogni accesso successivo, peggiorando il problema che si
  voleva risolvere).

Ri-simulazione dopo la patch: numeri identici a quelli del Passo 1 (nessuna modifica alla
logica di calcolo).

**Secondo tentativo (riuscito).** Il comando `--apply` con commit a lotti ha completato tutte
le 13.263 righe senza errori, con log di avanzamento ogni 500 righe (8:44 UTC → 9:43 UTC, ~59
minuti totali, piu' lento del previsto ma senza interruzioni):

```
righe_modificate               13,263
chiavi_rimosse                 85,218
chiavi_riscritte               99,040
righe_rimaste_vuote               470
righe_gia_corrette              1,022
righe_senza_bucket                768
partite_senza_payload               0
righe_non_odds_api_saltate          0

backup del PRIMA salvato in scripts\maintenance\_backup_under_over_2_5.jsonl (13,263 righe)
commit (a lotti da 500) completato alle 2026-09-14T09:43:02.623309+00:00
```

Backup verificato: `scripts/maintenance/_backup_under_over_2_5.jsonl` esiste, 13.263 righe,
6,4 MB (sotto la soglia di 50 MB → committato insieme al resto).

## Passo 3 — Verifica sul dato riscritto

Query diretta sul DB, righe `Odds` con `odds_from='odds-api'`, chiavi `over_2.5_*` /
`under_2.5_*` (formato underscore-punto, escluse `alternate_*` e quelle con lo spazio):

```
righe odds-api totali                         : 15,053
quote over_2.5_*/under_2.5_* totali           : 103,110
righe con bucket under_over_2_5 vuoto         : 1,238
valore massimo di un over_2.5_*               : 18.0  (chiave over_2.5_Unibet)
```

**Controllo che non torna**: il valore massimo atteso era "sotto ~6" (prima era 21.00), ma il
massimo osservato e' 18.0. Ho verificato che **non e' un residuo del bug**: il record e'
Reggiana-Parma del 2024-05-10, bookmaker Unibet, e il payload grezzo (CSV) conferma
`{'name': 'Over', 'price': 18.0, 'point': 2.5}` abbinato a `{'name': 'Under', 'price': 1.01,
'point': 2.5}` — una quota reale a linea 2.5 esatta, quasi certamente una rilevazione live/
in-play a partita gia' indirizzata (Under a 1.01). Lo script ha estratto correttamente la
linea giusta; l'aspettativa "sotto ~6" era una stima approssimativa dell'operatore che non
teneva conto di questi outlier live legittimi.

Per contestualizzare l'entità del fenomeno, distribuzione completa dei valori `over_2.5_*`:

```
quote over_2.5_* totali : 51,555
sopra 6                 : 16   (0.03%)
sopra 10                : 12
mediana                 : 1.86
top 10 valori più alti  : [15.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.5, 17.5, 18.0, 18.0]
```

16 quote su 51.555 (0,03%) superano 6 — coerente con rari snapshot live genuini, non con un
problema sistemico residuo.

## Passo 4 — Ri-estrazione del dataset grezzo

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
df = FilterMarketService().build_dataset(market="under_over_2_5", fill_missing=False)
df.to_csv("scripts/analysis/_export/under_over_2_5_raw.csv", index=False)
```

Controllo bloccante: NaN totali = 9.882 (≠ 0) → confermato che sta girando il codice
aggiornato, non quello vecchio.

| Metrica | Prima della correzione | Dopo la correzione |
|---|---|---|
| Shape | 15.340 righe × 90 colonne | 15.023 righe × 90 colonne |
| NaN totali | 9.990 | 9.882 |
| Base rate y | 0,5288 | 0,5267 |
| Periodo (prediction_at) | 2020-06-11 → 2026-09-13 | 2020-06-11T20:00:00+00:00 → 2026-09-13T19:30:00+00:00 |
| odds_count medio | 12,97 | 14,02 |

317 righe in meno nel dataset: coerente con le 470 righe Odds rimaste completamente vuote dopo
la rimozione delle quote di linea sbagliata (alcune combinazioni match/mercato perdono ogni
quota valida quando nessun bookmaker copriva davvero il 2.5).

## File prodotti

- `report_riscrittura_2_5.md` (questo report)
- `scripts/analysis/_export/under_over_2_5_raw.csv` (rigenerato, sovrascrive il vecchio con le
  quote sbagliate)
- `scripts/maintenance/_backup_under_over_2_5.jsonl` (backup del "prima", 13.263 righe, 6,4 MB —
  committato, sotto la soglia dei 50 MB)
- `scripts/maintenance/fix_odds_2_5_lines.py` (patch commit-a-lotti, autorizzata dall'operatore)

## Filtro quote live

Data esecuzione: 2026-09-14
Commit di partenza: `11f1110` (aggiunge la flag `--scarta-live` a `fix_odds_2_5_lines.py`)

Contesto: dopo la riscrittura sulla linea corretta (sezione precedente), restavano 15-16 quote
`over_2.5_*` sopra 6.00. Verificate: non un residuo del bug della linea, ma prezzi rilevati DOPO
il calcio d'inizio (`last_update` successivo a `commence_time`, in un caso 114 minuti dopo) — cioè
quote live, informazione dal futuro rispetto al momento della previsione pre-match. Stima
dell'operatore: 45 quote su 52.436, 28 fixture.

### Passo 1 — Simulazione

Comando: `python scripts/maintenance/fix_odds_2_5_lines.py --scarta-live` (senza `--apply`)

```
eventi nel CSV          : 15,499
quote Over lette prima  : 77,102
di linea SBAGLIATA      : 24,711  (32.05%)
eventi senza 2.5 vero   : 1,479
quote live scartate     : 422

partite con id_events a DB: 15,053

righe_modificate                   26
chiavi_rimosse                     92
chiavi_riscritte                    8
righe_rimaste_vuote                 9
righe_gia_corrette             13,789
righe_senza_bucket              1,238
partite_senza_payload               0
righe_non_odds_api_saltate          0
```

Controlli:
- `righe_modificate` = 26, ordine delle decine, vicino alle 28 fixture attese, ben sotto la
  soglia di stop (1.000) ✓
- `righe_gia_corrette` = 13.789 → conferma che il DB era già corretto sulla linea e che questa
  esecuzione tocca solo le fixture con quote live ✓
- `quote live scartate` = 422, **non** vicino ai ~45 attesi dall'operatore (scarto ~9,4×).

Letto il codice (`carica_quote_corrette`, righe 104-116) per capire lo scarto, senza modificarlo:
il contatore `quote_live_scartate` si incrementa per ogni voce bookmaker con `last_update`
successivo al calcio d'inizio, **prima** di controllare se quel bookmaker quotava la linea 2.5 —
quindi conta le quote live su qualunque linea (2.25, 2.75, 3.5, ecc.), non solo quelle che
sarebbero finite come `over_2.5_*`/`under_2.5_*` a DB. I 45 stimati dall'operatore erano invece
specifici delle quote già sulla linea 2.5. Le due cifre misurano popolazioni diverse; la metrica
che conta per la sicurezza della scrittura è `righe_modificate`, coerente con l'atteso. Nessun
problema nello script: comportamento spiegato dal codice, non un bug.

Simulazione coerente → si procede con l'applicazione.

### Passo 2 — Applicazione

Prima dell'`--apply`, copiato il backup esistente (che conteneva il "prima" della riscrittura
grande, sezione precedente — unico modo per tornare indietro da quella) in un file separato:

```
scripts/maintenance/_backup_under_over_2_5.jsonl                    → 6.637.345 byte (invariato, verificato)
scripts/maintenance/_backup_under_over_2_5_riscrittura_linee.jsonl  → 6.637.345 byte (copia, verificata identica)
```

Comando: `python scripts/maintenance/fix_odds_2_5_lines.py --scarta-live --apply`

```
APPLICATO
righe_modificate                   26
chiavi_rimosse                     92
chiavi_riscritte                    8
righe_rimaste_vuote                 9
righe_gia_corrette             13,789
righe_senza_bucket              1,238
partite_senza_payload               0
righe_non_odds_api_saltate          0

backup del PRIMA salvato in scripts\maintenance\_backup_under_over_2_5.jsonl (26 righe)
commit (a lotti da 500) completato alle 2026-09-14T12:36:53.674847+00:00
```

Nuovo backup (le 26 righe toccate da questa esecuzione, sovrascrive il vecchio): 11.409 byte.

### Passo 3 — Verifica sul dato riscritto

Query diretta sul DB, righe `Odds` con `odds_from='odds-api'`, sole chiavi `over_2.5_*`
(formato underscore-punto, escluse `alternate_*` e quelle con lo spazio):

```
quote over_2.5_* totali       : 51,513
valore massimo over_2.5_*     : 3.25
quote over_2.5_* >= 6.00      : 0
```

Massimo sceso da 18.00 a 3.25, zero quote residue sopra 6.00 ✓

### Passo 4 — Ri-estrazione del dataset grezzo

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
df = FilterMarketService().build_dataset(market="under_over_2_5", fill_missing=False)
df.to_csv("scripts/analysis/_export/under_over_2_5_raw.csv", index=False)
```

Controllo bloccante: NaN totali = 9.882 (≠ 0) → confermato che sta girando il codice aggiornato.

| Metrica | Dopo riscrittura linee (sezione precedente) | Dopo filtro live | Atteso |
|---|---|---|---|
| Shape | 15.023 × 90 | 15.014 × 90 | 15.023 × 90 |
| NaN totali | 9.882 | 9.882 | 9.882 |
| Base rate y | 0,5267 | 0,526842 | 0,5267 |
| Periodo (prediction_at) | 2020-06-11T20:00:00+00:00 → 2026-09-13T19:30:00+00:00 | 2020-06-11T20:00:00+00:00 → 2026-09-13T19:30:00+00:00 | — |
| odds_count medio | 14,02 | 14,027041 | 14,02 |

9 righe in meno rispetto alla sezione precedente, coerente con `righe_rimaste_vuote = 9` del
Passo 2 (le fixture le cui uniche quote 2.5 erano live restano senza bucket valido). Tutti gli
altri valori sono in linea con l'atteso.

### File prodotti in questa fase

- `report_riscrittura_2_5.md` (questa sezione)
- `scripts/analysis/_export/under_over_2_5_raw.csv` (rigenerato)
- `scripts/maintenance/_backup_under_over_2_5.jsonl` (backup delle 26 righe toccate ora, 11.409 byte)
- `scripts/maintenance/_backup_under_over_2_5_riscrittura_linee.jsonl` (copia del backup della
  riscrittura precedente, 6.637.345 byte — sotto la soglia di 50 MB, committata)

## Non fatto

Nessun retraining eseguito, come da istruzioni: il training lo esegue l'operatore
separatamente.
