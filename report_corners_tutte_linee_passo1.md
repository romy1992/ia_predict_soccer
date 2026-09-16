# Mercati `corners_line_{8_5,9_5,10_5,11_5}` — Passo 1 (dato grezzo dal DB)

Dati letti dal DB reale (stesso `DATABASE_URL` usato dall'app), sola
lettura: nessuna quota modificata, nessun modello riaddestrato.

## Bug corretto

`git pull` a `1b6dd50` ("build_dataset supporta i mercati a linea"), ma
l'estrazione falliva ancora (vedi `report_corners_tutte_linee_passo1.md`
precedente, versione bloccata): `_build_line_dataset`
(`filter_market_service.py`) cercava la colonna `y_<linea>` (es. `y_8_5`)
mentre il frame prodotto da `build_corners_frame_from_records`
(`corners_market.py`, `_line_label()`) la nomina `y_line_<linea>` (es.
`y_line_8_5`) — mismatch di naming introdotto dal fix `1b6dd50` stesso.

Corretto in `filter_market_service.py:536`:
`colonna_y = f"y_{linea_label}"` → `colonna_y = f"y_line_{linea_label}"`
(+ docstring aggiornata). Nessun'altra modifica.

## Passo 1 — estrazione grezza: OK per tutte e 4 le linee

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
for linea in ("corners_line_8_5", "corners_line_9_5", "corners_line_10_5", "corners_line_11_5"):
    df = FilterMarketService().build_dataset(market=linea, fill_missing=False)
    df.to_csv(f"scripts/analysis/_export/{linea}_raw.csv", index=False)
```

| Linea | Shape | NaN totali | `y` NaN | Base rate Over (`y.mean()`) |
|---|---|---:|---:|---:|
| `corners_line_8_5` | (8872, 1167) | 6.150.497 | 0 | 0,6115 |
| `corners_line_9_5` | (8872, 1167) | 6.150.497 | 0 | 0,4973 |
| `corners_line_10_5` | (8872, 1167) | 6.150.497 | 0 | 0,3867 |
| `corners_line_11_5` | (8872, 1167) | 6.150.497 | 0 | 0,2809 |

Le 4 linee condividono la stessa matrice di feature (1166 colonne feature
+ `y`), stesso NaN totali (colonne feature identiche indipendentemente
dalla linea, solo il target cambia): coerente con quanto documentato nel
builder (`build_corners_frame_from_records` calcola tutte le linee in un
colpo solo). `y` non ha mai NaN: il target reale (conteggio corner totali
> linea) e' sempre calcolabile per le partite `FT`. Base rate monotona e
decrescente all'aumentare della linea (61%→50%→39%→28%), come atteso.

4 CSV scritti in `scripts/analysis/_export/`:
`corners_line_{8_5,9_5,10_5,11_5}_raw.csv` (~47 MB ciascuno).

## Verifica rapida bookmaker BetMGM/BetRivers/Bovada

Il dataset NON ha colonne per-bookmaker (nessuna colonna `*_BetMGM*` /
`*_BetRivers*` / `*_Bovada*`): le feature quote sono gia' aggregate per
linea/esito (`odds_count_line_*`, `overround_line_*`, quote medie), non
scomponibili per bookmaker senza rifare la query DB — coerente con quanto
richiesto ("non serve rifare la query completa").

Controllo di ordine di grandezza su `odds_count_line_8_5` (righe non-NaN
= 8.853 su 8.872):

| Metrica | Valore |
|---|---:|
| Media | 9,10 |
| Mediana | 8 |
| Min / Max | 2 / 24 |

Dal report precedente (`report_corners_8_5_passo1.md`, query diretta
DB): BetMGM/BetRivers/Bovada erano presenti su **5.363 fixture distinte
ciascuno** con quota Over/Under 8.5 COSTANTE (stessa cifra su tutte le
partite, pattern da placeholder). 5.363 e' vicino al numero di righe con
`odds_count_line_8_5` non-NaN in questo dataset (8.853): questi 3
bookmaker non sono un campione marginale (3 su 19 ≈ 16%), ma coprono
**circa il 60% delle partite** — su una mediana di 8 bookmaker quotanti
per partita, significa che tipicamente **3 degli 8** (~37%) sono queste
quote costanti/placeholder, non 3 su 19. `odds_count_line_*` e
`overround_line_*` (aggregati/medie su tutti i bookmaker disponibili per
la riga) risultano quindi influenzati in modo non marginale da questi 3
bookmaker su gran parte delle partite: da escludere prima di qualunque
training sulle 4 linee, non solo sulla 8.5. Stima di ordine di grandezza,
non un ricalcolo esatto con/senza (fuori scope qui, richiede riquery DB
per-bookmaker).

## Conclusione operativa

- **Passo 1 sbloccato**: bug di naming corretto in
  `filter_market_service.py`, 4/4 estrazioni riuscite, `y` sempre
  popolato, NaN solo sulle feature (stesse ~6,15M su 1166 colonne × 8872
  righe, invariate tra linee).
- Base rate Over: 61% (8.5) → 50% (9.5) → 39% (10.5) → 28% (11.5),
  monotona come atteso.
- BetMGM/BetRivers/Bovada coprono ~60% delle partite con quote
  costanti/placeholder: impatto su `odds_count_line_*`/`overround_line_*`
  stimato rilevante (~3 su una mediana di 8 bookmaker per partita), non
  trascurabile. Esclusione da fare prima del training, con query
  per-bookmaker dedicata (fuori scope di questo passo).
