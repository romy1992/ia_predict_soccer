# Mercati `corners_line_{8_5,9_5,10_5,11_5}` — Passo 1 (dato grezzo dal DB)

Dati letti dal DB reale (stesso `DATABASE_URL` usato dall'app), sola
lettura: nessuna quota modificata, nessun modello riaddestrato, nessuno
script modificato (nessun bypass del blocco riportato sotto).

Preliminare: `git pull` eseguito, `HEAD` a `1b6dd50` ("build_dataset
supporta i mercati a linea (corners/cards), con NaN preservati") — il
commit atteso con il fix.

## Passo 1 — estrazione grezza: BLOCCATA (bug nel fix di `1b6dd50`)

Il comando richiesto

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
for linea in ("corners_line_8_5", "corners_line_9_5", "corners_line_10_5", "corners_line_11_5"):
    df = FilterMarketService().build_dataset(market=linea, fill_missing=False)
    df.to_csv(f"scripts/analysis/_export/{linea}_raw.csv", index=False)
    print(linea, df.shape, "NaN totali:", df.isna().sum().sum())
```

fallisce gia' sulla prima linea (`corners_line_8_5`) con:

```
Traceback (most recent call last):
  File "<string>", line 4, in <module>
    df = FilterMarketService().build_dataset(market=linea, fill_missing=False)
  File "C:\Users\trott\git\ia_predict_soccer\src\service_ia\training\market_service\filter_market_service.py", line 484, in build_dataset
    return self._build_line_dataset(market=market, seasons=seasons, fill_missing=fill_missing)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\trott\git\ia_predict_soccer\src\service_ia\training\market_service\filter_market_service.py", line 538, in _build_line_dataset
    raise ValueError(f"Colonna target {colonna_y} assente nel frame {famiglia}")
ValueError: Colonna target y_8_5 assente nel frame corners
```

**Nessun CSV prodotto** per nessuna delle 4 linee: il ciclo si e' fermato
al primo errore, come da controllo bloccante richiesto (nessuna delle 4
`corners_line_*_raw.csv` esiste in `scripts/analysis/_export/`).

### Causa (verificata leggendo il codice, non modificato)

`_build_line_dataset` (`filter_market_service.py`, righe 503-542) estrae
`linea_label` dal nome del mercato via regex:

```python
match_linea = re.match(r"^(corners|cards)_line_(\d+_\d+)$", market)
famiglia, linea_label = match_linea.group(1), match_linea.group(2)
...
colonna_y = f"y_{linea_label}"   # per market="corners_line_8_5" -> "y_8_5"
```

Ma il frame prodotto da `build_corners_frame_from_records`
(`src/ml/markets/corners/corners_market.py`) nomina le colonne target
usando `_line_label()` (riga 66-67):

```python
def _line_label(line: float) -> str:
    return f"line_{str(float(line)).replace('.', '_')}"   # -> "line_8_5"
```

applicata a riga 174: `row[f"y_{_line_label(line)}"] = ...` ->
**`y_line_8_5`**, non `y_8_5`.

Verificato direttamente (sola lettura, nessuna riga scritta su file
sorgenti) costruendo il frame grezzo su un campione di 50 match:

```
columns with y_: ['y_line_8_5', 'y_line_9_5', 'y_line_10_5', 'y_line_11_5']
```

Il fix di `1b6dd50` ha quindi un mismatch di naming: cerca `y_{linea_label}`
(`y_8_5`) mentre la colonna reale e' `y_line_{linea_label}` (`y_line_8_5`).
Sbaglia per tutte e 4 le linee nello stesso modo (stesso pattern di
costruzione nome), quindi il blocco vale per l'intera famiglia
`corners_line_*` (e presumibilmente anche `cards_line_*`, stesso codice
condiviso in `_build_line_dataset`, non verificato qui perche' fuori
scope).

## Bookmaker BetMGM/BetRivers/Bovada — verifica rapida (senza dataset)

Non essendoci un dataset estratto da `build_dataset` su cui calcolare
`odds_count`/overround "con vs senza" questi 3 bookmaker, non e' stato
possibile il confronto quantitativo richiesto (dipende dal Passo 1,
bloccato). Dal report precedente (`report_corners_8_5_passo1.md`, query
diretta Match+Odds sulla linea 8.5) risultava: 19 bookmaker totali con
Over+Under accoppiati, di cui BetMGM/BetRivers/Bovada con overround
COSTANTE (stessa quota Over e stessa quota Under su migliaia di fixture
diverse) — pattern da placeholder, non da mercato reale. Restano da
escludere quando il Passo 1 sara' sbloccato; non riverificati qui per le
altre 3 linee (9.5/10.5/11.5) per lo stesso motivo di blocco.

## Conclusione operativa

- **Passo 1 bloccato per tutte e 4 le linee**: bug di naming in
  `_build_line_dataset` (`y_{linea_label}` invece di
  `y_line_{linea_label}`), introdotto dal fix `1b6dd50`. Nessun CSV
  prodotto (0 file `corners_line_*_raw.csv`).
- Nessuna shape/NaN/base-rate disponibile per nessuna linea: dipende dalla
  correzione del bug di naming sopra, fuori scope per questo incarico
  (nessuna modifica al codice sorgente in questo passo).
- Verifica bookmaker BetMGM/BetRivers/Bovada rimandata: richiede il
  dataset del Passo 1.
