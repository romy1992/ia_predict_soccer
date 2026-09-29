# Cards — Verifica champion reali (2026-09-19)

Verifica dei ROI dello Step A (calcolati con un proxy logistic regression) usando invece i **champion reali** salvati come candidati (`best_models/cards/cards_line_X/`). Stessa selezione (stessi dati, stesso seed) riprodotta deterministicamente per ciascuna linea, catturando le probabilità OOF del champion vero + calibrazione, accuracy/confusion matrix a soglia 0.5, e ROI con intervallo di confidenza bootstrap (2000 resample, 95%) sulle soglie UNDER identificate come positive nello Step A.

Nota: `roi`/`intervallo_roi` sono calcolati sull'esito UNDER (`1-y`) con la quota `odds_mean_under`, indicizzando le probabilità OOF (ordinate per indice riga, come restituite da `CalibrationService.calibrate_estimator`) sulle stesse righe del dataframe filtrato.

## cards_line_3_5 — champion: **stacking** (atteso `stacking` → **MATCH**)

AUC 0.6526, ECE 0.0435 → 0.0404. Accuracy@0.5 = 59.3%

```
              precision    recall  f1-score   support
   Under 3.5       0.61      0.38      0.47       637
    Over 3.5       0.59      0.79      0.67       713
```

| Soglia | n | Precisione | ROI | IC95% |
|---|---|---|---|---|
| 0.45 | 243 | 65.4% | +10.0% | [-0.2%, +20.5%] ⚠️ **include zero** |
| 0.40 | 142 | 70.4% | +13.1% | [+0.7%, +24.9%] |
| 0.35 | 57 | 77.2% | +18.7% | [+0.1%, +34.4%] (margine minimo) |

## cards_line_4_5 — champion: **logistic** (atteso `logistic` → **MATCH**)

AUC 0.6520, ECE 0.1077 → 0.0353. Accuracy@0.5 = 63.0%

```
              precision    recall  f1-score   support
   Under 4.5       0.66      0.83      0.74       978
    Over 4.5       0.52      0.31      0.38       592
```

| Soglia | n | Precisione | ROI | IC95% |
|---|---|---|---|---|
| 0.45 | 885 | 70.6% | +5.9% | [+1.5%, +10.2%] |
| 0.40 | 629 | 75.2% | +8.7% | [+3.7%, +13.6%] |
| 0.35 | 442 | 78.1% | +9.6% | [+4.2%, +15.2%] |
| 0.30 | 297 | 83.8% | +14.2% | [+8.4%, +19.8%] |
| 0.25 | 166 | 88.0% | +16.4% | [+9.4%, +22.9%] |
| 0.20 | 103 | 85.4% | +10.2% | [+1.0%, +18.5%] |

Tutte le soglie robuste (IC95% mai include zero).

## cards_line_5_5 — champion: **random_forest_smote** (atteso `random_forest_smote` → **MATCH**)

AUC 0.6190, ECE 0.1741 → 0.0249. Accuracy@0.5 = 75.6%

```
              precision    recall  f1-score   support
   Under 5.5       0.76      1.00      0.86      1073
    Over 5.5       0.00      0.00      0.00       347
```

⚠️ Il modello a soglia 0.5 non predice mai "Over" (confusion matrix `[[1073,0],[347,0]]`) — accuracy 75.6% è solo la baseline della classe maggioritaria, non un segnale informativo. Le soglie basse usate nello Step A (probabilità ≤ soglia → UNDER) restano lo use-case reale.

| Soglia | n | Precisione | ROI | IC95% |
|---|---|---|---|---|
| 0.45 | 1415 | 75.5% | +5.0% | [+1.7%, +8.4%] |
| 0.40 | 1403 | 75.6% | +4.9% | [+1.8%, +8.2%] |
| 0.35 | 1344 | 76.3% | +5.0% | [+1.8%, +8.2%] |
| 0.30 | 985 | 79.7% | +5.1% | [+1.8%, +8.6%] |
| 0.25 | 574 | 81.9% | +3.9% | [-0.2%, +7.9%] ⚠️ **include zero** |
| 0.20 | 201 | 86.6% | +4.2% | [-1.5%, +10.0%] ⚠️ **include zero** |

## cards_line_6_5 — champion: **random_forest_smote** (atteso `random_forest_smote` → **MATCH**)

AUC 0.6686, ECE 0.1892 → 0.0145. Accuracy@0.5 = 85.3%

```
              precision    recall  f1-score   support
   Under 6.5       0.85      1.00      0.92       892
    Over 6.5       1.00      0.03      0.05       158
```

⚠️ Stesso problema di 5_5: a soglia 0.5 il modello predice quasi sempre "Under" (solo 4 "Over" su 158 reali). Accuracy alta ma non informativa a questa soglia.

| Soglia | n | Precisione | ROI | IC95% |
|---|---|---|---|---|
| 0.45 | 1045 | 85.3% | +2.2% | [-0.5%, +4.8%] ⚠️ **include zero** |
| 0.40 | 1043 | 85.2% | +2.1% | [-0.7%, +4.7%] ⚠️ **include zero** |
| 0.35 | 1041 | 85.2% | +2.0% | [-0.7%, +4.6%] ⚠️ **include zero** |
| 0.30 | 1031 | 85.5% | +2.3% | [-0.4%, +4.8%] ⚠️ **include zero** |
| 0.25 | 1006 | 86.2% | +2.8% | [+0.1%, +5.4%] (margine minimo) |
| 0.20 | 888 | 87.3% | +2.8% | [+0.0%, +5.4%] (margine minimo) |

## Conclusione

Tutti e 4 i champion riprodotti coincidono con quelli dello Step B (nessuna discrepanza, selezione deterministica confermata).

Confronto ROI proxy (Step A, logistic) vs champion reale, per linea:

- **3_5 (stacking)**: ROI reale inferiore al proxy su tutte le soglie (es. UNDER 0.45: proxy +14.4% → reale +10.0%). La soglia 0.45, segnata positiva nello Step A, **ora ha IC95% che include lo zero** — non è più un edge statisticamente robusto con il champion vero. Solo 0.40 e 0.35 restano solide (quest'ultima con margine minimo, n=57).
- **4_5 (logistic)**: stesso algoritmo del proxy → numeri quasi identici allo Step A, **tutte le soglie confermate robuste** (IC95% mai include zero).
- **5_5 (random_forest_smote)**: le soglie centrali (0.45–0.30) **confermano** un edge robusto simile al proxy. Le soglie più aggressive (0.25, 0.20), segnate positive nello Step A, **ora hanno IC95% che include lo zero** — campioni piccoli (n=574, n=201) e ROI non distinguibile da zero.
- **6_5 (random_forest_smote)**: il quadro **peggiora sensibilmente**. Le soglie 0.45–0.30, tutte segnate positive nello Step A con ROI ~+2.2/2.3%, **hanno IC95% che include lo zero** con il champion reale — l'edge apparente nel proxy non regge al bootstrap. Solo le soglie più estreme (0.25, 0.20) restano (appena) sopra zero.

**In sintesi**: il proxy dello Step A ha sovrastimato la robustezza dell'edge, soprattutto sulle linee 5_5 e 6_5 e sulle soglie meno popolate. L'unica linea che conferma pienamente il proxy è 4_5 (stesso algoritmo). Prima di qualsiasi promozione, va ristretto il set di soglie operative a quelle con IC95% chiaramente sopra zero:
- 3_5: 0.40 (0.35 solo con margine minimo, n basso)
- 4_5: tutte (0.45–0.20)
- 5_5: 0.45–0.30 (non 0.25/0.20)
- 6_5: 0.25–0.20 soltanto (non 0.45–0.30, nonostante fossero il grosso del volume nello Step A)

Nessuna promozione eseguita in questo giro.
