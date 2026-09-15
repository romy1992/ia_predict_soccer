# Rifacimento di un mercato da zero — procedura

Procedura seguita per Under/Over 1.5 il 2026-09-13 e da ripetere identica per
ogni altro mercato, uno alla volta. Nasce dalla richiesta dell'operatore:
*"ripariamo da 0... una alla volta... prima recuperiamo le feature a db e le
analizziamo facendo EDA"*.

Il principio che tiene insieme tutti i passi: **niente si decide a giudizio,
tutto si misura**. Ogni volta che in questo percorso ho dato per buona
un'intuizione — mia o dell'operatore — i dati l'hanno smentita.

---

## 0. Premessa: perché serve il dato GREZZO

`build_dataset` di default azzera i valori mancanti (`fillna(0)`, e prima
ancora `_safe_float` che converte `None` in `0.0`). Su quel dataset
"mancante" e "zero reale" sono lo stesso numero, quindi **qualunque analisi
sui buchi e' falsata in partenza**.

Usare sempre `build_dataset(market=..., fill_missing=False)` per l'analisi.
Il default resta l'azzeramento per non cambiare i modelli gia' registrati.

Errore realmente commesso: avevo stimato "expected_goals manca sul 46%"
contando gli zeri su un CSV gia' riempito. Sul dato grezzo mancava sull'1.3%.
Il 46% erano zeri veri.

---

## 1. Estrarre il dataset grezzo dal DB

La sessione cloud il Postgres non lo vede: l'estrazione gira sul PC
dell'operatore via bridge, e il CSV viaggia per git.

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
df = FilterMarketService().build_dataset(market="<MERCATO>", fill_missing=False)
df.to_csv("scripts/analysis/_export/<MERCATO>_raw.csv", index=False)
print(df.shape, "NaN totali:", df.isna().sum().sum())
```

**Controllo bloccante**: se i NaN totali sono zero, sta girando il codice
vecchio. Non proseguire.

**Vincolo**: l'app dell'operatore gira in Docker con la cartella del progetto
montata. Mai `git checkout` sulla working copy principale — si cambierebbe il
codice all'app in esecuzione. Usare un git worktree separato.

---

## 2. EDA

```
python scripts/analysis/eda_market.py <MERCATO>
```

Cinque blocchi: struttura e periodo, valori mancanti per colonna, mancanti
aggregati per grandezza, correlazione con il target, ridondanza fra feature.

Cosa cercare, con le soglie decisionali:

- **Mancanti sotto il 2%**: irrilevanti, qualunque trattamento va bene
  (scartare le righe e' la scelta piu' pulita, vedi punto 5).
  **Sopra il 20%**: la colonna va trattata a parte o scartata.
- **Copertura delle quote per esito**: se una feature di quota e' valorizzata
  su molto meno del 100%, ci sono alias non unificati (punto 3).
- **Sbilanciamento del target**: determina la strategia di soglia (punto 6) e
  rende l'accuracy una metrica inutile.
- **Coppie sopra 0.95**: spesso identita' matematiche (`odds_slot_1` **e'**
  `odds_min`; `implied_prob` **e'** 1/`odds_mean`). Tenerle e' dare al modello
  la stessa informazione piu' volte.

---

## 3. Unificare gli alias delle quote

Il provider ha cambiato convenzione di nome a meta' 2025: `alternate over
1.5` e `over 1.5` sono la stessa scommessa in periodi disgiunti. Trattandole
separate, ogni feature resta vuota su meta' delle partite.

Gia' gestito in `_normalize_outcome_name` tramite `_OUTCOME_ALIAS_PREFIXES`.
**Per un mercato nuovo va verificato** che non esistano altri prefissi:

```python
sorted({c[len("odds_mean_"):] for c in df.columns if c.startswith("odds_mean_")})
```

Se compaiono esiti inattesi, prima di unirli **dimostrare che sono lo stesso
evento**: distribuzioni di quota sovrapponibili, correlazione col target
equivalente, e periodi disgiunti (o valori concordi dove coesistono).

Attenzione: su Under/Over 2.5 questo controllo ha rivelato **quote
contaminate** (un "Over 2.5" quotato 18.00, che non esiste). Li' il merge va
sospeso finche' la contaminazione non e' capita.

Dopo la modifica: ri-estrarre e verificare che la copertura sia ~100% e
l'`overround` mediano sia ~1.05 (il margine reale del bookmaker). Un
overround molto piu' alto significa che si stanno sommando esiti di mercati
diversi.

---

## 4. Misurare quanto vale ogni blocco di feature

```
python scripts/analysis/measure_stats_and_ordering.py
```

Si parte dalle sole quote e si aggiunge un blocco per volta, guardando
l'AUC in CV walk-forward. Serve a rispondere con i numeri a "questa famiglia
di statistiche serve?".

Su Under/Over 1.5: i cartellini valgono quanto i tiri (+0.0121 contro
+0.0124), contro l'intuizione di entrambi. E tutte le 54 statistiche insieme
non fanno meglio dei soli tiri: **i blocchi sono intercambiabili**, misurano
la stessa cosa sottostante.

Lo stesso script misura anche l'effetto dell'ordinamento temporale: validare
a caso invece che per data gonfia l'AUC di +0.0179, **piu' dell'intero
contributo reale delle statistiche**. La CV del progetto
(`_build_temporal_cv`) fa gia' la cosa giusta.

---

## 5. Confrontare i set candidati

```
python scripts/analysis/compare_feature_sets_over_1_5.py
```

Stesso modello, stesso split, stesso seed su ogni configurazione: l'unica
variabile e' il set di feature. Serve a **ordinare** i candidati, non a
giudicarli in assoluto (niente grid search ne' ensemble).

Includere sempre:
- il set attuale di produzione, come riferimento;
- un baseline con la sola quota media, che dice quanta parte del segnale e'
  semplicemente il prezzo del bookmaker;
- una configurazione che isola il solo cambio del blocco quote, a parita' di
  statistiche.

Su Under/Over 1.5 ha vinto `quote + tiri + disciplina` (33), che batte le 60
complete: possesso, xG e corner aggiungevano rumore.

**Righe incomplete**: scartarle, non riempirle. Riempire con la mediana
dell'intero dataset introduce leakage temporale, e lo `StackingClassifier`
della pipeline ha `passthrough=True` — gira le feature grezze al modello
finale scavalcando gli imputer, quindi con NaN va proprio in errore.

---

## 6. Curva precisione/volume, prima di parlare di soldi

Su un mercato sbilanciato l'accuracy non significa nulla: con base rate 77%,
dire sempre "Over" da' gia' il 77% di precisione. **La soglia e' la leva**,
non il bilanciamento delle classi.

SMOTE e `class_weight='balanced'` spingono verso la classe MINORITARIA:
se l'esito che interessa e' quello maggioritario fanno il contrario di quello
che serve. Verificato due volte: su 2.5 il peso doppio sull'Over produceva
recall 99% con precisione pari al caso.

Il numero che conta non e' la precisione ma il **ROI**, perche' alzando la
soglia si scelgono partite con quota piu' bassa e il guadagno si annulla:

```
soglia 0.85 — precisione 88.0% (contro 76.6%), ma quota media da 1.264 a 1.134
   ROI alla quota media      -0.3%
   ROI alla quota migliore   +2.0%
```

Riportare sempre: intervallo di confidenza bootstrap, tenuta per semestre, e
il fatto che la soglia scelta a posteriori gonfia il risultato.

---

## 7. Training con la pipeline completa

```python
train_market(market="<MERCATO>", feature_columns=<SET SCELTO>, save_model=True)
```

Otto passaggi, invariati: dataset, ordinamento temporale, split feature/target,
CV walk-forward, grid search su 3 candidati, ensemble voting/stacking sui 2
migliori, champion per `selection_score`, calibrazione.

Dalla sessione cloud si sostituisce la sorgente dati col CSV esportato
(`patch.object(FilterMarketService, "build_dataset", return_value=df)`),
senza toccare la logica di training.

Guardare sempre la calibrazione: su Under/Over 1.5 ha portato l'ECE da 0.1437
a 0.0164. Per una strategia a soglia e' il guadagno piu' importante — senza,
"0.85" non significa davvero 85%.

---

## 8. Promozione e consegna

```python
ModelRegistry().promote_with_policy(run_id=..., to_stage="production",
                                    reason=..., actor="operator_request")
```

Mai forzare il gate. Il confronto con la production precedente deve passare.

Il registry di questa sessione e' effimero e non comunica con la macchina
dell'operatore: la consegna e' un pacchetto con .pkl zippati + le righe di
registry da APPENDERE (mai sovrascrivere).

Due accortezze imparate sul campo:

1. **Percorsi container.** L'app gira in Docker con
   `./best_models:/app/best_models`: nel registry va scritto
   `/app/best_models/...`, non il percorso Windows. Un percorso host produce
   "File modello non trovato" al primo utilizzo.
2. **Nome file distinto.** Mai riusare il nome del .pkl in produzione:
   sovrascrivendolo si perde il rollback, e la vecchia riga di registry
   punterebbe in silenzio al modello nuovo.
