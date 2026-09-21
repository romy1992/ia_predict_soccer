# Prompt: mercato Corners (linee 8.5/9.5/10.5/11.5) — stato, findings, prossimi passi

> **CORREZIONE (2026-09-19)**: questo file affermava "nessun modello
> corners è mai stato promosso a production" — **falso**, mai verificato
> direttamente sul registry. Controllo reale (`report_verifica_corners_
> production_esistente.md`): **esiste una production attiva dal 2026-09-12**
> per tutte e 4 le linee (stesso batch dei vecchi modelli cards, 73 feature
> incluse le quote legacy aggregate, AUC 0.51-0.54 — un segnale
> debolissimo, probabilmente ancora contaminato dai bookmaker placeholder
> visto che precede il fix del 16/09). Il resto di questo file — l'indagine
> rigorosa del 16/09, la conclusione "il segnale è troppo debole per
> giustificare un NUOVO training" — resta valido: la conclusione doveva
> solo essere "non promuovere un modello NUOVO", non "non c'è nulla in
> production da rivedere". La decisione su cosa fare con la production
> esistente (lasciarla, disattivarla, retrainarla con la stessa pulizia
> bookmaker applicata ai cards) è dell'operatore, non presa qui.

> **STATO (2026-09-16): indagine completata, training NON eseguito.**
> Conclusione: il segnale disponibile è troppo debole per giustificare
> training + promozione in produzione su nessuna delle 4 linee. Questo file
> documenta cosa è stato fatto, i numeri esatti, e le opzioni per chi
> volesse riprendere il lavoro (operatore o sessione futura).

---

## 1. Contesto e decisione architetturale

Richiesta operatore: rifare il mercato corner con la stessa procedura già
usata per goal_no_goal e Under/Over (EDA → misura contributo feature →
confronto set candidati → training → curva precisione/ROI → promozione),
partendo dalla linea 8.5.

**Domanda posta esplicitamente**: un modello per esito/linea o uno unico?
Risposta data (confermata, mai smentita dai dati raccolti): **un modello per
linea**, non un modello unico multi-classe. Motivi:
- i base rate per linea sono molto diversi e monotoni (61% → 50% → 39% → 28%
  su 8.5→9.5→10.5→11.5): un modello unico dovrebbe comunque emettere 4
  probabilità coerenti tra loro, complicando la calibrazione senza vantaggi
  chiari;
- esiste già un layer di proiezione monotona in fase di serving (le
  probabilità delle 4 linee vengono forzate coerenti tra loro all'atto
  della predizione), quindi il problema di coerenza è già risolto altrove;
- `build_corners_frame_from_records()` calcola comunque le feature per
  tutte e 4 le linee in un solo passaggio (cambia solo la colonna target),
  quindi il costo di estrazione è identico sia con 1 che con 4 modelli — il
  "farne uno solo" non avrebbe nemmeno risparmiato l'estrazione dati.

## 2. Percorso seguito, in ordine cronologico

1. **Bug propedeutico**: `FilterMarketService.build_dataset()` non
   supportava affatto i mercati a linea (`corners_line_*`, `cards_line_*`,
   definiti in `LINE_MARKETS` in `filter_market_service.py` ma mai
   collegati). Prima estrazione bloccata con
   `ValueError: Mercato non supportato: corners_line_8_5`.
   **Fix**: aggiunto `_build_line_dataset()` che instrada verso
   `build_corners_frame_from_records()` / `build_cards_frame_from_records()`
   e isola la colonna target della linea richiesta (`y_line_<linea>`).
   Aggiunto anche il parametro `fill_missing` ai due builder (prima
   azzeravano sempre i NaN, impedendo l'EDA sul dato grezzo — vedi §4 della
   procedura generale più sotto sul perché questo è bloccante).

2. **Secondo bug, scoperto durante l'estrazione**: il fix sopra usava
   `f"y_{linea_label}"` come nome colonna target, ma il builder reale la
   chiama `f"y_line_{linea_label}"` — mismatch di naming, corretto in
   `filter_market_service.py:536`.

3. **Prima estrazione completa (4 linee in un colpo solo)**: shape
   `(8872, 1167)` per ciascuna linea, NaN solo sulle feature (mai su `y`),
   base rate monotona come atteso (61%/50%/39%/28%).

4. **Scoperta contaminazione bookmaker**: analisi AUC preliminare su 8.5 con
   feature quote dava solo ~0.54, molto sotto lo 0.59-0.61 visto sugli altri
   mercati rifatti. Causa trovata con una query diretta: **BetMGM,
   BetRivers, Bovada** quotano Over/Under 8.5 con **lo stesso valore
   esatto** su migliaia di fixture distinte (es. BetMGM sempre 1.57/2.20 su
   5.363 fixture) — pattern da dato placeholder/sintetico, non quote di
   mercato reali. Coprono circa il 60% delle partite quotate, tipicamente
   3 bookmaker su una mediana di 8 per partita: non un rumore marginale.
   Il filtro anomalie esistente (overround < 1 o rapporto > 3) non li
   intercetta, perché il loro overround è plausibile — solo costante.

5. **Fix contaminazione**: aggiunto `CONTAMINATED_BOOKMAKERS = frozenset({
   "BetMGM", "BetRivers", "Bovada"})` in `corners_market.py`, applicato
   come `exclude_bookmakers` a `_extract_line_specific_odds_features()`
   (nuovo parametro in `filter_market_service.py`) **solo per il mercato
   corners** (non verificato/applicato a cards). Ri-estrazione confermata:
   `odds_count_line_8_5` medio 9,10 → 5,62, mediana 8 → 2.

6. **Misura AUC/accuracy/confusion matrix su dato pulito, tutte e 4 le
   linee** — risultati in §3.

7. **Verifica "play style" (possesso, passaggi, xG)**: richiesta esplicita
   dell'operatore dopo aver visto i numeri deboli. Trovato che queste
   feature **erano già nel dataset** (`mean_ball_possession_*_stat`,
   `mean_passes_*_stat`, `mean_passes_accurate_*_stat`,
   `mean_expected_goals_*_stat`, `mean_goals_prevented_*_stat`,
   copertura 98,6%) — il meccanismo `_extract_mean_features()` in
   `filter_market_service.py` è generico e flatten-izza QUALSIASI chiave
   presente in `Match.mean_statistics`, per qualunque mercato. Non era un
   buco di collegamento, solo una colonna non selezionata nei miei set di
   confronto precedenti (filtravo per substring "shot"/"card"/"foul",
   niente "possession"/"passes"). Testate: **non migliorano il segnale**
   (da sole leggermente peggio della baseline, combinate +0.001/+0.005 di
   AUC — rumore).

8. **Scoperta di una feature realmente mancante**: `Statistics
   .generic_statistics` (JSON grezzo per partita) contiene anche `Cross
   Attacks`, `Counter Attacks`, `Free Kicks`, `Assists`, `Goal Attempts`,
   `Substitutions`, `Throwins` — ma il job che calcola le medie pre-partita
   (`calculate_mean()` in `download_match_service.py`, lista
   `columns_mean` righe 659-664) **non le include mai**: restano salvate
   grezze ma non diventano mai feature pre-match utilizzabili, per NESSUN
   mercato attuale, non solo i corner. `Cross Attacks` è l'unica
   plausibilmente legata ai corner (un cross è spesso l'azione che genera
   un corner) mai provata finora — vedi opzione 1 in §5.

## 3. Risultati numerici finali (dato pulito, post-esclusione bookmaker)

Feature set vincente in tutti i confronti: **quote + corner_dedicated +
tiri + disciplina** (34 feature). Split: `TimeSeriesSplit(n_splits=5)`,
ultimo fold come test held-out.

| Linea | Base rate Over | AUC migliore | Accuracy | Comportamento del modello |
|---|---:|---:|---:|---|
| 8.5 | 61,15% | 0,546 | 0,610 | Collassa su "sempre Over" (recall Under 3%) |
| 9.5 | 49,73% | 0,541 | 0,564 | Segnale debole ma bilanciato (56-57% su entrambe le classi) |
| 10.5 | 38,67% | 0,548 | 0,597 | Collassa su "sempre Under" (recall Over 14%) |
| 11.5 | 28,09%* | 0,574 | 0,664 | Collassa su "sempre Under" (recall Over 10%) |

\* 11.5: quote disponibili solo sulle ultime ~3.790/8.872 partite (righe
2672-8871 in ordine cronologico) — un bookmaker ha iniziato a quotarla solo
di recente. AUC/accuracy calcolati sul sottoinsieme con quota disponibile,
non sull'intero storico.

Confusion matrix e classification report completi (per chi vuole i numeri
esatti, non solo il riassunto) sono nella cronologia della sessione che ha
prodotto questo file; i CSV sorgente sono in
`scripts/analysis/_export/corners_line_{8_5,9_5,10_5,11_5}_raw.csv`
(post-esclusione bookmaker, già nel repo).

**Confronto con lo standard del progetto**: gli altri mercati rifatti con
questa stessa procedura (Under/Over 1.5/2.5/3.5, goal_no_goal) hanno tutti
raggiunto AUC 0,59-0,61 prima di essere promossi. Nessuna linea corner ci
si avvicina.

## 3-bis. Verifica ROI sui dati puliti (2026-09-21) — CHIUDE LA QUESTIONE

Prima di rifare i modelli con la pulizia bookmaker applicata (richiesta
operatore "rifacciamoli con dati puliti"), e' stata eseguita la stessa
verifica ROI usata per i cards: soglie + IC bootstrap 95% (2000 resample)
su probabilita' OOF calibrate, dati post-esclusione BetMGM/BetRivers/
Bovada, righe filtrate su quota reale, quote-only per esito. Due modelli
(logistic e random forest), entrambe le direzioni, tutte e 4 le linee.
Criterio: procedere al training solo con almeno una soglia n>=50 e IC
interamente sopra lo zero.

**Esito: ZERO soglie robuste. E molte robustamente NEGATIVE.**

| Linea | Caso | n | ROI | IC 95% |
|---|---|---:|---:|---|
| 8.5 | OVER >= 0,55 | 3.846 | −7,7% | [−10,3%, −4,9%] |
| 8.5 | OVER >= 0,60 | 2.865 | −8,7% | [−11,8%, −5,5%] |
| 9.5 | UNDER <= 0,45 (rf) | 1.316 | −5,8% | [−10,4%, −0,8%] |
| 10.5 | UNDER <= 0,45 | 4.155 | −2,5% | [−4,8%, −0,2%] |
| 10.5 | UNDER <= 0,25 | 338 | −8,2% | [−15,2%, −1,4%] |
| 11.5 | UNDER <= 0,45 | 1.825 | −6,6% | [−9,4%, −3,9%] |
| 11.5 | UNDER <= 0,40 | 1.767 | −6,1% | [−8,8%, −3,4%] |

**Il dato che spiega tutto**: i ROI si concentrano fra −3% e −9%, cioe'
l'ordine di grandezza del MARGINE DEL BOOKMAKER. E' la firma di "nessun
segnale reale": il modello non sbaglia le partite piu' del caso, semplicemente
ogni scommessa lascia sul piatto il vig.

**Caso istruttivo**: la linea 11.5 con logistic ha l'AUC piu' alta di tutte
(0,5906 — superiore a diversi modelli promossi in questo progetto) e perde
comunque su OGNI soglia, tutte con IC sotto zero. AUC e profitto sono due
cose diverse: si puo' ordinare le partite meglio del caso e perdere lo
stesso contro il margine.

Script: `/scratchpad` della sessione (non committato); la logica e'
identica allo Step A dei cards, vedi `report_cards_training_step_a_b.md`.

## 4. Conclusione operativa

**Non procedere con training/promozione** su nessuna delle 4 linee corner
nello stato attuale dei dati. La debolezza è strutturale (le feature
aggregate pre-partita disponibili — quote, tiri, cartellini, possesso,
passaggi, xG — non hanno un legame forte col numero di corner), non un
artefatto di data quality: la pulizia dei bookmaker contaminati non ha
cambiato l'AUC in modo apprezzabile, e aggiungere le feature "play style"
già presenti non ha aiutato.

L'unico comportamento non-degenere è sulla linea 9.5 (la più vicina al 50%
di base rate): lì il modello discrimina davvero, anche se debolmente
(precision/recall ~56-57% su entrambe le classi). Le altre 3 linee, essendo
sbilanciate, fanno collassare il classificatore sulla classe maggioritaria
— un problema di soglia/bilanciamento più che di segnale puro, ma comunque
non sufficiente a battere lo standard del progetto.

**DECISIONE FINALE (2026-09-21)**: alla luce della verifica ROI del §3-bis,
il mercato corner è chiuso. Non solo non si promuovono modelli nuovi: i 4
modelli del 2026-09-12 sono stati **retrocessi da `production` a `retired`**
su richiesta esplicita dell'operatore. Motivo: erano addestrati su dati
CONTAMINATI (precedono il fix bookmaker del 16/09) con AUC 0,51-0,54, e
ora sappiamo che nemmeno con dati puliti e modelli migliori il mercato
batte il margine del bookmaker — quindi quei modelli stavano quasi
certamente facendo perdere soldi a chi ne seguiva i segnali. Nessuna
previsione è meglio di una previsione sistematicamente perdente.

Conseguenza automatica: senza una production per questi mercati,
`evaluate_line_market_signal()` non riceve più un `p_over` reale e le
soglie corner in `line_market_signal_policy.py` diventano inerti da sole
(restano nel file come riferimento storico, vedi il commento lì).

## 5. Cosa si può ancora provare (nessuna garantita, in ordine di costo)

1. **Aggiungere `Cross Attacks` (e forse `Counter Attacks`) alle feature
   pre-match.** Costo: modificare `columns_mean` in
   `download_match_service.py`, poi **ribackfillare** `mean_statistics`
   per tutte le partite storiche di tutte le squadre
   (`calculate_mean(force_mean=True)`) — un ricalcolo massivo, non
   istantaneo, da fare sul PC dell'operatore (unico posto con accesso al
   DB reale) via bridge. Beneficio atteso: incerto — possesso/passaggi/xG
   già testati non hanno aiutato, ma "cross attempts" è più direttamente
   causale sui corner di quanto lo sia "possesso palla" in generale.

2. **Modellare il conteggio totale corner come regressione (es. Poisson)
   invece di classificazione binaria per soglia.** Non ancora esplorato in
   questo progetto per nessun mercato a linea. Un solo modello di
   regressione sul totale corner permetterebbe di derivare la probabilità
   per QUALUNQUE soglia (8.5, 9.5, 10.5, 11.5, e altre) dalla stessa
   distribuzione stimata, invece di 4 classificatori indipendenti — più
   efficiente se funzionasse, ma richiede una pipeline di training diversa
   da quella esistente (`train_market()` è pensata per classificazione
   binaria).

3. **Fermarsi definitivamente sui corner** e considerare il mercato non
   rifattibile con i dati oggi disponibili. Riprendere solo se in futuro
   arrivano nuove fonti dati (es. tracking-based stats, non disponibili
   con il provider attuale).

**Raccomandazione**: opzione 3 salvo che l'operatore non voglia investire
il costo di un backfill (opzione 1) per un beneficio incerto. Opzione 2 è
un cambio di architettura più grande, da valutare solo se il mercato
corner viene considerato prioritario a prescindere dal costo/beneficio.

## 6. File toccati in questo lavoro (per chi riprende)

- `src/service_ia/training/market_service/filter_market_service.py`:
  `_build_line_dataset()` (dispatch `LINE_MARKETS` → builder corners/cards),
  `_extract_line_specific_odds_features()` con parametro
  `exclude_bookmakers`.
- `src/ml/markets/corners/corners_market.py`: parametro `fill_missing`,
  costante `CONTAMINATED_BOOKMAKERS`.
- `src/ml/markets/cards/cards_market.py`: parametro `fill_missing` (NON ha
  ancora l'esclusione bookmaker — da verificare se cards ha lo stesso
  problema prima di fidarsi delle sue quote).
- CSV puliti: `scripts/analysis/_export/corners_line_{8_5,9_5,10_5,11_5}
  _raw.csv`.
- Report intermedi (cronologia completa passo-passo):
  `report_corners_8_5_passo1.md`, `report_corners_tutte_linee_passo1.md`.
- Commit rilevanti su `main` (già mergiati in `feature/soccer-oracle-v2`):
  `0ebe7ca`, `1b6dd50`, `f1e11a6`, `9404815`, `b8f81b1`, `b378776`.

**Nota per il training, se si procede**: `model_paths.py` non ha ancora una
convenzione di cartella per i corner (`MERCATI_CARTELLA_EVENTO` include
oggi solo `h2h`, `dc`, `goal_no_goal`) — da decidere il naming
(es. `best_models/corners/corners_line_8_5/`) prima di uno script di
promozione, seguendo lo stesso pattern di `promuovi_goal_no_goal.py`.

**Nota bridge**: la promozione di goal_no_goal era finita per errore nel
worktree sbagliato (`ia_predict_soccer_export` invece di
`ia_predict_soccer`, l'unico montato da Docker) e non era visibile
all'app finché non è stata corretta manualmente. Qualunque sessione bridge
futura va istruita esplicitamente: **lavorare SOLO in `ia_predict_soccer`**.

---

# Procedura generale: rifacimento di un mercato, da EDA a training

Questa è la procedura standard usata per ogni mercato rifatto in questo
progetto (Under/Over 1.5/2.5/3.5, goal_no_goal, e l'indagine sui corner
sopra). Fonte canonica, con più dettaglio ed esempi:
`docs/soccer_oracle_v2_detailed/PROMPT_rifacimento_mercato.md` — questo è
un riepilogo autosufficiente per chi non vuole aprire due file.

Principio guida: **niente si decide a giudizio, tutto si misura**. Ogni
volta che un'intuizione (mia o dell'operatore) non è stata verificata sui
dati, si è rivelata sbagliata almeno una volta nel corso del progetto.

## Passo 0 — Perché serve il dato grezzo, non riempito

`build_dataset()` di default azzera i valori mancanti (`fillna(0)`). Su
quel dataset "mancante" e "zero reale" sono lo stesso numero: qualunque
analisi sui buchi è falsata. Usare sempre
`build_dataset(market=..., fill_missing=False)` per l'EDA — il default
resta l'azzeramento per non cambiare i modelli già in produzione.
Errore realmente commesso: stimare "una feature manca sul 46%" contando
zeri su un CSV già riempito, quando sul dato grezzo mancava sull'1,3% — il
46% erano zeri veri, non buchi.

## Passo 1 — Estrarre il dataset grezzo dal DB

La sessione cloud non vede il Postgres reale: l'estrazione gira sul PC
dell'operatore via bridge, il CSV viaggia per git.

```python
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
df = FilterMarketService().build_dataset(market="<MERCATO>", fill_missing=False)
df.to_csv("scripts/analysis/_export/<MERCATO>_raw.csv", index=False)
print(df.shape, "NaN totali:", df.isna().sum().sum())
```

Controllo bloccante: se i NaN totali sono zero, sta girando codice vecchio
(il fillna di default) — non proseguire. Vincolo: mai `git checkout` sulla
working copy principale dell'operatore (l'app gira in Docker con quella
cartella montata) — usare sempre un worktree separato, e **verificare
sempre quale worktree sta usando la sessione bridge**, perché lavorare in
quello sbagliato produce modelli invisibili all'app (vedi §6 sopra).

## Passo 2 — EDA

```
python scripts/analysis/eda_market.py <MERCATO>
```

Cinque blocchi: struttura/periodo, mancanti per colonna, mancanti
aggregati per famiglia di feature, correlazione col target, ridondanza fra
feature. Soglie decisionali:

- Mancanti sotto il 2%: irrilevante, scartare le righe è la scelta più
  pulita (vedi passo 5). Sopra il 20%: la colonna va trattata a parte o
  scartata.
- Copertura delle quote per esito sotto il 100%: probabile alias non
  unificato (passo 3).
- Sbilanciamento del target: determina la strategia di soglia (passo 6) e
  rende l'accuracy grezza una metrica poco informativa.
- Coppie di feature correlate sopra 0,95: spesso identità matematiche
  (es. `odds_slot_1` == `odds_min`) — tenerle entrambe dà al modello la
  stessa informazione due volte, senza guadagno.

## Passo 3 — Unificare gli alias delle quote

Il provider ha cambiato convenzione di nome a metà 2025 (es. `alternate
over 1.5` e `over 1.5` sono la stessa scommessa in periodi disgiunti).
Senza unificarli, ogni feature quota resta vuota su metà delle partite.
Gestito centralmente in `_normalize_outcome_name` /
`_OUTCOME_ALIAS_PREFIXES` (`filter_market_service.py`) — per un mercato
nuovo va verificato che non esistano altri prefissi non ancora gestiti:

```python
sorted({c[len("odds_mean_"):] for c in df.columns if c.startswith("odds_mean_")})
```

Prima di unire due esiti dal nome diverso, dimostrare che sono davvero lo
stesso evento: distribuzioni di quota sovrapponibili, correlazione col
target equivalente, periodi disgiunti (o valori concordi dove coesistono).
**Attenzione**: questo controllo, su mercati diversi, ha rivelato due
pattern di quote contaminate — un "Over 2.5" quotato 18.00 (impossibile) e
i 3 bookmaker placeholder sui corner (§2.4 sopra). In questi casi il merge
va sospeso finché la contaminazione non è capita e trattata.

Dopo la modifica: ri-estrarre e verificare copertura ~100% e overround
mediano ~1,05 (il margine reale del bookmaker) — un overround molto più
alto indica che si stanno sommando esiti di mercati diversi.

## Passo 4 — Misurare quanto vale ogni blocco di feature

```
python scripts/analysis/measure_stats_and_ordering.py
```

Si parte dalle sole quote e si aggiunge un blocco per volta, guardando
l'AUC in CV walk-forward: risponde con i numeri a "questa famiglia di
statistiche serve davvero?" invece di assumerlo. Lezione ricorrente: i
blocchi di statistiche spesso sono **intercambiabili** (misurano la stessa
cosa sottostante) — su un mercato tutte le 54 statistiche insieme non
hanno fatto meglio dei soli tiri.

Lo stesso script misura anche l'effetto dell'ordinamento temporale nella
CV: validare a caso invece che per data ha gonfiato l'AUC di +0,0179 in un
caso — più dell'intero contributo reale delle statistiche aggiuntive. La
CV del progetto (`_build_temporal_cv`) fa già la cosa giusta (walk-forward,
mai shuffle).

## Passo 5 — Confrontare i set di feature candidati

Stesso modello, stesso split, stesso seed su ogni configurazione: serve a
**ordinare** i candidati, non a giudicarli in assoluto (niente grid search
né ensemble a questo stadio). Includere sempre:
- il set attuale di produzione, come riferimento (se il mercato esiste già);
- un baseline con la sola quota media (dice quanta parte del segnale è
  semplicemente il prezzo del bookmaker);
- una configurazione che isola il solo cambio del blocco quote, a parità
  di statistiche.

**Righe incomplete**: scartarle, non riempirle. Riempire con la mediana
introduce leakage temporale, e la pipeline finale (`StackingClassifier`
con `passthrough=True`) gira le feature grezze al modello finale
scavalcando gli imputer — con NaN va in errore, quindi il training reale
richiede comunque righe complete.

**Leakage da controllare sempre**: colonne che contengono il risultato
reale della partita (es. `total_corners`, il conteggio finale) finiscono
facilmente in una selezione per substring insieme alle feature legittime
(es. "corner_kicks_stat") — un AUC improvvisamente vicino a 1.0 è quasi
sempre questo, non un modello buono.

## Passo 6 — Curva precisione/volume, prima di parlare di soldi

Su un mercato sbilanciato l'accuracy non significa nulla: con base rate
77%, dire sempre "Over" dà già il 77% di precisione. **La soglia è la
leva**, non il bilanciamento delle classi — `SMOTE` e
`class_weight='balanced'` spingono verso la classe minoritaria: se l'esito
che interessa è quello maggioritario, fanno il contrario di quello che
serve (verificato: un peso doppio sulla classe maggioritaria ha prodotto
recall 99% con precisione pari al caso).

Il numero che conta è il **ROI**, non la precisione, perché alzando la
soglia si scelgono partite con quota più bassa e il guadagno può
annullarsi:

```
soglia 0.85 — precisione 88.0% (contro 76.6%), ma quota media da 1.264 a 1.134
   ROI alla quota media      -0.3%
   ROI alla quota migliore   +2.0%
```

Riportare sempre: intervallo di confidenza bootstrap, tenuta per semestre
(non solo sull'intero periodo), e il fatto che una soglia scelta a
posteriori sui dati di test gonfia il risultato.

## Passo 7 — Training con la pipeline completa

```python
from src.service_ia.training.train_multi_market import train_market
train_market(market="<MERCATO>", feature_columns=<SET SCELTO>, save_model=True)
```

Otto passaggi fissi, sempre gli stessi: costruzione dataset, ordinamento
temporale, split feature/target, CV walk-forward, grid search su 3
candidati, ensemble voting/stacking sui 2 migliori, selezione champion per
`selection_score`, calibrazione (`CalibrationService`).

Dalla sessione cloud si sostituisce la sorgente dati col CSV esportato
(`patch.object(FilterMarketService, "build_dataset", return_value=df)`),
senza toccare la logica di training vera e propria.

Guardare sempre l'ECE post-calibrazione (es. un caso è passato da 0,1437 a
0,0164): per una strategia a soglia è il guadagno più importante — senza
calibrazione, "0.85" non significa davvero 85% di probabilità reale.

## Passo 8 — Promozione e consegna

```python
ModelRegistry().promote_with_policy(run_id=..., to_stage="production",
                                    reason=..., actor="operator_request")
```

Mai forzare il gate: il confronto con la production precedente deve
passare normalmente. Il registry della sessione cloud è effimero e non
comunica con la macchina dell'operatore — la consegna reale è sempre un
pacchetto di `.pkl` + righe di registry da **appendere** (mai sovrascrivere)
sul worktree reale (`ia_predict_soccer`, mai `ia_predict_soccer_export`).

Due accortezze imparate sul campo, entrambe causa di modelli promossi ma
invisibili all'app finché non corrette:
1. **Percorsi container**: l'app gira in Docker con
   `./best_models:/app/best_models` montato — nel registry va scritto
   `/app/best_models/...`, mai un percorso Windows/host. Usare
   `to_container_path()` / `resolve_model_path()` (`model_paths.py`).
2. **Nome file distinto e cartella corretta**: mai riusare il nome del
   `.pkl` già in produzione (si perde il rollback). Rispettare la
   convenzione cartelle di `model_paths.py`
   (`MERCATI_NUOVA_PROCEDURA` → `best_models/under_over/<mercato>/`,
   `MERCATI_CARTELLA_EVENTO` → `best_models/<mercato>/`, tutto il resto in
   `best_models/` root finché non rifatto) — se il mercato non ha ancora
   una convenzione (come i corner oggi), va decisa e aggiunta a
   `model_paths.py` prima di promuovere.
