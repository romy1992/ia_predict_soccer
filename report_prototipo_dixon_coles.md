# Prototipo: modelli generativi sui gol vs classificatori binari (2026-09-21)

Indagine chiesta dall'operatore: *"si possono creare algoritmi custom con
formule specifiche che estendano il base estimator? i soliti random forest
o logistic non sono idonei al 100% per casi come questo?"*

Codice riproducibile: `scripts/analysis/prototipo_dixon_coles.py`.
**Nessun modello e' stato registrato o promosso**: e' un prototipo.

---

## L'idea in una riga

I modelli attuali trattano ogni mercato come una domanda si'/no indipendente
e addestrano un classificatore per ciascuna. Un modello generativo stima
invece **quanti gol segna ciascuna squadra**, e legge ogni mercato come una
somma di celle della matrice dei punteggi — un fit solo, tutti i mercati.

Esempio con le medie reali di questo progetto (casa 1,54 – trasferta 1,26):

```
            0      1      2      3      4
casa 0 | 6.08   7.66   4.83   2.03   0.64
casa 1 | 9.36  11.80   7.43   3.12   0.98
casa 2 | 7.21   9.09   5.72   2.40   0.76
```

| Mercato | Come si ottiene | Risultato |
|---|---|---|
| Over 2.5 | celle con casa+ospite ≥ 3 | 52,4% |
| Over 1.5 | celle con casa+ospite ≥ 2 | 76,2% |
| Goal/Goal | celle con entrambi ≥ 1 | 55,8% |
| 1 (casa) | celle con casa > ospite | 43,3% |

L'Over 1.5 derivato cosi' da 76,2% contro un base rate reale del 77,0%,
senza che il modello sia mai stato addestrato su quella domanda.

## Perche' la Poisson e' giustificata su questi dati

Totale gol: **media 2,797, varianza 2,824 → rapporto 1,010**. Equidispersione
quasi perfetta, il test da manuale per una Poisson (se la varianza fosse
molto maggiore della media servirebbe una Binomiale Negativa).

Il **rho di Dixon-Coles** (correzione sui punteggi bassi 0-0/1-0/0-1/1-1)
stimato per massima verosimiglianza risulta **negativo e stabile su ogni
fold di ogni test eseguito**: da −0,05 a −0,12. Il modello ritrova da solo
nei dati l'effetto che la letteratura calcistica documenta dal 1997.

---

## Risultato principale

Stesse identiche 42 feature per ogni modello (quote 1X2 + Under/Over +
statistiche squadra), 15.290 partite, CV temporale a 5 fold, calibrazione
isotonica su tutti. Unica variabile: il modo di modellare.

| Mercato | Dixon-Coles | logistic | random forest | RF + feature DC |
|---|---:|---:|---:|---:|
| Over 1.5 | 0,5736 | 0,5733 | 0,5793 | **0,5850** |
| Over 2.5 | 0,5755 | 0,5760 | 0,5869 | **0,5880** |
| Over 3.5 | 0,5838 | 0,5851 | **0,5971** | 0,5944 |
| Goal/No Goal | 0,5354 | 0,5323 | 0,5311 | **0,5373** |
| 1X2 (casa) | 0,7121 | 0,6885 | **0,7176** | 0,7174 |

*(AUC out-of-fold. Il Dixon-Coles produce tutte e cinque le colonne da **un
solo fit**; gli altri richiedono un fit per mercato.)*

**Violazioni di monotonia** sulla scala Over (P(Over 1.5) deve essere ≥
P(Over 2.5) ≥ P(Over 3.5)), su 12.740 partite:

| modello | violazioni |
|---|---:|
| dixon_coles | **0** (garantite per costruzione) |
| logistic | 16 |
| random forest | 0 |
| RF + feature DC | 0 |

## Conclusioni

**1. Il random forest resta il piu' accurato.** Vince 3 mercati su 5 in
purezza, con margini piccoli ma sistematici (0,005-0,013 di AUC). La
risposta onesta alla domanda iniziale e': no, non sono "inadatti" — sui
questi dati sono tuttora la scelta migliore sull'accuratezza pura.

**2. Il Dixon-Coles non e' piu' preciso, ma non e' inferiore.** Resta entro
0,01 di AUC ovunque pur essendo vincolato a spiegare tutto tramite i gol, e
**vince su goal/no goal**. Batte nettamente la logistic sull'1X2 (0,7121 vs
0,6885). I suoi vantaggi reali sono strutturali:
- un modello invece di cinque;
- coerenza fra mercati garantita per costruzione, non ereditata dalle quote;
- **copertura**: puo' prezzare il goal/no goal su **12.765 partite contro le
  805** del classificatore dedicato (16x), perche' non dipende dalle quote
  gng — vedi il punto 4.

**3. L'ibrido e' la configurazione migliore.** Dare all'RF le stime λ_casa/
λ_trasferta e le probabilita' derivate dal DC come feature aggiuntive vince
su 3 mercati su 5 (il migliore in assoluto su Over 1.5, Over 2.5 e goal/no
goal) ed e' neutro sugli altri due, migliorando comunque il logloss
sull'1X2 (0,6141 → 0,6102). I guadagni sono modesti (+0,006 di AUC sui
mercati dove vince) ma nell'ordine di grandezza che in questo progetto ha
gia' giustificato promozioni.
*Nota metodologica*: le feature DC per il training sono generate in
cross-validation interna a 3 fold, mai in-sample — senza questa accortezza
il test risulterebbe falsamente ottimista.

**4. Scoperta collaterale sui dati (indipendente dal modello).** Nel dataset
goal/no goal le quote lato "goal" (`prob_norm_goal`, `odds_mean_goal`,
`odds_std_goal`) esistono solo su **969 righe su 15.321 (6,3%)**, mentre il
lato "no_goal" c'e' sempre. E' l'impronta del bug di case-sensitivity Yes/No
pre 2026-09-08, ancora presente nell'export. Significa che il modello
goal_no_goal in produzione lavora su una base di ~1.000 partite, non 15.000.
Da verificare separatamente se sia il caso di rigenerare quell'export.

## Cosa NON dimostra questo prototipo

- Non dimostra che il Dixon-Coles debba sostituire qualcosa in produzione:
  contro un RF ben tarato non passerebbe il gate di promozione sulla
  maggioranza dei mercati.
- Il confronto usa feature identiche per tutti, non la ricetta esatta di
  ciascun modello in produzione: i valori assoluti non sono direttamente
  confrontabili con le metriche del registry.
- L'ibrido e' stato provato con un RF a iperparametri fissi, non con la
  grid search della pipeline reale.

## Se si volesse portarlo avanti

L'unico candidato con un caso concreto e' **l'ibrido su goal/no goal**: e'
il mercato dove vince, dove il segnale e' piu' scarso, e dove la copertura
16x risolverebbe un problema di dati reale. Servirebbe rifarlo dentro
`train_market()` con la grid search vera e il confronto sul `selection_score`
contro la production attuale.
