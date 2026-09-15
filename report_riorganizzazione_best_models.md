# Riorganizzazione di `best_models/` — 2026-09-15

Richiesta dell'operatore: mettere in `archivio/` tutti i modelli della vecchia
procedura — compresi `goal_no_goal`, `under_over_4_5`, `cards` e `corners` — e
dare una forma ai tre modelli nuovi, in `best_models/under_over/under_over_<linea>/`,
adattando il codice dove serve perche' le predizioni continuino a funzionare.

## Cosa e' stato consegnato, e cosa no

`best_models/` e' in `.gitignore` e non viaggia per git: i `.pkl` e il registry
vero vivono solo sulla macchina dell'operatore, montati dai container con
`./best_models:/app/best_models`. **Da qui non e' possibile spostare nessun
file**: quello che questo intervento consegna e' lo strumento che fa lo
spostamento, piu' le modifiche al codice perche' il nuovo layout regga.

Lo spostamento va lanciato dall'operatore, dalla radice del repo:

```
python scripts/maintenance/riorganizza_best_models.py            # simulazione: stampa il piano, non scrive
python scripts/maintenance/riorganizza_best_models.py --apply    # esegue
docker restart soccer_api soccer_scheduler                       # dopo l'esecuzione
```

## Il layout di arrivo

```
best_models/
    under_over/
        under_over_1_5/   under_over_1_5_champion_20260914.pkl + il suo calibratore
        under_over_2_5/   under_over_2_5_champion_20260914.pkl + il suo calibratore
        under_over_3_5/   under_over_3_5_champion_20260914.pkl + il suo calibratore
    archivio/             tutto il resto: 4.5, goal_no_goal, corners, cards, ensemble,
                          campioni a 69 feature, qualunque .pkl rimasto nella radice
    registry/             index.jsonl, promotion_history.jsonl, metadati (mai toccata)
```

Sotto `under_over/<mercato>/` finisce **solo il modello in produzione della
procedura nuova** con il suo calibratore: il criterio e' `production` a 33
feature, lo stesso gia' usato da `promuovi_e_archivia.py`. Un mercato ancora a
69 feature non prende una cartella dedicata — il suo modello e' vecchia
procedura e va in archivio come gli altri.

`archivio/` resta piatta di proposito: e' conservazione, i nomi file sono gia'
unici e una gerarchia li' dentro non servirebbe a nessuno.

## Il punto da tenere presente: archiviare non e' ritirare

`under_over_4_5`, `goal_no_goal`, `corners` e `cards` sono ancora **in
produzione** nel registry. Questo intervento sposta il loro file e riscrive la
riga che lo referenzia; **non cambia lo stage**. Continuano a servire
predizioni esattamente come prima, solo da un percorso diverso.

Detto chiaramente perche' la cosa suona strana a leggerla: dopo lo
spostamento, quattro mercati in produzione verranno serviti da `archivio/`.
E' l'effetto voluto della richiesta ("mettimi in /archivio tutti i modelli
che fanno parte della vecchia procedura, compresi..."), ed e' innocuo finche'
il registry e' allineato, ma la cartella non dira' piu' da sola quali modelli
sono vivi. Quando quei mercati verranno rifatti con la procedura nuova
prenderanno anche loro una cartella dedicata e l'anomalia si chiudera' da se'.
Se invece si preferisce lasciarli nella radice finche' non vengono rifatti,
basta non lanciare lo script: nessuna delle modifiche al codice li tocca.

## Come lavora lo script

1. **Legge il registry vero** e individua le production a 33 feature dei tre
   mercati nuovi (modello + calibratore).
2. **Pianifica** lo spostamento di ogni `.pkl` nella radice di `best_models`:
   quei file nella cartella del mercato, tutti gli altri in `archivio/`.
3. **Si ferma se una destinazione e' occupata**, prima di spostare qualunque
   cosa: sovrascrivere il file in archivio significherebbe perdere l'unica
   copia di qualcosa che non si puo' piu' riprodurre.
4. **Copia il registry** in `registry/_backup_<data>_riorganizzazione/`,
   verificando che la copia abbia lo stesso numero di righe dell'originale.
5. **Sposta i file e riscrive le righe insieme**, mai una sola delle due: un
   file spostato con la riga vecchia darebbe "File modello non trovato" al
   primo utilizzo. Vengono aggiornate **tutte** le righe che puntano a un file
   spostato, non solo la prima — fino ai nomi con suffisso data piu' run
   condividevano lo stesso nome file, ed e' il difetto che il 2026-09-14
   aveva lasciato 4 righe rotte.
6. **Normalizza in forma container** i percorsi rimasti assoluti di un'altra
   macchina: il 2026-09-14 erano stati corretti a mano solo i `model_path`,
   mentre `calibrator_path` e `metadata_path` delle tre righe promosse erano
   rimasti con il percorso Windows del worktree di lavoro.
7. **Verifica**: zero righe che puntano a file inesistenti, e caricamento di
   prova dei tre modelli in produzione.

Senza `--apply` non scrive niente: stampa il piano, il numero di campi di
registry che toccherebbe, ed esce. Non cancella mai niente, solo sposta.

## Collisioni di nome (riscontrata sul registry vero, 2026-09-15)

Alla prima esecuzione sui file veri lo script si e' fermato: in `archivio/`
esisteva gia' un `under_over_2_5_champion.pkl` (archiviato il 14/09) e un file
con lo **stesso nome** era ricomparso nella radice, insieme al suo calibratore.
E' l'eredita' del vecchio schema di naming, dove ogni riaddestramento
sovrascriveva `<mercato>_champion.pkl`: due modelli diversi si contendono un
nome.

Fermarsi era giusto — sovrascrivere avrebbe fatto sparire uno dei due senza
che nessuna riga di registry se ne accorgesse — ma fermarsi e basta non
risolve. Ora lo script, quando trova una collisione:

1. **dice cosa sono i due file**: dimensione, data di modifica, e se sono
   byte per byte identici (lo stesso modello copiato due volte) o diversi
   (due modelli distinti). Le due cose si risolvono in modo opposto, quindi
   non indovina;
2. con `--risolvi-collisioni` archivia quello della radice con un nome
   distinto, `nome__<data di modifica>.pkl`. Nessuno dei due file va perso, e
   le righe di registry restano separate: quella che puntava alla radice segue
   il file rinominato, quella che puntava ad archivio non si muove.

```
python scripts/maintenance/riorganizza_best_models.py --apply --risolvi-collisioni
```

Se invece il file nella radice e' un residuo da buttare, lo si cancella a mano
e si rilancia senza il flag: lo script non cancella mai niente per conto suo.

## Modifiche al codice

### Nuovo: `src/service_ia/training/model_paths.py`

Tre punti del codice ricostruivano il percorso di un modello con
`os.path.basename`. Con le sottocartelle quella scorciatoia appiattisce
`under_over/under_over_1_5/m.pkl` in `m.pkl`, e il file non si trova piu'. Il
modulo tiene la regola in un posto solo:

- `to_container_path` — da percorso locale a `/app/best_models/...`,
  **preservando la sottocartella** (si usa quando si SCRIVE nel registry);
- `resolve_model_path` — percorso locale esistente per un modello, o `None`
  (si usa quando si LEGGE per caricare);
- `destination_subdir` — dove salvare un modello nuovo.

`resolve_model_path` prova **sempre per primo il percorso registrato**, che
vince quando il file esiste: il modello servito non cambia mai finche' il
registry e' allineato. Solo se quel file non c'e' piu' cerca lo stesso nome
nelle posizioni previste dal layout (radice, `archivio/`, le tre cartelle di
mercato). Serve a un caso preciso: una riga rimasta indietro dopo uno
spostamento, dove l'alternativa e' una predizione che sparisce in silenzio.

### Serving

`prediction_snapshot_service.py`, `api/main.py` (`POST /predict/{market}`) e
`model_diagnostics_service.py` passano da `os.path.exists(model_path)` a
`resolve_model_path(model_path)`. E' la parte "adattando il codice se serve
per predire": rende il serving indifferente a dove sta fisicamente il file.

### Promozione

- `scripts/analysis/promuovi_mercato.py` salva modello e calibratore
  direttamente in `best_models/under_over/<mercato>/`, e converte i percorsi
  con `to_container_path` invece che con `basename`.
- `SaveLoad` accetta un `filename` con sottocartella (crea la cartella da se':
  gia' lo faceva, ora e' documentato).
- `archivia()` in `scripts/maintenance/promuovi_e_archivia.py` ora guarda
  **tutti i mercati del registry**, non solo i tre rifatti: quello che decide
  e' se il file e' ancora quello in produzione, non di che mercato sia. Prima
  un campione superato di `under_over_4_5` sarebbe rimasto nella radice per
  sempre. Ricostruisce il percorso di partenza dal relativo (non dal nome
  file) e non sovrascrive mai un omonimo gia' in archivio.

### Due script che il nuovo layout avrebbe rotto

- `scripts/fix_registry_model_paths.py` riscriveva ogni `model_path` come
  `<radice>/<nome file>`. Dopo la riorganizzazione quel comportamento
  riporterebbe **tutte** le righe nella radice — disfarebbe lo spostamento e
  romperebbe l'intero registry in un colpo solo. Ora riscrive solo la radice e
  lascia intatta la posizione dentro `best_models`.
- `scripts/analysis/apply_monotonic_to_champions.py` cercava
  `<mercato>_champion.pkl` solo nella radice. Ora risolve il percorso, cosi'
  trova il champion anche in `archivio/` o nella cartella del mercato, e se
  davvero non c'e' lo dice con un errore esplicito invece di un traceback
  generico.

## Verifica fatta qui

`python3 -m pytest tests/service/model_paths_test.py tests/service/riorganizza_best_models_test.py`
→ 31 test, tutti verdi. Coprono: le tre forme di percorso (container, Windows,
relativo), la sottocartella preservata nella conversione, il piano di
spostamento, le righe multiple che condividono lo stesso file, le righe gia' in
archivio lasciate stare, la simulazione che non tocca il disco, la collisione
segnalata prima di spostare, e il giro completo che finisce con zero righe
rotte.

Insieme alle suite che toccano il codice modificato
(`prediction_snapshot_service_test`, `model_diagnostics_service_test`,
`model_registry_test`): 81 test, tutti verdi.

Quello che **non** e' stato verificato qui, e va guardato dopo l'esecuzione
sulla macchina dell'operatore: lo spostamento vero sui file veri, il
caricamento dei modelli veri (lo script lo fa da se' come ultimo passo) e la
dashboard dopo il riavvio dei container.
